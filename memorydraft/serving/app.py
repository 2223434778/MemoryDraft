from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import queue
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from memorydraft.runtime import GenerationRequest, RequestCancelled
from memorydraft.serving.worker import InferenceWorker, QueueFull


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: StrictStr = Field(max_length=262144)


class GenerateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: StrictStr = Field(min_length=1, max_length=128)
    user_id: StrictStr = Field(min_length=1, max_length=128)
    order: StrictInt = Field(ge=0)
    messages: list[Message] = Field(min_length=1, max_length=128)
    max_new_tokens: StrictInt = Field(default=128, ge=1, le=4096)
    method: StrictStr = "AR"
    stream: bool = False
    timeout_seconds: float = Field(default=60.0, gt=0, le=3600)


def create_app(engine_factory, *, queue_capacity=8, event_capacity=32):
    @asynccontextmanager
    async def lifespan(app):
        engine = await asyncio.to_thread(engine_factory)
        app.state.worker = InferenceWorker(engine, queue_capacity=queue_capacity, event_capacity=event_capacity)
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.worker.close)

    app = FastAPI(title="MemoryDraft", version="0.3.0", lifespan=lifespan)

    @app.get("/healthz")
    async def health(request: Request):
        worker = request.app.state.worker
        return {"status": "ready" if not worker.closing else "closing", "methods": list(worker.engine.methods),
                "synthetic": worker.engine.describe().get("synthetic", False)}

    @app.get("/metrics")
    async def metrics(request: Request):
        return request.app.state.worker.metrics()

    def error_details(job):
        if isinstance(job.error, RequestCancelled):
            timeout = time.perf_counter() >= job.submitted + job.timeout_seconds
            return (504, "Request deadline exceeded") if timeout else (499, "Request cancelled")
        if isinstance(job.error, ValueError):
            return 422, str(job.error)
        return 500, "Generation failed"

    def response(job):
        return {**asdict(job.result), "queue_seconds": job.queue_seconds,
                "server_seconds": job.finished - job.submitted,
                "server_ttft_seconds": (job.queue_seconds + job.result.ttft_seconds
                                        if job.result.ttft_seconds is not None else None)}

    def event(name, value):
        return "event: " + name + "\ndata: " + json.dumps(value, ensure_ascii=False) + "\n\n"

    @app.post("/v1/generate")
    async def generate(body: GenerateBody, request: Request):
        try:
            generation = GenerationRequest(body.id, body.user_id, body.order,
                                           tuple(m.model_dump() for m in body.messages), body.max_new_tokens)
            worker = request.app.state.worker
            job = worker.submit(generation, body.method, timeout_seconds=body.timeout_seconds, stream=body.stream)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except QueueFull as error:
            raise HTTPException(429, str(error), headers={"Retry-After": "1"}) from error
        except RuntimeError as error:
            raise HTTPException(503, "Worker unavailable") from error

        async def wait():
            while not job.done.is_set():
                if await request.is_disconnected():
                    worker.cancel(job)
                    raise HTTPException(499, "Client disconnected")
                if job.should_stop():
                    worker.cancel(job)
                    raise HTTPException(504, "Request deadline exceeded")
                await asyncio.sleep(0.01)

        if not body.stream:
            try:
                await wait()
                if job.error:
                    code, message = error_details(job)
                    raise HTTPException(code, message)
                return response(job)
            finally:
                if not job.done.is_set():
                    worker.cancel(job)

        async def stream():
            try:
                yield event("start", {"request_id": generation.id, "method": body.method})
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        tokens = job.events.get_nowait()
                    except queue.Empty:
                        tokens = None
                    if tokens:
                        yield event("tokens", {"token_ids": tokens})
                    elif job.done.is_set():
                        break
                    else:
                        if job.should_stop():
                            worker.cancel(job)
                            yield event("error", {"code": 504, "message": "Request deadline exceeded"})
                            return
                        await asyncio.sleep(0.01)
                if job.error:
                    code, message = error_details(job)
                    yield event("error", {"code": code, "message": message})
                else:
                    yield event("complete", response(job))
            finally:
                if not job.done.is_set():
                    worker.cancel(job)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
