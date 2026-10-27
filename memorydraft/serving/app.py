from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import queue
import secrets
import time
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from memorydraft.runtime import GenerationRequest, RequestCancelled
from memorydraft.serving.worker import InferenceWorker, QueueFull
from memorydraft.serving.text import PrefixText
from memorydraft.serving.inline import InlinePresentation
from memorydraft import __version__


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
    text_stream: bool = False
    visualize: bool = False
    timeout_seconds: float = Field(default=60.0, gt=0, le=3600)


class TokenCountBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[Message] = Field(min_length=1, max_length=128)
    max_new_tokens: StrictInt = Field(default=1024, ge=1, le=4096)


def create_app(engine_factory, *, queue_capacity=8, event_capacity=32, api_key=None):
    async def authorize(request: Request):
        if api_key is not None and not secrets.compare_digest(
                request.headers.get("authorization", "").encode(), ("Bearer " + api_key).encode()):
            raise HTTPException(401, "Invalid inference credentials")

    @asynccontextmanager
    async def lifespan(app):
        engine = await asyncio.to_thread(engine_factory)
        app.state.worker = InferenceWorker(engine, queue_capacity=queue_capacity, event_capacity=event_capacity)
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.worker.close)

    app = FastAPI(title="MemoryDraft", version=__version__, lifespan=lifespan,
                  dependencies=[Depends(authorize)])

    @app.get("/healthz")
    async def health(request: Request):
        worker = request.app.state.worker
        description = worker.engine.describe()
        return {"status": "ready" if not worker.closing else "closing", "methods": list(worker.engine.methods),
                "synthetic": description.get("synthetic", False),
                "approximate_methods": description.get("approximate_methods", []),
                "draft_adapter_sha256": (description.get("draft_adapter") or {}).get("sha256"),
                "cost_tree": description.get("cost_tree"), "text_stream": True,
                "speculation_events": description.get("speculation_events", False)}

    @app.post("/v1/token-count")
    async def token_count(body: TokenCountBody, request: Request):
        engine = request.app.state.worker.engine
        messages = [message.model_dump() for message in body.messages]
        if hasattr(engine, "tokenizer"):
            def count():
                return len(engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                           **engine.config.template_options))
            tokens = await asyncio.to_thread(count)
            limit = min(engine.config.max_input_tokens,
                        getattr(engine.target.config, "max_position_embeddings", engine.config.max_input_tokens)
                        - body.max_new_tokens)
        elif engine.describe().get("backend") == "demo":
            tokens, limit = sum(len(m["content"].encode()) for m in messages), 8192
        else:
            raise HTTPException(501, "Token counting requires the engine tokenizer")
        return {"input_tokens": tokens, "input_limit": limit}

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
            if body.visualize and not body.stream:
                raise ValueError("Visualization requires streaming")
            if body.visualize and body.method != "AR" and not worker.engine.describe().get("speculation_events"):
                raise ValueError("This engine does not provide speculation events")
            job = worker.submit(generation, body.method, timeout_seconds=body.timeout_seconds,
                                stream=body.stream, visualize=body.visualize)
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
            decoder = PrefixText(worker.engine) if body.text_stream else None
            presentation = InlinePresentation(worker.engine) if body.visualize and body.text_stream and body.method != "AR" else None
            heartbeat = time.perf_counter()
            try:
                yield event("start", {"request_id": generation.id, "method": body.method,
                    "approximate": body.method in worker.engine.describe().get("approximate_methods", []),
                    "queued": job.started is None})
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        tokens = job.events.get_nowait()
                    except queue.Empty:
                        tokens = None
                    if isinstance(tokens, dict):
                        value = tokens["speculation"]
                        if presentation is not None:
                            value = {**value, "inline": presentation.present(value, decoder.tokens)}
                        yield event("speculation", value)
                    elif tokens:
                        value = {"token_ids": tokens}
                        if decoder is not None:
                            value["text"] = decoder.append(tokens)
                        yield event("tokens", value)
                    elif job.done.is_set():
                        break
                    else:
                        if job.should_stop():
                            worker.cancel(job)
                            yield event("error", {"code": 504, "message": "Request deadline exceeded"})
                            return
                        if time.perf_counter() - heartbeat >= 10:
                            yield event("waiting", {"queued": job.started is None})
                            heartbeat = time.perf_counter()
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
