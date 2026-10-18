import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from memorydraft.runtime.contracts import GenerationRequest, RequestCancelled
from memorydraft.runtime.demo import DemoEngine
from memorydraft.serving.app import create_app
from memorydraft.serving.worker import InferenceWorker, QueueFull


BODY = {"id": "r", "user_id": "u", "order": 10,
        "messages": [{"role": "user", "content": "Hello"}], "max_new_tokens": 32}


def req():
    return GenerationRequest.from_dict(BODY)


class BlockingEngine(DemoEngine):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.closed = False

    def generate(self, request, method, *, on_tokens=None, should_stop=None, **kwargs):
        self.calls += 1
        self.started.set()
        while not self.release.wait(0.005):
            if should_stop and should_stop():
                raise RequestCancelled("Cancelled in test engine")
        return super().generate(request, method, on_tokens=on_tokens, should_stop=should_stop)

    def close(self):
        self.closed = True


def test_bounded_queue_cancels_waiting_request_and_frees_capacity():
    engine = BlockingEngine()
    worker = InferenceWorker(engine, queue_capacity=1)
    try:
        active = worker.submit(req(), "AR")
        assert engine.started.wait(2)
        waiting = worker.submit(req(), "AR")
        with pytest.raises(QueueFull):
            worker.submit(req(), "AR")
        worker.cancel(waiting)
        assert waiting.done.wait(2)
        assert isinstance(waiting.error, RequestCancelled)
        replacement = worker.submit(req(), "AR")
        worker.cancel(active)
        assert active.done.wait(2)
        engine.release.set()
        assert replacement.done.wait(2)
        assert replacement.error is None
        assert engine.calls == 2
        assert worker.metrics()["cancelled"] == 2
        assert worker.metrics()["rejected"] == 1
    finally:
        worker.close()
    assert engine.closed


def test_expired_waiting_job_never_executes_and_shutdown_cleans_up():
    engine = BlockingEngine()
    worker = InferenceWorker(engine)
    active = worker.submit(req(), "AR")
    assert engine.started.wait(2)
    expired = worker.submit(req(), "AR", timeout_seconds=0.01)
    assert not expired.done.wait(0.03)
    engine.release.set()
    assert active.done.wait(2) and expired.done.wait(2)
    assert isinstance(expired.error, RequestCancelled)
    assert engine.calls == 1
    assert worker.metrics()["timed_out"] == 1
    worker.close()
    assert not worker.thread.is_alive()


def test_slow_stream_is_bounded_and_deadline_recovers_worker():
    worker = InferenceWorker(DemoEngine(), event_capacity=1)
    try:
        slow = worker.submit(req(), "AR", timeout_seconds=0.05, stream=True)
        assert slow.done.wait(2)
        assert isinstance(slow.error, RequestCancelled)
        assert slow.events.qsize() == 1
        next_job = worker.submit(req(), "AR")
        assert next_job.done.wait(2)
        assert next_job.error is None
        assert worker.metrics()["timed_out"] == 1
    finally:
        worker.close()


def test_http_json_sse_and_metrics_share_same_verified_tokens():
    with TestClient(create_app(DemoEngine)) as client:
        assert client.get("/healthz").json()["synthetic"]
        response = client.post("/v1/generate", json=BODY)
        assert response.status_code == 200
        payload = response.json()
        assert payload["server_seconds"] >= payload["queue_seconds"]
        streamed = client.post("/v1/generate", json={**BODY, "stream": True})
        assert streamed.status_code == 200
        blocks = [block for block in streamed.text.split("\n\n") if block]
        events = [(b.splitlines()[0][7:], json.loads(b.splitlines()[1][6:])) for b in blocks]
        tokens = [t for name, data in events if name == "tokens" for t in data["token_ids"]]
        assert tokens == payload["token_ids"]
        assert events[-1][0] == "complete"
        assert events[-1][1]["text"] == payload["text"]
        assert client.get("/metrics").json()["completed"] == 2


@pytest.mark.parametrize("change", [{"method": "unknown"}, {"temperature": 0.5}, {"order": True},
                                    {"max_new_tokens": 0}, {"messages": [{"role": "tool", "content": "x"}]}])
def test_http_invalid_requests_are_rejected(change):
    with TestClient(create_app(DemoEngine)) as client:
        assert client.post("/v1/generate", json={**BODY, **change}).status_code == 422


def test_http_queue_overload_returns_429_before_stream_headers():
    engine = BlockingEngine()
    with TestClient(create_app(lambda: engine, queue_capacity=1)) as client:
        worker = client.app.state.worker
        active = worker.submit(req(), "AR")
        assert engine.started.wait(2)
        waiting = worker.submit(req(), "AR")
        overloaded = client.post("/v1/generate", json={**BODY, "stream": True})
        assert overloaded.status_code == 429
        assert overloaded.headers["retry-after"] == "1"
        worker.cancel(waiting)
        worker.cancel(active)


@pytest.mark.parametrize("stream", [False, True])
def test_http_timeout_recovers_for_next_request(stream):
    engine = BlockingEngine()
    with TestClient(create_app(lambda: engine)) as client:
        response = client.post("/v1/generate", json={**BODY, "timeout_seconds": 0.02, "stream": stream})
        if stream:
            assert "event: error" in response.text and '"code": 504' in response.text
        else:
            assert response.status_code == 504
        engine.release.set()
        assert client.post("/v1/generate", json=BODY).status_code == 200


def test_worker_survives_backend_exception_without_leaking_internal_message():
    class Broken(DemoEngine):
        calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("private backend detail")
            return super().generate(*args, **kwargs)

    with TestClient(create_app(Broken)) as client:
        response = client.post("/v1/generate", json=BODY)
        assert response.status_code == 500
        assert "private" not in response.text
        assert client.post("/v1/generate", json=BODY).status_code == 200
        assert client.get("/metrics").json()["failed"] == 1


def test_failed_job_does_not_retain_generation_buffers():
    import weakref

    references = []

    class Buffer:
        pass

    class Broken(DemoEngine):
        def generate(self, *args, **kwargs):
            buffer = Buffer()
            references.append(weakref.ref(buffer))
            raise RuntimeError("Failure with a live request-local buffer")

    worker = InferenceWorker(Broken())
    try:
        job = worker.submit(req(), "AR")
        assert job.done.wait(2)
        assert job.error.__traceback__ is None
        assert references[0]() is None
    finally:
        worker.close()


def test_asgi_disconnect_cancels_running_generation():
    import asyncio

    async def scenario():
        engine = BlockingEngine()
        app = create_app(lambda: engine)
        async with app.router.lifespan_context(app):
            sent = []
            calls = 0

            async def receive():
                nonlocal calls
                calls += 1
                if calls == 1:
                    return {"type": "http.request", "body": json.dumps({**BODY, "stream": True}).encode(),
                            "more_body": False}
                while not engine.started.is_set():
                    await asyncio.sleep(.001)
                return {"type": "http.disconnect"}

            async def send(message):
                sent.append(message)

            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                     "method": "POST", "scheme": "http", "path": "/v1/generate", "raw_path": b"/v1/generate",
                     "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")],
                     "server": ("test", 80), "client": ("test", 1000)}
            await asyncio.wait_for(app(scope, receive, send), timeout=3)
            for _ in range(200):
                if app.state.worker.metrics()["cancelled"]:
                    break
                await asyncio.sleep(.005)
            assert app.state.worker.metrics()["cancelled"] == 1
            assert app.state.worker.metrics()["running"] == 0

    asyncio.run(scenario())


def test_load_test_counts_errors_and_validates_stream(monkeypatch, tmp_path):
    import httpx
    from memorydraft.serving.loadtest import load_test

    client_type = httpx.Client

    def handler(request):
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"synthetic": True})
        index = int(json.loads(request.content)["id"].split("-")[1])
        if index == 1:
            return httpx.Response(429)
        tokens = [1, 2]
        final = [9] if index == 2 else tokens
        text = ('event: tokens\ndata: ' + json.dumps({"token_ids": tokens}) + '\n\n' +
                'event: complete\ndata: ' + json.dumps({"token_ids": final, "queue_seconds": .01}) + '\n\n')
        return httpx.Response(200, text=text)

    monkeypatch.setattr(httpx, "Client", lambda **kw: client_type(transport=httpx.MockTransport(handler), **kw))
    summary = load_test("http://test", BODY, tmp_path / "load", count=3, concurrency=2)
    assert summary["successful"] == 1 and summary["failed"] == 2
    assert summary["synthetic"] is True
    assert summary["p95_seconds"] is None
