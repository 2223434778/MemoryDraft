import json

from fastapi.testclient import TestClient
import httpx
import pytest

from memorydraft.runtime.contracts import RequestCancelled
from memorydraft.serving.app import create_app
from memorydraft.serving.llama import LlamaEngine
from test_serving import BODY, req


def native(*, incomplete=False, context=12288):
    def handler(request):
        body = json.loads(request.content or b"{}")
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/props":
            return httpx.Response(200, json={"eos_token": "<eos>", "default_generation_settings": {"n_ctx": context}})
        if request.url.path == "/apply-template":
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            return httpx.Response(200, json={"prompt": "formatted"})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [0] if body["content"] == "<eos>" else [7, 8]})
        if request.url.path == "/detokenize":
            assert 0 not in body["tokens"]
            return httpx.Response(200, json={"content": "你好"[:len(body["tokens"])]})
        assert request.url.path == "/completion" and body["prompt"] == [7, 8]
        assert body["temperature"] == 0 and body["return_tokens"] and not body["cache_prompt"]
        events = [{"tokens": [9], "stop": False}, {"tokens": [10], "stop": False}, {"tokens": [0], "stop": False}]
        if not incomplete:
            events.append({"tokens": [], "stop": True, "stop_type": "eos"})
        return httpx.Response(200, text="".join("data: " + json.dumps(e) + "\n\n" for e in events))
    return httpx.MockTransport(handler)


def test_native_adapter_uses_actual_prompt_tokens_and_stays_ar_only():
    engine = LlamaEngine(transport=native())
    with TestClient(create_app(lambda: engine, api_key="key")) as client:
        assert client.get("/healthz").status_code == 401
        client.headers["Authorization"] = "Bearer key"
        health = client.get("/healthz").json()
        assert health["methods"] == ["AR"] and not health["speculation_events"]
        assert client.post("/v1/token-count", json={"messages": BODY["messages"]}).json()["input_tokens"] == 2
        result = client.post("/v1/generate", json={**BODY, "text_stream": True, "stream": True})
        assert "你好" in result.text and "event: complete" in result.text
        assert '"token_ids": [9, 10, 0]' in result.text and "event: speculation" not in result.text
        assert client.post("/v1/generate", json={**BODY, "method": "ASD"}).status_code == 422


def test_native_cancel_and_incomplete_stream_do_not_return_success():
    engine = LlamaEngine(transport=native())
    try:
        with pytest.raises(RequestCancelled):
            engine.generate(req(), "AR", should_stop=lambda: True)
    finally:
        engine.close()
    engine = LlamaEngine(transport=native(incomplete=True))
    try:
        with pytest.raises(RuntimeError, match="without a final"):
            engine.generate(req(), "AR")
    finally:
        engine.close()


def test_native_backend_cannot_be_redirected_to_external_host():
    with pytest.raises(ValueError, match="loopback"):
        LlamaEngine("http://example.com")


def test_native_context_is_discovered_and_reserves_long_output():
    engine = LlamaEngine(transport=native(context=32768))
    with TestClient(create_app(lambda: engine)) as client:
        response = client.post("/v1/token-count", json={"messages": BODY["messages"], "max_new_tokens": 16384})
        assert response.json() == {"input_tokens": 2, "input_limit": 8192}
        assert client.post("/v1/generate", json={**BODY, "max_new_tokens": 16384}).status_code == 200


def test_native_adapter_rejects_unreported_or_overstated_context():
    with pytest.raises(ValueError, match="actual context"):
        LlamaEngine(transport=native(context=None))
    with pytest.raises(ValueError, match="exceeds"):
        LlamaEngine(context=32768, transport=native(context=12288))
