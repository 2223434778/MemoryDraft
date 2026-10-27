from dataclasses import replace

from fastapi.testclient import TestClient
import pytest

from memorydraft.chat.app import create_chat_app
from memorydraft.chat.config import ChatConfig
from memorydraft.chat.trace import TraceHistory
from memorydraft.chat.upstream import UpstreamError
from memorydraft.runtime.config import RuntimeConfig
from memorydraft.runtime.contracts import GenerationRequest
from memorydraft.runtime.demo import DemoEngine
from memorydraft.serving.app import create_app
from test_chat import config, ORIGIN, register, transport
from test_serving import BODY


@pytest.mark.parametrize("limit", [4097, 8192, 16384])
def test_long_output_is_admitted_consistently(config, limit):
    assert replace(config, max_new_tokens=limit).max_new_tokens == limit
    assert RuntimeConfig(backend="demo", max_new_tokens=limit).max_new_tokens == limit
    assert GenerationRequest.from_dict({**BODY, "max_new_tokens": limit}).max_new_tokens == limit
    with TestClient(create_app(DemoEngine)) as client:
        assert client.post("/v1/generate", json={**BODY, "max_new_tokens": limit}).status_code == 200
        assert client.post("/v1/token-count", json={"messages": BODY["messages"], "max_new_tokens": limit}).status_code == 200


@pytest.mark.parametrize("limit", [0, True, 16385])
def test_output_admission_remains_bounded_and_integer(config, limit):
    with pytest.raises(ValueError):
        replace(config, max_new_tokens=limit)
    with pytest.raises(ValueError):
        RuntimeConfig(backend="demo", max_new_tokens=limit)
    with pytest.raises(ValueError):
        GenerationRequest.from_dict({**BODY, "max_new_tokens": limit})
    with TestClient(create_app(DemoEngine)) as client:
        assert client.post("/v1/generate", json={**BODY, "max_new_tokens": limit}).status_code == 422
        assert client.post("/v1/token-count", json={"messages": BODY["messages"], "max_new_tokens": limit}).status_code == 422


def test_gateway_forwards_long_limit_and_deadline_to_both_context_and_generation(config):
    calls = []
    settings = replace(config, max_new_tokens=16384, timeout_seconds=3600)
    with TestClient(create_chat_app(settings, transport=transport(calls)), base_url=ORIGIN) as client:
        register(client)
        conversation = client.post("/api/conversations").json()["id"]
        result = client.post(f"/api/conversations/{conversation}/messages", json={"content":"问题"})
        assert "event: complete" in result.text
    count = next(body for path,body,_ in calls if path == "/v1/token-count")
    generate = next(body for path,body,_ in calls if path == "/v1/generate")
    assert count["max_new_tokens"] == generate["max_new_tokens"] == 16384
    assert generate["timeout_seconds"] == 3600


def test_trace_after_old_round_limit_is_kept_bounded():
    trace = TraceHistory(capacity=2)
    for number in [4096,4097,16384]:
        draft = {"version":1,"phase":"draft","round":number,"proposal":[],"tree":None}
        trace.add(draft)
        trace.add({"version":1,"phase":"verified","round":number,"accepted":0,
                   "committed":[],"correction":[],"selected_path":None,"relaxed":[]})
    assert trace.export()["round_count"] == 16384 and len(trace.rounds) == 2
    with pytest.raises(UpstreamError):
        trace.add({**draft,"round":16385})


def test_gateway_output_and_timeout_are_configurable_from_environment(config, monkeypatch):
    values = {"MEMORYDRAFT_CHAT_DB":str(config.database), "MEMORYDRAFT_UPSTREAM_URL":config.upstream_url,
              "MEMORYDRAFT_API_KEY":config.upstream_key, "MEMORYDRAFT_INVITE_CODE":config.invite_code,
              "MEMORYDRAFT_PUBLIC_ORIGIN":config.public_origin,
              "MEMORYDRAFT_CHAT_MAX_TOKENS":"16384", "MEMORYDRAFT_CHAT_TIMEOUT_SECONDS":"3600"}
    for key,value in values.items():
        monkeypatch.setenv(key,value)
    settings = ChatConfig.from_env()
    assert settings.max_new_tokens == 16384 and settings.timeout_seconds == 3600
