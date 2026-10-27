from dataclasses import replace
import json

from fastapi.testclient import TestClient
import httpx
import pytest

from memorydraft.chat.app import create_chat_app, sse
from memorydraft.chat.trace import TraceHistory
from memorydraft.chat.upstream import UpstreamError
from memorydraft.serving.app import create_app
from memorydraft.serving.speculation import payload
from test_chat import config, register, ORIGIN
from test_runtime import engine, request
from test_runtime_cost_tree import tree_engine


@pytest.mark.parametrize("method", ["AR", "DFlash", "ASD", "BudgetTree-ASD"])
@pytest.mark.parametrize("limit", [1, 2, 19])
def test_observer_preserves_output_budget_and_real_commit_paths(tree_engine, method, limit):
    req = request(max_new_tokens=limit)
    expected = tree_engine.generate(req, method)
    observed, commits = [], []
    actual = tree_engine.generate(req, method, on_tokens=commits.extend,
        on_event=lambda value: observed.append(payload(tree_engine, value)))
    assert commits == actual.token_ids == expected.token_ids
    assert actual.asd == expected.asd and actual.counters == expected.counters
    assert actual.prompt_sha256 == expected.prompt_sha256
    assert not observed if method == "AR" or limit == 1 else observed
    verified = [v for v in observed if v["phase"] == "verified"]
    assert [i for v in verified for i in v["committed_ids"]] == (actual.token_ids[1:] if method != "AR" else [])
    history = TraceHistory()
    for draft, done in zip(observed[::2], observed[1::2]):
        assert draft["phase"] == "draft" and done["phase"] == "verified"
        assert draft["round"] == done["round"]
        assert len(done["correction_ids"]) <= 1
        if draft["tree"]:
            tree, path = draft["tree"], done["selected_path"]
            assert [tree["token_ids"][i] for i in path[1:]] == done["committed_ids"][:done["accepted"]]
            assert all(tree["parents"][i] == p for p, i in zip(path, path[1:]))
        else:
            assert draft["proposal_ids"][:done["accepted"]] == done["committed_ids"][:done["accepted"]]
        history.add(draft); history.add(done)
    json.dumps(history.export(), allow_nan=False)


def test_eos_observer_is_clipped_to_actual_output(tree_engine):
    baseline = tree_engine.generate(request(), "BudgetTree-ASD")
    tree_engine.stops = [baseline.token_ids[2]]
    events = []
    result = tree_engine.generate(request(), "BudgetTree-ASD", on_event=events.append)
    assert [i for e in events if e["phase"] == "verified" for i in e["committed_ids"]] == result.token_ids[1:]
    assert result.token_ids[-1] in tree_engine.stops


def test_serving_process_events_are_opt_in_and_ordered(tree_engine):
    body = {"id": "test", "user_id": "u", "order": 1, "messages": [{"role": "user", "content": "w1 w2"}],
            "method": "BudgetTree-ASD", "max_new_tokens": 19, "stream": True, "text_stream": True}
    with TestClient(create_app(lambda: tree_engine, api_key="key")) as client:
        assert client.get("/healthz").status_code == 401
        client.headers["Authorization"] = "Bearer key"
        assert client.get("/healthz").json()["speculation_events"]
        assert "event: speculation" not in client.post("/v1/generate", json=body).text
        response = client.post("/v1/generate", json={**body, "visualize": True})
        assert response.status_code == 200
        blocks = response.text.split("\n\n")
        events = [json.loads(b.split("data: ")[1]) for b in blocks if b.startswith("event: speculation")]
        assert events and [e["phase"] for e in events] == ["draft", "verified"] * (len(events) // 2)
        assert blocks[-2].startswith("event: complete")
        assert client.post("/v1/generate", json={**body, "stream": False, "visualize": True}).status_code == 422


def test_catalog_routes_whitelisted_models_and_saves_bounded_trace(config):
    settings = replace(config, models=({"id": "large", "label": "Large", "url": "http://127.0.0.2"},))
    calls = []
    token = {"id": 9, "text": "你好"}
    def handler(req):
        large = req.url.host == "127.0.0.2"
        if req.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ready", "methods": ["AR"] if large else ["ASD"],
                                             "speculation_events": not large})
        body = json.loads(req.content); calls.append((req.url.host, req.url.path, body))
        if req.url.path == "/v1/token-count":
            return httpx.Response(200, json={"input_tokens": 3, "input_limit": 8192})
        events = ""
        if body["visualize"]:
            for i in range(1, 21):
                events += sse("speculation", {"version": 1, "round": i, "phase": "draft", "proposal": [token], "tree": None})
                events += sse("speculation", {"version": 1, "round": i, "phase": "verified", "accepted": 1,
                    "committed": [token], "correction": [], "selected_path": None, "relaxed": []})
        return httpx.Response(200, text=events + sse("complete", {"text": "你好", "method": body["method"]}),
                              headers={"content-type": "text/event-stream"})
    with TestClient(create_chat_app(settings, transport=httpx.MockTransport(handler)), base_url=ORIGIN) as client:
        assert client.get("/api/models").status_code == 401
        register(client)
        models = client.get("/api/models").json()
        assert models["models"][1]["methods"] == ["AR"] and "url" not in models["models"][1]
        cid = client.post("/api/conversations").json()["id"]; url = f"/api/conversations/{cid}/messages"
        assert client.post(url, json={"content": "x", "model": "large", "method": "ASD"}).status_code == 422
        assert client.post(url, json={"content": "x", "model": "http://evil.example"}).status_code == 422
        answer = client.post(url, json={"content": "x", "method": "ASD", "visualize": True})
        assert "event: speculation" in answer.text
        metadata = client.get(f"/api/conversations/{cid}").json()["messages"][-1]["metadata"]
        assert metadata["speculation"]["round_count"] == 20
        assert len(metadata["speculation"]["rounds"]) == 16 and metadata["speculation"]["truncated"]
        assert client.post(url, json={"content": "y", "model": "large", "method": "AR"}).status_code == 200
        assert calls[-1][0] == "127.0.0.2" and calls[-1][2]["method"] == "AR"
        assert all(set(m) == {"role", "content"} for m in calls[-1][2]["messages"])


@pytest.mark.parametrize("value", [None, {}, {"version": 1, "round": True, "phase": "draft"},
    {"version": 1, "round": 1, "phase": "draft", "proposal": [], "tree": []},
    {"version": 1, "round": 1, "phase": "draft", "proposal": [], "tree": {"parents": [-1, 2], "depths": [0, 1], "tokens": [{"id": 1, "text": "a"}] * 2}},
    {"version": 1, "round": 1, "phase": "verified", "accepted": 1}])
def test_malformed_trace_is_rejected(value):
    with pytest.raises(UpstreamError):
        TraceHistory().add(value)


@pytest.mark.parametrize("models", [(None,), ({"id": []},), ({"id": "x", "url": ORIGIN, "label": "x"},),
    ({"id": "qwen3-8b", "url": ORIGIN, "label": "duplicate"},)])
def test_malformed_catalog_is_a_configuration_error(config, models):
    with pytest.raises(ValueError):
        replace(config, models=models)
