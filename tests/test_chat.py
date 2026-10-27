import asyncio
from dataclasses import replace
import json
import sqlite3

from fastapi.testclient import TestClient
import httpx
import pytest

from memorydraft.chat.app import COOKIE, create_chat_app, sse
from memorydraft.chat.config import ChatConfig
from memorydraft.chat.security import verify_password
from memorydraft.chat.store import ChatStore, StoreError
from memorydraft.chat.upstream import fit_context
from memorydraft.runtime.demo import DemoEngine
from memorydraft.serving.app import create_app
from memorydraft.serving.text import PrefixText


ORIGIN = "http://127.0.0.1"
KEY = "test-upstream-key-not-a-secret-12345"
INVITE = "test-invitation-not-a-secret"


@pytest.fixture
def config(tmp_path):
    return ChatConfig(tmp_path / "private" / "chat.sqlite3", ORIGIN, KEY, INVITE, ORIGIN)


def register(client, username="alice", **changes):
    response = client.post("/api/register", json={"username": username, "password": "test-password-123",
        "invite_code": INVITE, **changes}, headers={"Origin": ORIGIN})
    assert response.status_code == 200, response.text
    client.headers["X-CSRF-Token"] = response.json()["csrf"]
    client.headers["Origin"] = ORIGIN
    return response


def transport(calls, *, incomplete=False):
    def handler(request):
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ready", "methods": ["BudgetTree-ASD"]})
        body = json.loads(request.content)
        calls.append((request.url.path, body, request.headers.get("authorization")))
        if request.url.path == "/v1/token-count":
            return httpx.Response(200, json={"input_tokens": 30, "input_limit": 8192})
        data = sse("start", {"queued": True}) + sse("tokens", {"token_ids": [9], "text": "你好"})
        if not incomplete:
            data += sse("complete", {"text": "你好，世界。", "method": "BudgetTree-ASD", "approximate": True,
                                     "finish_reason": "stop", "token_ids": [9], "queue_seconds": .1})
        return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})
    return httpx.MockTransport(handler)


def test_chat_round_trip_identity_history_and_no_raw_tokens(config):
    calls = []
    with TestClient(create_chat_app(config, transport=transport(calls)), base_url=ORIGIN) as client:
        registered = register(client)
        assert "httponly" in registered.headers["set-cookie"].lower()
        assert "samesite=strict" in registered.headers["set-cookie"].lower()
        assert client.get("/api/me").json()["user"]["username"] == "alice"
        assert client.get("/api/status").json()["ready"]
        conversation = client.post("/api/conversations").json()["id"]
        url = f"/api/conversations/{conversation}/messages"
        answer = client.post(url, json={"content": "问题一"})
        assert answer.status_code == 200
        assert 'event: text\ndata: {"text": "你好"}' in answer.text
        assert "token_ids" not in answer.text
        assert "Bearer " not in answer.text
        saved = client.get(f"/api/conversations/{conversation}").json()
        assert [m["content"] for m in saved["messages"]] == ["问题一", "你好，世界。"]
        assert saved["messages"][-1]["status"] == "complete"
        assert saved["messages"][-1]["metadata"]["approximate"]
        assert client.post(url, json={"content": "问题二"}).status_code == 200
        generation = [body for path, body, _ in calls if path == "/v1/generate"][-1]
        assert generation["messages"] == [{"role": "user", "content": "问题一"},
            {"role": "assistant", "content": "你好，世界。"}, {"role": "user", "content": "问题二"}]
        assert generation["user_id"] == registered.json()["user"]["id"]
        assert generation["method"] == "BudgetTree-ASD" and generation["text_stream"]
        assert all(auth == "Bearer " + KEY for _, _, auth in calls)
        assert client.post(url, json={"content": "x", "user_id": "somebody-else"}).status_code == 422
        assert client.delete(f"/api/conversations/{conversation}").status_code == 200
        assert client.get(f"/api/conversations/{conversation}").status_code == 404


def test_auth_isolation_csrf_invite_logout_and_password_storage(config):
    with TestClient(create_chat_app(config, transport=transport([])), base_url=ORIGIN) as client:
        assert client.get("/api/conversations").status_code == 401
        assert client.post("/api/register", json={"username": "alice", "password": "test-password-123",
            "invite_code": "wrong"}, headers={"Origin": ORIGIN}).status_code == 403
        registered = register(client)
        alice_token = client.cookies.get(COOKIE)
        alice_csrf = registered.json()["csrf"]
        conversation = client.post("/api/conversations").json()["id"]
        assert client.post("/api/conversations", headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.post("/api/conversations", headers={"X-CSRF-Token": "wrong"}).status_code == 403
        with sqlite3.connect(config.database) as db:
            digest = db.execute("SELECT password FROM users").fetchone()[0]
            assert "test-password" not in digest and verify_password("test-password-123", digest)
            assert db.execute("SELECT token FROM sessions").fetchone()[0] != alice_token
        client.cookies.clear()
        register(client, "bob")
        assert client.get("/api/conversations").json() == []
        for method, suffix, data in (("GET", "", None), ("DELETE", "", None),
                                     ("POST", "/cancel", {}), ("POST", "/messages", {"content": "x"})):
            assert client.request(method, f"/api/conversations/{conversation}{suffix}", json=data).status_code == 404
        assert client.post("/api/logout").status_code == 200
        assert client.get("/api/me").status_code == 401
        assert client.post("/api/login", json={"username": "alice", "password": "wrong-password"}).status_code == 401
        assert client.post("/api/login", json={"username": "alice", "password": "test-password-123"}).status_code == 200
        client.cookies.clear(); client.cookies.set(COOKIE, alice_token)
        client.headers["X-CSRF-Token"] = alice_csrf
        assert client.get(f"/api/conversations/{conversation}").status_code == 200


def test_incomplete_stream_retains_partial_and_excludes_failed_turn_from_context(config):
    with TestClient(create_chat_app(config, transport=transport([], incomplete=True)), base_url=ORIGIN) as client:
        register(client)
        conversation = client.post("/api/conversations").json()["id"]
        result = client.post(f"/api/conversations/{conversation}/messages", json={"content": "hello"})
        assert "event: error" in result.text
        saved = client.get(f"/api/conversations/{conversation}").json()["messages"]
        assert saved[-1]["content"] == "你好" and saved[-1]["status"] == "failed"
        _, _, history = client.app.state.store.begin_turn(
            client.get("/api/me").json()["user"]["id"], conversation, "retry")
        assert history == []


def test_queue_limits_restart_recovery_and_delete_protection(tmp_path):
    store = ChatStore(tmp_path / "chat.db")
    users = [store.create_user("user" + str(i), "hash")["id"] for i in range(3)]
    conversations = [store.new_conversation(u)["id"] for u in users]
    answer, _, _ = store.begin_turn(users[0], conversations[0], "one")
    store.save_partial(answer, "partial")
    with pytest.raises(StoreError, match="已有一个"):
        store.begin_turn(users[0], conversations[0], "duplicate")
    store.begin_turn(users[1], conversations[1], "two")
    with pytest.raises(StoreError) as full:
        store.begin_turn(users[2], conversations[2], "three")
    assert full.value.status == 429
    with pytest.raises(StoreError, match="先停止"):
        store.delete_conversation(users[0], conversations[0])
    restarted = ChatStore(tmp_path / "chat.db")
    assert restarted.conversation(users[0], conversations[0])["messages"][-1]["status"] == "interrupted"
    assert restarted.conversation(users[0], conversations[0])["messages"][-1]["content"] == "partial"
    restarted.begin_turn(users[0], conversations[0], "after restart")


def test_context_trims_whole_turns_and_refuses_oversized_latest_question():
    from memorydraft.chat.upstream import UpstreamError
    async def scenario():
        def handler(request):
            count = len(json.loads(request.content)["messages"])
            return httpx.Response(200, json={"input_tokens": count * 10, "input_limit": 15})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=ORIGIN) as client:
            messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old answer"},
                        {"role": "user", "content": "latest"}]
            fitted, dropped, count = await fit_context(client, messages, 128)
            assert fitted == messages[-1:] and dropped == 1 and count == 10
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(
                200, json={"input_tokens": 50, "input_limit": 10})), base_url=ORIGIN) as client:
            with pytest.raises(UpstreamError, match="缩短"):
                await fit_context(client, messages[-1:], 128)
    asyncio.run(scenario())


def test_explicit_stop_closes_upstream_and_releases_admission(config):
    async def scenario():
        started, closed = asyncio.Event(), asyncio.Event()
        class Slow(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield (sse("start", {}) + sse("tokens", {"text": "partial"})).encode()
                started.set()
                await asyncio.sleep(60)
            async def aclose(self):
                closed.set()
        def handler(request):
            if request.url.path == "/v1/token-count":
                return httpx.Response(200, json={"input_tokens": 3, "input_limit": 100})
            return httpx.Response(200, stream=Slow(), headers={"content-type": "text/event-stream"})
        app = create_chat_app(config, transport=httpx.MockTransport(handler))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ORIGIN,
                                         headers={"Origin": ORIGIN}) as client:
                result = await client.post("/api/register", json={"username": "alice", "password": "test-password-123", "invite_code": INVITE})
                client.headers["X-CSRF-Token"] = result.json()["csrf"]
                conversation = (await client.post("/api/conversations")).json()["id"]
                task = asyncio.create_task(client.post(f"/api/conversations/{conversation}/messages", json={"content": "x"}))
                await asyncio.wait_for(started.wait(), 2)
                assert (await client.post(f"/api/conversations/{conversation}/cancel")).status_code == 200
                response = await asyncio.wait_for(task, 2)
                assert "event: stopped" in response.text
                assert closed.is_set() and not app.state.streams
                saved = (await client.get(f"/api/conversations/{conversation}")).json()["messages"][-1]
                assert saved["content"] == "partial" and saved["status"] == "cancelled"
    asyncio.run(scenario())


def test_unicode_prefix_snapshot_and_internal_api_auth():
    class Tokenizer:
        def decode(self, tokens, **kwargs):
            return bytes(tokens).decode(errors="replace")
    class Engine:
        tokenizer = Tokenizer()
    decoder = PrefixText(Engine())
    assert decoder.append([0xe4]) == ""
    assert decoder.append([0xbd]) == ""
    assert decoder.append([0xa0]) == "你"
    assert decoder.append([0xe5, 0xa5, 0xbd]) == "你好"
    with TestClient(create_app(DemoEngine, api_key=KEY)) as client:
        assert client.get("/healthz").status_code == 401
        client.headers["Authorization"] = "Bearer " + KEY
        assert client.get("/healthz").json()["text_stream"]
        body = {"id": "r", "user_id": "u", "order": 1, "messages": [{"role": "user", "content": "hi"}],
                "max_new_tokens": 50, "stream": True, "text_stream": True}
        result = client.post("/v1/generate", json=body)
        assert '"text": "MemoryDraft transport demo."' in result.text
        counts = client.post("/v1/token-count", json={"messages": body["messages"]}).json()
        assert counts == {"input_tokens": 2, "input_limit": 8192}


@pytest.mark.parametrize("origin", ["http://203.0.113.1", "http://0.0.0.0", "https://example.com/a", "http://user:pass@127.0.0.1"])
def test_config_prevents_public_plain_http_and_credential_urls(config, origin):
    with pytest.raises(ValueError):
        replace(config, public_origin=origin)


def test_static_page_host_validation_and_invite_rate_limit(config):
    with TestClient(create_chat_app(config, transport=transport([])), base_url=ORIGIN) as client:
        response = client.get("/")
        assert response.status_code == 200 and "树流" in response.text and "MemoryDraft" not in response.text
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        assert client.get("/", headers={"Host": "evil.example"}).status_code == 400
        for _ in range(12):
            assert client.post("/api/register", json={"username": "alice", "password": "test-password-123", "invite_code": "bad"}, headers={"Origin": ORIGIN}).status_code == 403
        assert client.post("/api/register", json={"username": "alice", "password": "test-password-123", "invite_code": INVITE}, headers={"Origin": ORIGIN}).status_code == 429
