from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
import secrets
import time

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictStr
from starlette.middleware.trustedhost import TrustedHostMiddleware
from urllib.parse import urlsplit

from memorydraft.chat.config import ChatConfig
from memorydraft.chat.security import LoginLimiter, password_hash, verify_password
from memorydraft.chat.store import ChatStore, StoreError
from memorydraft.chat.upstream import ChatStopped, UpstreamError, events, fit_context, until_stopped


COOKIE = "memorydraft_session"


class LoginBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: StrictStr = Field(min_length=3, max_length=32)
    password: StrictStr = Field(min_length=10, max_length=128)


class RegisterBody(LoginBody):
    invite_code: StrictStr = Field(max_length=256)


class ChatBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: StrictStr = Field(min_length=1, max_length=16384)


def sse(name, value):
    return "event: " + name + "\ndata: " + json.dumps(value, ensure_ascii=False) + "\n\n"


def create_chat_app(config: ChatConfig, *, transport=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = ChatStore(config.database)
        app.state.streams = {}
        app.state.limiter = LoginLimiter()
        app.state.hash_slots = asyncio.Semaphore(2)
        app.state.dummy_password = await asyncio.to_thread(password_hash, secrets.token_urlsafe(32))
        async with httpx.AsyncClient(base_url=config.upstream_url,
                headers={"Authorization": "Bearer " + config.upstream_key},
                transport=transport, trust_env=False, follow_redirects=False,
                timeout=httpx.Timeout(connect=5, read=25, write=10, pool=5),
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4)) as client:
            app.state.upstream = client
            try:
                yield
            finally:
                for _, signal in app.state.streams.values():
                    signal.set()

    app = FastAPI(title="MemoryDraft Chat", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[urlsplit(config.public_origin).hostname])

    @app.exception_handler(StoreError)
    async def store_error(request, error):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": error.message}, status_code=error.status)

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"})
        return response

    def check_origin(request):
        origin = request.headers.get("origin")
        if origin != config.public_origin:
            raise HTTPException(403, "请求来源不正确，请通过网站页面操作")

    def user(request: Request):
        session = request.app.state.store.session(request.cookies.get(COOKIE, ""))
        if not session:
            raise HTTPException(401, "请先登录")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            check_origin(request)
            if not secrets.compare_digest(request.headers.get("x-csrf-token", "").encode(), session["csrf"].encode()):
                raise HTTPException(403, "登录状态已变化，请刷新页面")
        return session

    def login_result(request, response, account):
        token, csrf = request.app.state.store.new_session(account["id"])
        response.set_cookie(COOKIE, token, max_age=7 * 86400, httponly=True,
                            secure=config.secure_cookie, samesite="strict", path="/")
        return {"user": {"id": account["id"], "username": account["username"]}, "csrf": csrf}

    def auth_attempt(request):
        check_origin(request)
        if not request.app.state.limiter.allow(request.client.host if request.client else "unknown"):
            raise HTTPException(429, "尝试次数较多，请五分钟后重试")

    @app.post("/api/register")
    async def register(body: RegisterBody, request: Request, response: Response):
        auth_attempt(request)
        if not secrets.compare_digest(body.invite_code.encode(), config.invite_code.encode()):
            raise HTTPException(403, "邀请码不正确")
        username = body.username.casefold()
        if not re.fullmatch(r"[a-z0-9_\u4e00-\u9fff-]{3,32}", username):
            raise HTTPException(422, "用户名使用3至32位汉字、字母、数字、下划线或横线")
        async with request.app.state.hash_slots:
            digest = await asyncio.to_thread(password_hash, body.password)
        account = request.app.state.store.create_user(username, digest)
        return login_result(request, response, account)

    @app.post("/api/login")
    async def login(body: LoginBody, request: Request, response: Response):
        auth_attempt(request)
        account = request.app.state.store.find_user(body.username.casefold())
        digest = account["password"] if account else request.app.state.dummy_password
        async with request.app.state.hash_slots:
            valid = await asyncio.to_thread(verify_password, body.password, digest)
        if not valid or account is None:
            raise HTTPException(401, "用户名或密码不正确")
        return login_result(request, response, account)

    @app.get("/api/me")
    def me(account=Depends(user)):
        return {"user": {"id": account["id"], "username": account["username"]}, "csrf": account["csrf"]}

    @app.post("/api/logout")
    def logout(request: Request, response: Response, account=Depends(user)):
        request.app.state.store.logout(request.cookies.get(COOKIE, ""))
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @app.get("/api/status")
    async def status(request: Request, account=Depends(user)):
        try:
            response = await request.app.state.upstream.get("/healthz", timeout=5)
            health = response.json() if response.status_code == 200 else {}
            ready = health.get("status") == "ready" and config.method in health.get("methods", [])
            return {"ready": ready, "method": config.method, "synthetic": health.get("synthetic", False)}
        except (httpx.HTTPError, ValueError):
            return {"ready": False, "method": config.method}

    @app.get("/api/conversations")
    def conversations(request: Request, account=Depends(user)):
        return request.app.state.store.conversations(account["id"])

    @app.post("/api/conversations")
    def new_conversation(request: Request, account=Depends(user)):
        return request.app.state.store.new_conversation(account["id"])

    @app.get("/api/conversations/{conversation_id}")
    def conversation(conversation_id: str, request: Request, account=Depends(user)):
        return request.app.state.store.conversation(account["id"], conversation_id)

    @app.delete("/api/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str, request: Request, account=Depends(user)):
        request.app.state.store.delete_conversation(account["id"], conversation_id)
        return {"ok": True}

    @app.post("/api/conversations/{conversation_id}/cancel")
    def cancel(conversation_id: str, request: Request, account=Depends(user)):
        request.app.state.store.conversation(account["id"], conversation_id)
        running = request.app.state.streams.get(conversation_id)
        if running and running[0] == account["id"]:
            running[1].set()
        return {"ok": True}

    @app.post("/api/conversations/{conversation_id}/messages")
    async def chat(conversation_id: str, body: ChatBody, request: Request, account=Depends(user)):
        content = body.content.strip()
        if not content:
            raise HTTPException(422, "请输入消息")
        store = request.app.state.store
        assistant_id, turn_id, history = store.begin_turn(account["id"], conversation_id, content)
        signal = asyncio.Event()
        request.app.state.streams[conversation_id] = (account["id"], signal)

        async def stream():
            text, final_status, metadata = "", "cancelled", {}
            source = None
            try:
                yield sse("accepted", {"assistant_id": assistant_id, "method": config.method})
                messages, dropped, tokens = await until_stopped(fit_context(request.app.state.upstream,
                    history + [{"role": "user", "content": content}], config.max_new_tokens), signal)
                metadata = {"dropped_turns": dropped + max(0, (len(history) - 120) // 2), "input_tokens": tokens}
                yield sse("context", metadata)
                source = events(request.app.state.upstream, {"id": turn_id, "user_id": account["id"],
                    "order": time.time_ns(), "messages": messages, "max_new_tokens": config.max_new_tokens,
                    "method": config.method, "stream": True, "text_stream": True,
                    "timeout_seconds": config.timeout_seconds})
                saved = time.monotonic()
                while True:
                    try:
                        name, value = await until_stopped(source.__anext__(), signal)
                    except StopAsyncIteration:
                        break
                    if name in {"start", "waiting"}:
                        yield sse("waiting", {"queued": value.get("queued", False)})
                    elif name == "tokens":
                        if not isinstance(value.get("text"), str):
                            raise UpstreamError("模型服务需要升级文本流接口")
                        text = value["text"]
                        if len(text) > 262144:
                            raise UpstreamError("模型回答超过长度限制")
                        if time.monotonic() - saved >= 1:
                            store.save_partial(assistant_id, text)
                            saved = time.monotonic()
                        yield sse("text", {"text": text})
                    elif name == "error":
                        code = value.get("code")
                        message = "本次回答等待超时，请重试" if code == 504 else "模型生成失败，请重试"
                        raise UpstreamError(message, code if code in {422, 429, 504} else 502)
                    elif name == "complete":
                        if not isinstance(value.get("text"), str):
                            raise UpstreamError("模型返回了无效的回答")
                        text = value["text"]
                        if len(text) > 262144:
                            raise UpstreamError("模型回答超过长度限制")
                        metadata.update({k: value[k] for k in ("method", "approximate", "finish_reason",
                            "latency_seconds", "server_seconds", "queue_seconds", "server_ttft_seconds") if k in value})
                        final_status = "complete"
                        store.finish(assistant_id, text, final_status, metadata)
                        yield sse("complete", {"text": text, "metadata": metadata})
                        return
                raise UpstreamError("连接提前结束，已保留收到的回答")
            except ChatStopped:
                yield sse("stopped", {"text": text})
            except UpstreamError as error:
                final_status = "failed"
                metadata["error"] = str(error)
                yield sse("error", {"code": error.code, "message": str(error)})
            except (httpx.HTTPError, ValueError):
                final_status = "failed"
                metadata["error"] = "模型连接中断，请稍后重试"
                yield sse("error", {"code": 502, "message": metadata["error"]})
            finally:
                with anyio.CancelScope(shield=True):
                    if source is not None:
                        await source.aclose()
                store.finish(assistant_id, text, final_status, metadata)
                if request.app.state.streams.get(conversation_id) == (account["id"], signal):
                    request.app.state.streams.pop(conversation_id, None)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    return app
