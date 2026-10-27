"""Exercise the real upstream through an isolated gateway database, not user chats."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import tempfile

import httpx

from memorydraft.chat.app import create_chat_app
from memorydraft.chat.config import ChatConfig


async def validate(env_path, output):
    for line in env_path.read_text().splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key] = shlex.split(value)[0]
    production = ChatConfig.from_env()
    calls = []

    class CaptureLimits(httpx.AsyncHTTPTransport):
        async def handle_async_request(self, request):
            if request.url.path in {"/v1/token-count", "/v1/generate"}:
                body = json.loads(request.content)
                calls.append({"path":request.url.path,"max_new_tokens":body["max_new_tokens"],
                              "timeout_seconds":body.get("timeout_seconds")})
            return await super().handle_async_request(request)

    report = {"isolated_temporary_sqlite":True,"production_test_accounts_created":0,
              "production_records_deleted":0,"checks":[],
              "configured_output_limit":production.max_new_tokens,"configured_deadline":production.timeout_seconds}
    with tempfile.TemporaryDirectory(prefix="length-gateway-",dir=env_path.parent) as temporary:
        settings = replace(production,database=Path(temporary)/"chat.sqlite3")
        app = create_chat_app(settings,transport=CaptureLimits())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url=settings.public_origin,
                    headers={"Origin":settings.public_origin},timeout=3600) as client:
                response = await client.post("/api/register",json={"username":"length_"+secrets.token_hex(4),
                    "password":secrets.token_urlsafe(24),"invite_code":settings.invite_code})
                response.raise_for_status()
                client.headers["X-CSRF-Token"] = response.json()["csrf"]
                catalog = (await client.get("/api/models")).json()
                for model in catalog["models"]:
                    assert model["ready"]
                    for method in model["methods"]:
                        calls.clear()
                        cid = (await client.post("/api/conversations")).json()["id"]
                        response = await client.post(f"/api/conversations/{cid}/messages",json={
                            "content":"Reply with only the word READY.","model":model["id"],"method":method,
                            "visualize":method!="AR"})
                        response.raise_for_status()
                        block = next((b for b in response.text.split("\n\n") if b.startswith("event: complete")),None)
                        assert block is not None
                        value = json.loads(block.split("data: ",1)[1])
                        saved = (await client.get(f"/api/conversations/{cid}")).json()["messages"][-1]
                        assert saved["content"]==value["text"] and saved["status"]=="complete"
                        assert len(calls)==2 and all(c["max_new_tokens"]==16384 for c in calls)
                        assert calls[-1]["timeout_seconds"]==3600
                        report["checks"].append({"model":model["id"],"method":method,"forwarded_limits":list(calls),
                            "stored_equals_complete":True,"text_sha256":hashlib.sha256(value["text"].encode()).hexdigest()})
    report["passed"] = True
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"gateway_checks":len(report["checks"]),"passed":True}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    asyncio.run(validate(args.env,args.output))


if __name__=="__main__":
    main()
