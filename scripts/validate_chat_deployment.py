"""Opt-in real deployment checks using temporary, synthetic chat accounts.

Run on the gateway host. Credentials and browser state stay in its private
directory; the exported report contains no cookie, password or invitation code.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import time
from urllib.parse import urlsplit

import httpx


async def streamed(client, path, content, *, started=None, stop_after_text=False):
    names, text, metadata = Counter(), "", {}
    begun, first = time.perf_counter(), None
    async with client.stream("POST", path + "/messages", json={"content": content}) as response:
        assert response.status_code == 200, response.status_code
        name = None
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                value = json.loads(line[6:])
                names[name] += 1
                if name == "error":
                    raise AssertionError(value.get("message"))
                if name == "waiting" and value.get("queued"):
                    names["observed_queued"] += 1
                if name == "text":
                    text = value["text"]
                    first = first or time.perf_counter() - begun
                    if started:
                        started.set()
                    if stop_after_text:
                        result = await client.post(path + "/cancel", json={})
                        assert result.status_code == 200
                        stop_after_text = False
                if name in {"complete", "stopped"}:
                    text, metadata = value["text"], value.get("metadata", {})
    assert names["complete"] + names["stopped"] == 1
    assert text and "\ufffd" not in text
    return {"events": dict(names), "characters": len(text), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "wall_seconds": time.perf_counter() - begun, "first_text_seconds": first,
            "metadata": metadata}


async def validate(env, output, private):
    origin = env["MEMORYDRAFT_PUBLIC_ORIGIN"]
    invite = env["MEMORYDRAFT_INVITE_CODE"]
    private.mkdir(parents=True, exist_ok=True)
    accounts = []
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "real_model": True, "checks": {}}
    clients = [httpx.AsyncClient(base_url=origin, headers={"Origin": origin}, trust_env=False,
                               timeout=httpx.Timeout(650, connect=10)) for _ in range(2)]
    try:
        for index, client in enumerate(clients):
            username = "deploycheck_" + secrets.token_hex(5) + "_" + str(index)
            result = await client.post("/api/register", json={"username": username,
                "password": secrets.token_urlsafe(24), "invite_code": invite})
            assert result.status_code == 200, result.status_code
            value = result.json()
            client.headers["X-CSRF-Token"] = value["csrf"]
            accounts.append(value["user"]["id"])
        report["checks"]["two_registrations"] = True
        paths = []
        for client in clients:
            result = await client.post("/api/conversations", json={})
            assert result.status_code == 200
            paths.append("/api/conversations/" + result.json()["id"])
        assert (await clients[1].get(paths[0])).status_code == 404
        report["checks"]["cross_user_isolation"] = True
        status = (await clients[0].get("/api/status")).json()
        assert status["ready"] and not status["synthetic"] and status["method"] == "BudgetTree-ASD"
        report["checks"]["real_budget_tree_asd"] = True
        report["first_turn"] = await streamed(clients[0], paths[0], "请用一句中文介绍你自己。")
        report["second_turn"] = await streamed(clients[0], paths[0], "我刚才让你做了什么？请用一句话回答。")
        saved = (await clients[0].get(paths[0])).json()
        assert len(saved["messages"]) == 4 and all(m["status"] == "complete" for m in saved["messages"])
        report["checks"]["multi_turn_persistence"] = True
        first_token = asyncio.Event()
        first_task = asyncio.create_task(streamed(clients[0], paths[0],
            "请逐项介绍一百个Python标准库函数，每项给出一句使用说明，按编号连续输出。", started=first_token))
        await asyncio.wait_for(first_token.wait(), 90)
        second_task = asyncio.create_task(streamed(clients[1], paths[1],
            "请用中文写一段约四百字的故事，主题是两个人一起修好一艘小船。"))
        report["simultaneous_requests"] = await asyncio.gather(first_task, second_task)
        assert any(row["events"].get("observed_queued") for row in report["simultaneous_requests"])
        report["checks"]["two_user_queue"] = True
        report["cancelled"] = await streamed(clients[0], paths[0],
            "从一开始写出一百个不同的编程练习，每个练习详细描述要求。", stop_after_text=True)
        assert report["cancelled"]["events"].get("stopped") == 1
        assert (await clients[0].get(paths[0])).json()["messages"][-1]["status"] == "cancelled"
        report["checks"]["stop_persisted"] = True
        report["recovered"] = await streamed(clients[0], paths[0], "只回答：连接正常。")
        report["checks"]["generation_after_stop"] = True
        cookie = clients[0].cookies.get("memorydraft_session")
        state = {"cookies": [{"name": "memorydraft_session", "value": cookie,
            "domain": urlsplit(origin).hostname, "path": "/", "expires": int(time.time()) + 3600,
            "httpOnly": True, "secure": origin.startswith("https:"), "sameSite": "Strict"}], "origins": []}
        (private / "ui-state.json").write_text(json.dumps(state), encoding="utf-8")
        (private / "check-accounts.json").write_text(json.dumps(accounts), encoding="utf-8")
        for p in private.glob("*.json"):
            p.chmod(0o600)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"checks": report["checks"], "report": str(output)}))
    finally:
        for client in clients:
            await client.aclose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--private", required=True, type=Path)
    args = parser.parse_args()
    env = dict(line.split("=", 1) for line in args.env.read_text().splitlines() if line and not line.startswith("#"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(validate(env, args.output, args.private))


if __name__ == "__main__":
    main()
