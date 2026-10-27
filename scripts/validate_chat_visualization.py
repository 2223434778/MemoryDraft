"""Opt-in GPU observer equivalence and end-to-end checks with synthetic input."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import secrets
import time

import httpx

from memorydraft.chat.trace import TraceHistory
from memorydraft.chat.upstream import events


async def inference(url, key, output):
    report = {"real_gpu": True, "checks": [], "purpose": "observer correctness; not a speed benchmark"}
    async with httpx.AsyncClient(base_url=url, headers={"Authorization": "Bearer " + key},
                                timeout=650, trust_env=False) as client:
        health = (await client.get("/healthz")).json()
        assert health["speculation_events"]
        report["health"] = health
        for method in ("AR", "DFlash", "ASD", "BudgetTree-ASD"):
            body = {"id": "visual-check-" + secrets.token_hex(4), "user_id": "synthetic-observer-check", "order": 1,
                "messages": [{"role": "user", "content": "Explain speculative decoding in three short sentences."}],
                "method": method, "max_new_tokens": 64, "timeout_seconds": 600}
            response = await client.post("/v1/generate", json=body)
            response.raise_for_status(); baseline = response.json()
            history, phases, actual = TraceHistory(), [], None
            async for name, value in events(client, {**body, "stream": True, "text_stream": True, "visualize": True}):
                if name == "error":
                    raise AssertionError(value)
                if name == "speculation":
                    history.add(value); phases.append(value["phase"])
                if name == "complete":
                    actual = value
            assert actual is not None and actual["token_ids"] == baseline["token_ids"]
            assert actual["asd"] == baseline["asd"] and actual["counters"] == baseline["counters"], {
                "method": method, "asd": [baseline["asd"], actual["asd"]],
                "counter_differences": {k: [baseline["counters"].get(k), actual["counters"].get(k)]
                    for k in baseline["counters"].keys() | actual["counters"].keys()
                    if baseline["counters"].get(k) != actual["counters"].get(k)}}
            assert actual["prompt_sha256"] == baseline["prompt_sha256"]
            assert phases == ["draft", "verified"] * (len(phases) // 2)
            assert bool(phases) == (method != "AR")
            report["checks"].append({"method": method, "identical_tokens": True, "identical_counters": True,
                "identical_asd": True, "rounds": len(phases) // 2, "output_tokens": len(actual["token_ids"]),
                "token_sha256": hashlib.sha256(json.dumps(actual["token_ids"]).encode()).hexdigest(),
                "asd": actual["asd"], "tree": actual["tree"]})
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gpu_checks": len(report["checks"]), "output": str(output)}))


async def gateway(env, output, private):
    origin = env["MEMORYDRAFT_PUBLIC_ORIGIN"]
    async with httpx.AsyncClient(base_url=origin, headers={"Origin": origin}, timeout=650, trust_env=False) as client:
        response = await client.post("/api/register", json={"username": "visualcheck_" + secrets.token_hex(5),
            "password": secrets.token_urlsafe(24), "invite_code": env["MEMORYDRAFT_INVITE_CODE"]})
        response.raise_for_status(); account = response.json()
        client.headers["X-CSRF-Token"] = account["csrf"]
        models = (await client.get("/api/models")).json()
        model = next(m for m in models["models"] if m["id"] == models["default_model"])
        report = {"models": models, "checks": []}
        for method in ("AR", "DFlash", "ASD", "BudgetTree-ASD"):
            cid = (await client.post("/api/conversations")).json()["id"]
            response = await client.post(f"/api/conversations/{cid}/messages", json={"content": "请简短解释投机解码。",
                "model": model["id"], "method": method, "visualize": method != "AR"})
            response.raise_for_status()
            assert "event: complete" in response.text and "event: error" not in response.text
            saved = (await client.get(f"/api/conversations/{cid}")).json()["messages"][-1]
            trace = saved["metadata"].get("speculation")
            assert bool(trace) == (method != "AR")
            report["checks"].append({"method": method, "characters": len(saved["content"]),
                "rounds": trace["round_count"] if trace else 0, "persisted": saved["status"] == "complete"})
        # Keep a short-lived browser session private for real UI acceptance.
        private.mkdir(parents=True, exist_ok=True)
        state = {"cookies": [{"name": "memorydraft_session", "value": client.cookies.get("memorydraft_session"),
            "domain": httpx.URL(origin).host, "path": "/", "expires": int(time.time()) + 7200,
            "httpOnly": True, "secure": origin.startswith("https:"), "sameSite": "Strict"}], "origins": []}
        (private / "ui-state.json").write_text(json.dumps(state), encoding="utf-8")
        (private / "check-accounts.json").write_text(json.dumps([account["user"]["id"]]), encoding="utf-8")
        for path in private.glob("*.json"):
            path.chmod(0o600)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"gateway_checks": len(report["checks"]), "output": str(output)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url")
    parser.add_argument("--private", type=Path)
    args = parser.parse_args()
    env = dict(line.split("=", 1) for line in args.env.read_text().splitlines() if line and not line.startswith("#"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.url:
        asyncio.run(inference(args.url, env["MEMORYDRAFT_API_KEY"], args.output))
    else:
        if not args.private:
            parser.error("gateway checks require --private")
        asyncio.run(gateway(env, args.output, args.private))


if __name__ == "__main__":
    main()
