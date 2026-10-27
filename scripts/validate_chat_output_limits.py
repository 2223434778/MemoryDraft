"""Bounded real-service long-output checks with public synthetic input only."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import secrets
import shlex
import time
from urllib.parse import urlsplit

import httpx

from memorydraft.chat.trace import TraceHistory
from memorydraft.chat.upstream import events


async def validate(env_path, output, limit, urls):
    env = {key: shlex.split(value)[0] for line in env_path.read_text().splitlines()
           if line and not line.startswith("#") for key, value in [line.split("=", 1)]}
    report = {"real_gpu": True, "requested_output_limit": limit, "checks": [], "progress": {},
              "peak_gpu_used_mib": {}, "purpose": "length and stability check; not a speed benchmark"}
    finished = asyncio.Event()
    started = time.monotonic()

    def save():
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        temporary = output.with_suffix(".next.json")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)

    async def monitor():
        while not finished.is_set():
            process = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE)
            stdout, _ = await process.communicate()
            if process.returncode != 0:
                raise RuntimeError("GPU memory monitor failed")
            for line in stdout.decode().splitlines():
                gpu, memory = line.split(",")
                gpu = gpu.strip()
                report["peak_gpu_used_mib"][gpu] = max(int(memory), report["peak_gpu_used_mib"].get(gpu, 0))
            save()
            try:
                await asyncio.wait_for(finished.wait(), 3)
            except TimeoutError:
                pass

    async def check(url, method, *, long):
        port = urlsplit(url).port
        identifier = f"{port}-{method}-{'long' if long else 'smoke'}"
        prompt = ("Output only a JSON array containing every integer from 0 to 49999 in ascending order. "
                  "Do not abbreviate, omit, use ellipses, or describe the answer. Continue until all integers are included."
                  if long else "Reply with only the word READY.")
        body = {"id": "length-check-" + secrets.token_hex(6), "user_id": "synthetic-length-check", "order": time.time_ns(),
                "messages": [{"role": "user", "content": prompt}], "method": method,
                "max_new_tokens": limit, "timeout_seconds": 3600,
                "stream": True, "text_stream": True, "visualize": method != "AR"}
        trace, count, actual = TraceHistory(), 0, None
        call_started = time.monotonic()
        async with httpx.AsyncClient(base_url=url, trust_env=False,
                headers={"Authorization": "Bearer " + env["MEMORYDRAFT_API_KEY"]},
                timeout=httpx.Timeout(connect=5, read=30, write=15, pool=5)) as client:
            counts = await client.post("/v1/token-count", json={"messages": body["messages"], "max_new_tokens": limit})
            counts.raise_for_status()
            assert counts.json()["input_limit"] == 8192, counts.json()
            async for name, value in events(client, body):
                if name == "error":
                    raise AssertionError({"check": identifier, "error": value})
                if name == "tokens":
                    count += len(value["token_ids"])
                    report["progress"][identifier] = {"output_tokens": count, "round": trace.total,
                                                      "seconds": round(time.monotonic() - call_started, 2)}
                if name == "speculation":
                    trace.add(value)
                if name == "complete":
                    actual = value
        assert actual is not None and len(actual["token_ids"]) == count
        assert count <= limit
        if long:
            assert count == limit and actual["finish_reason"] == "length", {"check": identifier, "tokens": count,
                                                                            "finish_reason": actual["finish_reason"]}
        report["checks"].append({"port": port, "method": method, "long": long,
            "output_tokens": count, "finish_reason": actual["finish_reason"], "rounds": trace.total,
            "stream_count_matches_final": True, "input_limit": counts.json()["input_limit"],
            "text_characters": len(actual["text"]), "token_sha256": hashlib.sha256(json.dumps(actual["token_ids"]).encode()).hexdigest(),
            "elapsed_seconds": round(time.monotonic() - call_started, 2), "peak_torch_bytes": actual.get("peak_gpu_bytes")})
        save()

    output.parent.mkdir(parents=True, exist_ok=True)
    observer = asyncio.create_task(monitor())
    try:
        await asyncio.gather(check(urls[0], "BudgetTree-ASD", long=True), check(urls[1], "AR", long=True))
        for method in ("AR", "DFlash", "ASD"):
            await check(urls[0], method, long=False)
        report["passed"] = True
    except Exception as error:
        report["passed"] = False
        report["error"] = repr(error)
        raise
    finally:
        finished.set()
        await observer
        save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--large-url", required=True)
    parser.add_argument("--limit", type=int, default=16384, choices=[8192,16384])
    args = parser.parse_args()
    asyncio.run(validate(args.env, args.output, args.limit, (args.url, args.large_url)))


if __name__ == "__main__":
    main()
