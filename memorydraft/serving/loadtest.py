"""Closed-loop HTTP load test. Client observations are distinct from engine timing."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import httpx

from memorydraft.runtime.benchmark import percentile, write_json


def load_test(url, body, output, *, count=20, concurrency=2, timeout_seconds=120.0):
    if type(count) is not int or type(concurrency) is not int or min(count, concurrency) < 1:
        raise ValueError("count and concurrency must be positive integers")
    if not url.startswith(("http://", "https://")):
        raise ValueError("Use an explicit HTTP(S) service URL")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with httpx.Client(base_url=url.rstrip("/"), timeout=timeout_seconds, trust_env=False) as client:
        health = client.get("/healthz")
        health.raise_for_status()
        health_data = health.json()

        def send(index):
            started = time.perf_counter()
            row = {"index": index, "status": None, "token_count": 0, "ttft_seconds": None,
                   "queue_seconds": None, "error": None}
            payload = {**body, "id": f"load-{index}", "stream": True, "timeout_seconds": timeout_seconds}
            completed, complete_tokens, received = False, None, []
            try:
                with client.stream("POST", "/v1/generate", json=payload) as response:
                    row["status"] = response.status_code
                    if response.status_code != 200:
                        row["error"] = "HTTP error"
                    else:
                        name = None
                        for line in response.iter_lines():
                            if line.startswith("event: "):
                                name = line[7:]
                            elif line.startswith("data: "):
                                data = json.loads(line[6:])
                                if name == "tokens":
                                    if row["ttft_seconds"] is None and data["token_ids"]:
                                        row["ttft_seconds"] = time.perf_counter() - started
                                    received.extend(data["token_ids"])
                                elif name == "complete":
                                    completed = True
                                    complete_tokens = data["token_ids"]
                                    row["queue_seconds"] = data["queue_seconds"]
                                elif name == "error":
                                    row.update(status=data["code"], error=data["message"])
                        if not completed and row["error"] is None:
                            row["error"] = "Missing completion event"
                        if completed and received != complete_tokens:
                            row["error"] = "Stream tokens differ from completed result"
            except (httpx.HTTPError, ValueError, KeyError) as error:
                row["error"] = type(error).__name__
            row["latency_seconds"] = time.perf_counter() - started
            row["token_count"] = len(received)
            row["success"] = row["status"] == 200 and completed and row["error"] is None
            return row

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(send, range(count)))
        elapsed = time.perf_counter() - started
    successful = [row for row in rows if row["success"]]
    latencies = [row["latency_seconds"] for row in successful]
    first = [row["ttft_seconds"] for row in successful if row["ttft_seconds"] is not None]
    summary = {"synthetic": health_data.get("synthetic", False), "mode": "closed_loop",
               "requests": count, "concurrency": concurrency, "successful": len(successful),
               "failed": count - len(successful), "wall_seconds": elapsed,
               "successful_requests_per_second": len(successful) / elapsed,
               "completed_tokens_per_second": sum(row["token_count"] for row in successful) / elapsed,
               "p50_seconds": percentile(latencies, .5),
               "p95_seconds": percentile(latencies, .95) if len(latencies) >= 20 else None,
               "p99_seconds": percentile(latencies, .99) if len(latencies) >= 100 else None,
               "p50_ttft_seconds": percentile(first, .5),
               "note": "Client wall time includes queue and HTTP/SSE transport; percentiles cover successful requests only"}
    write_json(output / "summary.json", summary)
    with (output / "rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return summary
