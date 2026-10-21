"""Bounded real-model acceptance: pairing, streaming, cancellation, profiler and HTTP."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import socket
import threading
import time

import httpx
import uvicorn

from memorydraft.memory.store import MemoryEvent, MemoryStore
from memorydraft.runtime.benchmark import benchmark, profile_request, provenance, write_json
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest, RequestCancelled
from memorydraft.serving.app import create_app
from memorydraft.serving.loadtest import load_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    config = RuntimeConfig.load(args.config)
    if config.backend != "dflash":
        raise ValueError("Acceptance requires the prepared real DFlash profile")
    database = args.output.resolve() / "history.db"
    with MemoryStore(database) as store:
        store.append([MemoryEvent("example", "past-quotation", 1,
                                  "alpha bravo charlie delta echo foxtrot.\n" * 4)])
        history_before = store.snapshot("example", 1000).snapshot_id
    config = replace(config, db=str(database))
    engine = load_engine(config)
    info = {**provenance(engine), "status": "started"}
    write_json(args.output / "acceptance.json", info)
    try:
        # Fresh process/cache preparation is controlled by the deployment runner.
        # These are cold-call costs, separate from the warmed benchmark below.
        cold_request = GenerationRequest("__cold__", "__cold__", 0,
                                         ({"role": "user", "content": "Say hello."},), 8)
        cold = [asdict(engine.generate(cold_request, method)) for method in ("AR", "DFlash")]
        write_json(args.output / "cold-start.json", cold)
        info["cold_start"] = cold
        requests = [GenerationRequest.from_dict(json.loads(line)) for line in
                    (root / "examples/requests.jsonl").read_text().splitlines() if line.strip()]
        benchmark(engine, requests, engine.methods, args.output / "benchmark", repeats=2)
        benchmark_rows = [json.loads(line) for line in (args.output/"benchmark/rows.jsonl").read_text().splitlines()]
        info["reuse_accepted_tokens"] = sum(r["counters"].get("reuse_accepted_tokens", 0) for r in benchmark_rows if r["method"] == "R+DFlash")
        if not info["reuse_accepted_tokens"]:
            raise RuntimeError("Acceptance copy workload did not exercise accepted retrieval")
        request = replace(requests[0], max_new_tokens=16)
        reference = engine.generate(request, "AR")
        body = {"id": request.id, "user_id": request.user_id, "order": request.order,
                "messages": list(request.messages), "max_new_tokens": request.max_new_tokens}
        cancelled = threading.Event()
        try:
            engine.generate(request, "R+DFlash", on_tokens=lambda _: cancelled.set(), should_stop=cancelled.is_set)
            raise RuntimeError("Cancellation was not observed")
        except RequestCancelled:
            pass
        if engine.generate(request, "DFlash").token_ids != reference.token_ids:
            raise RuntimeError("Post-cancellation generation mismatch")
        info["cancellation_recovered"] = True
        profile_request(engine, requests[1], "R+DFlash", args.output / "profile")
        trace = json.loads((args.output / "profile/trace.json").read_text())
        events = trace["traceEvents"]
        info["profiler"] = {"events": len(events), "cuda_kernel_events": sum(e.get("cat") == "kernel" for e in events),
                            "stage_names": sorted({e["name"] for e in events if e.get("name", "").startswith("memorydraft::")})}
        del trace, events
        if not info["profiler"]["cuda_kernel_events"]:
            raise RuntimeError("CUDA profiler did not record actual kernels")
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        url = "http://127.0.0.1:" + str(sock.getsockname()[1])
        server = uvicorn.Server(uvicorn.Config(create_app(lambda: engine), log_level="warning", access_log=False))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.perf_counter() + 30
            while not server.started:
                if not thread.is_alive() or time.perf_counter() > deadline:
                    raise RuntimeError("HTTP server startup failed")
                time.sleep(.02)
            info["http_load"] = load_test(url, {**body, "method": "R+DFlash"}, args.output / "http-load",
                                          count=6, concurrency=2)
            # One server lifespan owns this engine, closing it only at the end.
            with httpx.Client(trust_env=False, timeout=120) as client:
                streams = {}
                for method in engine.methods:
                    response = client.post(url + "/v1/generate", json={**body, "method": method, "stream": True})
                    response.raise_for_status()
                    tokens, completed = [], None
                    for block in response.text.split("\n\n"):
                        lines = block.splitlines()
                        if len(lines) != 2:
                            continue
                        event, data = lines[0][7:], json.loads(lines[1][6:])
                        if event == "tokens":
                            tokens.extend(data["token_ids"])
                        elif event == "complete":
                            completed = data
                        elif event == "error":
                            raise RuntimeError("HTTP stream error: " + str(data))
                    if (completed is None or tokens != reference.token_ids or tokens != completed["token_ids"]
                            or completed["execution_profile"] != config.execution_profile):
                        raise RuntimeError("Real-model HTTP stream mismatch: " + method)
                    streams[method] = {"correct": True, "tokens": len(tokens), "execution_profile": completed["execution_profile"]}
                info["http_streams"] = streams
                info["http_metrics"] = client.get(url + "/metrics").json()
            if info["http_load"]["failed"]:
                raise RuntimeError("HTTP load acceptance failed")
        finally:
            server.should_exit = True
            thread.join(120)
            sock.close()
            if thread.is_alive():
                raise RuntimeError("HTTP shutdown did not finish")
        with MemoryStore(database) as store:
            info["history_unchanged"] = store.snapshot("example", 1000).snapshot_id == history_before
        if not info["history_unchanged"]:
            raise RuntimeError("Acceptance generated output was written into history")
        info["final_engine"] = engine.describe()
        if engine._regional is not None:
            info["compiler"] = engine._regional.summary()
        info["status"] = "complete"
    except Exception as error:
        info.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            engine.close()
        except Exception as error:
            info.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            write_json(args.output / "acceptance.json", info)
    print(json.dumps({"status": info["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
