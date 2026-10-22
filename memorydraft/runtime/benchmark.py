"""Paired correctness/performance runner. Profile runs are separate from timings."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib import metadata, resources
import json
import math
from pathlib import Path
import random
import statistics
import time

from memorydraft import __version__
from memorydraft.runtime.contracts import GenerationRequest


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_digest():
    files = {}
    root = resources.files("memorydraft")

    def visit(directory, prefix=""):
        for item in sorted(directory.iterdir(), key=lambda p: p.name):
            name = prefix + item.name
            if item.is_dir() and item.name != "__pycache__":
                visit(item, name + "/")
            elif item.is_file() and item.name.endswith((".py", ".json", ".txt")):
                files[name] = hashlib.sha256(item.read_bytes()).hexdigest()

    visit(root)
    return {"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), "files": files}


def provenance(engine):
    versions = {}
    for package in ("torch", "transformers", "numpy", "pydivsufsort", "fastapi", "uvicorn"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return {"version": __version__, "engine": engine.describe(), "packages": versions, "source": source_digest(),
            "timing": "Synchronized engine latency includes snapshot, tokenization, indexing, generation and decode; load/warmup excluded"}


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def summarize(rows, methods, *, baseline="AR", synthetic=False):
    result = {"synthetic": synthetic, "baseline": baseline, "methods": {},
              "percentile_policy": "P95 requires >=20 observations; P99 requires >=100. Repeats are not independent requests."}
    for method in methods:
        selected = [r for r in rows if r["method"] == method]
        if not selected:
            raise ValueError("Missing benchmark method: " + method)
        latencies = [r["latency_seconds"] for r in selected]
        total = sum(latencies)
        ttft = [r["ttft_seconds"] for r in selected if r["ttft_seconds"] is not None]
        tpot = [(r["latency_seconds"] - r["ttft_seconds"]) / (len(r["token_ids"]) - 1)
                for r in selected if r["ttft_seconds"] is not None and len(r["token_ids"]) > 1]
        result["methods"][method] = dict(observations=len(selected), independent_requests=len({r["request_id"] for r in selected}),
                                         total_seconds=total, output_tokens=sum(len(r["token_ids"]) for r in selected),
                                         tokens_per_second=sum(len(r["token_ids"]) for r in selected) / total if total else None,
                                         p50_seconds=percentile(latencies, 0.5),
                                         p95_seconds=percentile(latencies, 0.95) if len(selected) >= 20 else None,
                                         p99_seconds=percentile(latencies, 0.99) if len(selected) >= 100 else None,
                                         mean_ttft_seconds=statistics.mean(ttft) if ttft else None,
                                         mean_tpot_seconds=statistics.mean(tpot) if tpot else None,
                                         peak_gpu_bytes=max(r["peak_gpu_bytes"] for r in selected),
                                         strict_correct=all(r["correct"] for r in selected))
    base = result["methods"][baseline]["total_seconds"]
    for value in result["methods"].values():
        value["speedup"] = base / value["total_seconds"] if value["total_seconds"] else None
    return result


def compare(result, reference):
    return (not getattr(result, "approximate", False) and result.token_ids == reference.token_ids and result.prompt_sha256 == reference.prompt_sha256
            and result.snapshot_id == reference.snapshot_id
            and getattr(result, "execution_profile", "eager") == getattr(reference, "execution_profile", "eager"))


def benchmark(engine, requests, methods, output, *, repeats=3, warmup=True):
    requests, methods = list(requests), list(methods)
    if not requests or len({r.id for r in requests}) != len(requests):
        raise ValueError("Provide nonempty requests with unique IDs")
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(engine.methods):
        raise ValueError("Benchmark methods must be unique and supported by the engine")
    if type(repeats) is not int or not 1 <= repeats <= 1000:
        raise ValueError("repeats must be in [1, 1000]")
    if set(methods) & set(engine.describe().get("approximate_methods", [])):
        raise ValueError("Strict benchmark excludes approximate methods; use the ASD runtime acceptance runner")
    if "AR" not in methods:
        methods.insert(0, "AR")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    info = {**provenance(engine), "status": "started", "repeats": repeats, "methods": methods,
            "request_sha256": hashlib.sha256(json.dumps([asdict(r) for r in requests], sort_keys=True).encode()).hexdigest()}
    write_json(output / "provenance.json", info)
    rows = []
    try:
        if warmup:
            request = GenerationRequest("__warmup__", "__warmup__", 0,
                                        ({"role": "user", "content": "Say hello."},),
                                        min(8, min(r.max_new_tokens for r in requests)))
            for method in methods:
                engine.generate(request, method)
        for request in requests:
            reference = engine.generate(request, "AR")
            for repeat in range(repeats):
                order = list(methods)
                random.Random(f"memorydraft-v0.3:{request.id}:{repeat}").shuffle(order)
                for method in order:
                    result = engine.generate(request, method)
                    row = {**asdict(result), "repeat": repeat, "correct": compare(result, reference)}
                    # Text stays local in responses; benchmark artifacts use token IDs and hashes.
                    row.pop("text")
                    with (output / "rows.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
                    if not row["correct"]:
                        write_json(output / "failure.json", {"result": row, "reference": asdict(reference)})
                        raise RuntimeError("Strict token/prompt/snapshot mismatch: " + request.id + "/" + method)
        summary = summarize(rows, methods, synthetic=engine.describe().get("synthetic", False))
        write_json(output / "summary.json", summary)
        info["status"] = "complete"
        return summary
    except Exception as error:
        info.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        info["rows"] = len(rows)
        write_json(output / "provenance.json", info)


def profile_request(engine, request, method, output):
    if engine.describe().get("synthetic"):
        raise ValueError("Profiler requires a real model backend")
    import torch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    engine.generate(request, method)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if engine.target.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    started = time.perf_counter()
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
        result = engine.generate(request, method, profile=True)
    profiler.export_chrome_trace(str(output / "trace.json"))
    events = sorted(profiler.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
    write_json(output / "operators.json", [{"name": e.key, "calls": e.count,
                                            "device_type": str(e.device_type),
                                            "self_cpu_us": e.self_cpu_time_total,
                                            "self_device_us": getattr(e, "self_device_time_total", 0)} for e in events])
    write_json(output / "provenance.json", {**provenance(engine), "instrumented": True,
                                           "note": "Profiler overhead included; not a benchmark result",
                                           "wall_seconds": time.perf_counter() - started,
                                           "request_id": request.id, "method": method,
                                           "prompt_sha256": result.prompt_sha256, "output_tokens": len(result.token_ids)})
    return {"trace": str(output / "trace.json"), "output_tokens": len(result.token_ids), "instrumented": True}


def check_regression(current, baseline, *, max_slowdown=.10):
    """Reject incomparable runs before evaluating a caller-selected timing budget."""
    current, baseline = Path(current), Path(baseline)
    if not math.isfinite(max_slowdown) or max_slowdown < 0:
        raise ValueError("max_slowdown must be finite and nonnegative")
    runs = []
    for directory in (current, baseline):
        info = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        if info["status"] != "complete" or summary["synthetic"]:
            raise ValueError("Regression requires completed real-model runs")
        if not rows or any(not row["correct"] or row.get("approximate", False) for row in rows):
            raise ValueError("Regression requires strictly correct nonempty rows")
        runs.append((info, summary, rows))
    new, old = runs
    for key in ("request_sha256", "repeats"):
        if new[0][key] != old[0][key]:
            raise ValueError("Incomparable regression inputs: " + key)
    for key in ("backend", "torch", "transformers", "gpu", "device", "dtype", "target_revision"):
        if key not in new[0]["engine"] or new[0]["engine"][key] != old[0]["engine"].get(key):
            raise ValueError("Incomparable regression environment: " + key)
    if new[0]["engine"].get("execution_profile", "eager") != old[0]["engine"].get("execution_profile", "eager"):
        raise ValueError("Incomparable regression environment: execution_profile")
    if ((new[0]["engine"].get("draft_adapter") or {}).get("sha256") !=
            (old[0]["engine"].get("draft_adapter") or {}).get("sha256")):
        raise ValueError("Incomparable regression environment: draft_adapter")
    if not new[0]["engine"]["target_revision"]:
        raise ValueError("Regression requires a pinned Target revision")

    def keyed(rows):
        result = {(row["request_id"], row["repeat"], row["method"]): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate regression observations")
        return result

    new_rows, old_rows = keyed(new[2]), keyed(old[2])
    if new_rows.keys() != old_rows.keys():
        raise ValueError("Incomparable regression method/request pairs")
    for key in new_rows:
        for field in ("token_ids", "prompt_sha256", "snapshot_id"):
            if new_rows[key][field] != old_rows[key][field]:
                raise ValueError("Incomparable regression outputs or snapshots: " + field)
    methods = {}
    for method in new[1]["methods"]:
        new_seconds = sum(r["latency_seconds"] for r in new_rows.values() if r["method"] == method)
        old_seconds = sum(r["latency_seconds"] for r in old_rows.values() if r["method"] == method)
        if old_seconds <= 0 or new_seconds <= 0:
            raise ValueError("Regression timings must be positive")
        ratio = new_seconds / old_seconds
        methods[method] = {"current_over_baseline": ratio, "passed": ratio <= 1 + max_slowdown}
    report = {"passed": all(v["passed"] for v in methods.values()), "max_slowdown": max_slowdown,
              "methods": methods, "baseline_source": old[0]["source"]["sha256"],
              "current_source": new[0]["source"]["sha256"],
              "note": "Engineering threshold only; not a statistical significance test"}
    write_json(current / "regression.json", report)
    return report
