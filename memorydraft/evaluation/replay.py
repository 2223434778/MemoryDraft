from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

from memorydraft.memory.store import MemoryStore, Snapshot


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def controlled_snapshot(snapshot, tokenizer, condition):
    if condition == "relevant":
        return snapshot
    if condition == "empty":
        return Snapshot(snapshot.user_id, snapshot.before_order, ())
    if condition != "unrelated":
        raise ValueError("Unknown history condition")
    records = []
    filler = tokenizer.encode(" The botanical garden contains ferns, orchids and moss beside a quiet pond.", add_special_tokens=False)
    for record in snapshot.records:
        count = len(tokenizer.encode(record.text, add_special_tokens=False))
        ids = (filler * (count // len(filler) + 1))[:count]
        records.append(replace(record, record_id=record.record_id + ":synthetic-unrelated", text=tokenizer.decode(ids)))
    return Snapshot(snapshot.user_id, snapshot.before_order, tuple(records))


def replay(backend, store: MemoryStore, requests, methods, output, *, repeats=1, encoder=None,
           history_condition="relevant", **options):
    if repeats < 1 or "AR" not in methods or len(set(methods)) != len(methods):
        raise ValueError("Replay requires unique methods, AR reference and positive repeats")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    results = []
    with Path(output).open("w", encoding="utf-8") as stream:
        for request in requests:
            # Frozen snapshot for every method/repeat. Never write test answers here.
            snapshot_start = time.perf_counter()
            snapshot = store.snapshot(request["user_id"], request["order"])
            snapshot = controlled_snapshot(snapshot, backend.tokenizer, history_condition)
            snapshot_seconds = time.perf_counter() - snapshot_start
            # Separate untimed reference, so measured AR can participate in randomized order.
            reference = backend.generate(request["messages"], snapshot, method="AR", encoder=encoder, **options).token_ids
            for repeat in range(repeats):
                shuffled = list(methods)
                random.Random(repeat + request["order"]).shuffle(shuffled)
                for method in shuffled:
                    result = backend.generate(request["messages"], snapshot, method=method, encoder=encoder, **options)
                    result.latency_seconds += snapshot_seconds
                    if result.ttft_seconds is not None:
                        result.ttft_seconds += snapshot_seconds
                    result.stage_seconds["snapshot_shared"] = snapshot_seconds
                    row = asdict(result)
                    row.update(request_id=request["id"], scenario=request.get("scenario", "unspecified"),
                               repeat=repeat, correct=result.token_ids == reference, history_condition=history_condition)
                    if not row["correct"]:
                        row["reference_token_ids"] = reference
                        row["first_difference"] = next((i for i, (a, b) in enumerate(zip(reference, result.token_ids)) if a != b),
                                                       min(len(reference), len(result.token_ids)))
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    results.append(row)
                    if not row["correct"]:
                        raise RuntimeError(f"Greedy mismatch: {request['id']} / {method}; result saved")
    return summarize(results)


def summarize(rows):
    by_method = {}
    reference = {(r["request_id"], r["repeat"]): r for r in rows if r["method"] == "AR"}
    for method in sorted({r["method"] for r in rows}):
        group = [r for r in rows if r["method"] == method]
        keys = [(r["request_id"], r["repeat"]) for r in group]
        if len(keys) != len(set(keys)) or any(key not in reference for key in keys):
            raise ValueError("Unpaired or duplicate replay rows")
        elapsed = sum(r["latency_seconds"] for r in group)
        ar_time = sum(reference[key]["latency_seconds"] for key in keys)
        by_method[method] = {
            "requests": len(group), "total_seconds": elapsed,
            "speedup_vs_ar": ar_time / elapsed, "all_correct": all(r["correct"] for r in group),
            "generated_tokens": sum(len(r["token_ids"]) for r in group),
            "median_latency_seconds": statistics.median(r["latency_seconds"] for r in group),
            "routes": {route: {
                metric: sum(r.get("route_stats", {}).get(route, {}).get(metric, 0) for r in group)
                for metric in ("rounds", "proposed", "accepted", "committed")
            } for route in ("R", "H", "D", "AR")},
        }
    return by_method
