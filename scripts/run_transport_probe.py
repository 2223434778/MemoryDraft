"""Paired real-model OT-R pilot on frozen, previously observed workloads."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time

import ot

from memorydraft.evaluation.runtime_migration import check_result, file_hash, input_hashes
from memorydraft.evaluation.transport_probe import METHODS, REPEATS, audit, probe_inputs, task_definitions
from memorydraft.memory.store import MemoryStore
from memorydraft.runtime.benchmark import provenance, write_json
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest
from scripts.run_runtime_migration import append


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = {name: input_hashes(ROOT / "results" / name) for name in ("reuse-confirmation", "infra-v0.3")}
    tasks, events = probe_inputs(ROOT)
    database = args.output.resolve() / "history.db"
    with MemoryStore(database) as store:
        store.append(events)
        for task in tasks:
            if store.snapshot(task.request.user_id, task.request.order).snapshot_id != task.snapshot_id:
                raise ValueError("Wrong legal snapshot")
    config = replace(RuntimeConfig.load(args.config), db=str(database), retrieval="suffix", alignment={})
    if config.backend != "dflash" or config.reuse != {"max_tokens": 32, "min_match": 6, "cooldown_rounds": 1, "max_pending_tokens": 1024}:
        raise ValueError("Use the frozen DFlash settings")
    started = time.perf_counter()
    engine = load_engine(config)
    info = {**provenance(engine), "status": "started", "methods": METHODS, "repeats": REPEATS,
            "load_seconds": time.perf_counter() - started, "POT": ot.__version__, "original_sha256": original,
            "runner_sha256": file_hash(Path(__file__)), "protocol_sha256": file_hash(ROOT / "docs/OT_R_PROTOCOL.md"),
            "tasks": task_definitions(tasks), "config_per_arm": {m: asdict(replace(config, retrieval={"Edit-R": "edit", "OT-R": "ot"}.get(m, "suffix"))) for m in METHODS}}
    write_json(args.output / "provenance.json", info)

    def execute(request, method):
        engine.config = replace(config, retrieval={"Edit-R": "edit", "OT-R": "ot"}.get(method, "suffix"))
        return engine.generate(request, "DFlash" if method == "DFlash" else "R+DFlash")

    rows = []
    try:
        warm = GenerationRequest("__warmup__", "__warmup__", 0,
                                 ({"role": "user", "content": "Say hello."},), 16)
        for method in METHODS:
            execute(warm, method)
        for task in tasks:
            result = engine.generate(task.request, "AR")
            check_result(result, task)
            append(args.output / "fresh-ar.jsonl", asdict(result))
            print("AR checked", task.request.id, flush=True)
        info["fresh_ar_correct"] = len(tasks)
        write_json(args.output / "provenance.json", info)
        for task in tasks:
            for repeat in range(REPEATS):
                methods = list(METHODS)
                random.Random(f"ot-r-v1:{task.request.id}:{repeat}").shuffle(methods)
                for method in methods:
                    result = execute(task.request, method)
                    check_result(result, task)
                    row = {**asdict(result), "method": method, "correct": True, "repeat": repeat,
                           "project_id": task.project_id, "budget": task.request.max_new_tokens}
                    row.pop("text")
                    rows.append(row)
                    append(args.output / "rows.jsonl", row)
            print("completed", task.request.id, flush=True)
        if original != {name: input_hashes(ROOT / "results" / name) for name in original}:
            raise ValueError("Original evidence changed")
        summary = audit(rows, tasks)
        write_json(args.output / "summary.json", summary)
        info.update(status="complete", original_unchanged=True, rows=len(rows))
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        info.update(status="failed", error_type=type(error).__name__, error=str(error), rows=len(rows))
        raise
    finally:
        engine.close()
        write_json(args.output / "provenance.json", info)


if __name__ == "__main__":
    main()
