"""Finite real-action diagnostic on frozen development requests, preserving live KV."""
from dataclasses import asdict, replace
import argparse
import json
import os
from pathlib import Path
import random
import time
from unittest.mock import patch

from memorydraft.backends.allocation import ACTIONS, ForkAction, features
from memorydraft.evaluation.asd_suite import request_for
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import protected
from scripts.run_asd_execution_probe import selected_records
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
REPEATS, WINDOW, SOURCE_LIMIT, REQUEST_LIMIT = 3, 64, 512, 128
STATE_ROUNDS = (0, 4, 12)
REQUEST_ACTIONS = ("native", "half", "verify-half", "AR")


def references():
    return {r["request_id"]: r for i in (0, 1) for r in read_rows(
        ROOT/f"results/asd-rmsnorm-suite/part-{i}/rows.jsonl")
        if r["method"] == "DFlash" and r["profile"] == "rmsnorm-inductor" and r["repeat"] == 0}


def execute(engine, record, action, limit, *, fork=None):
    from memorydraft.runtime import engine as module
    native = module.generate_reuse
    detail = None
    def hook(*args, **kwargs):
        nonlocal detail
        kwargs.update(incremental_stop=True)
        if fork is not None:
            kwargs["round_action"] = fork
        elif action != "native":
            kwargs["round_action"] = lambda *args: action
        detail = native(*args, **kwargs)
        return detail
    with patch.object(module, "generate_reuse", hook):
        result = engine.generate(request_for(record, limit), "AR" if action == "AR" else "DFlash")
    engine.synchronize()
    local = time.perf_counter()-fork.started if fork is not None and fork.started is not None else None
    row = dict(asdict(result), benchmark=record["benchmark"], action=action, max_new_tokens=limit,
               rounds=detail.rounds if detail is not None else [], local_seconds=local)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [r for i, r in enumerate(selected_records()) if i % 2 == args.shard]
    old = references()
    before = protected(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    info = dict(status="started", task_ids=[r["id"] for r in records], split="development",
                shard=args.shard, original_sha256=before, source_limit=SOURCE_LIMIT, window=WINDOW,
                request_limit=REQUEST_LIMIT, repeats=REPEATS, state_rounds=STATE_ROUNDS)
    write(args.output/"provenance.json", info)
    engine = None
    try:
        engine = load_engine(replace(RuntimeConfig.load(args.config), db=None, max_new_tokens=4096))
        if engine.draft.block_size != 16 or engine.config.execution_profile != "rmsnorm-inductor":
            raise ValueError("Expected frozen native16 RMSNorm execution")
        info.update(runtime=provenance(engine), stop_token_ids=engine.stops)
        warm = dict(benchmark="gsm8k", id="warmup", question="What is 3 plus 4?", answer="#### 7")
        for action in (*ACTIONS, "AR"):
            append(args.output/"warmup.jsonl", execute(engine, warm, action, 32))
        info["compiler_after_warmup"] = engine._regional.summary()
        counts = dict(sources=0, requests=0, branches=0)
        for record in records:
            source = execute(engine, record, "native", SOURCE_LIMIT)
            expected = old[record["id"]]
            if (source["token_ids"] != expected["token_ids"][:SOURCE_LIMIT]
                    or source["prompt_sha256"] != expected["prompt_sha256"] or source["snapshot_id"] != expected["snapshot_id"]):
                raise ValueError("Source changed frozen development output/input: "+record["id"])
            source["full_reference_quality"] = expected["quality"]
            append(args.output/"sources.jsonl", source)
            counts["sources"] += 1
            seen = {}
            for repeat in range(REPEATS):
                actions = list(REQUEST_ACTIONS)
                random.Random(f"allocation:request:{record['id']}:{repeat}").shuffle(actions)
                for action in actions:
                    row = execute(engine, record, action, REQUEST_LIMIT)
                    row.update(stage="request", repeat=repeat, token_match=row["token_ids"] == source["token_ids"][:REQUEST_LIMIT])
                    if action in seen and row["token_ids"] != seen[action]:
                        raise ValueError("Request action repeat changed tokens")
                    seen[action] = row["token_ids"]
                    append(args.output/"requests.jsonl", row)
                    counts["requests"] += 1
            for index in STATE_ROUNDS:
                if index >= len(source["rounds"]):
                    append(args.output/"unavailable.jsonl", dict(request_id=record["id"], round=index, reason="source ended before state"))
                    continue
                boundary = source["rounds"][index]["generated_before"]
                state_features = features(boundary, source["rounds"][max(0, index-4):index], source["prompt_tokens"])
                seen = {}
                jobs = [(action, rep) for rep in range(REPEATS) for action in ACTIONS]
                random.Random(f"allocation:fork:{record['id']}:{index}").shuffle(jobs)
                for action, repeat in jobs:
                    fork = ForkAction(source["rounds"], index, action, engine.synchronize, time.perf_counter)
                    row = execute(engine, record, action, boundary+WINDOW, fork=fork)
                    if fork.started is None or row["token_ids"][:boundary] != source["token_ids"][:boundary]:
                        raise ValueError("Live-KV common prefix or source clock lost")
                    row.update(stage="branch", repeat=repeat, source_round=index, boundary=boundary, features=state_features,
                               actual_action=row["rounds"][index]["allocation"], continuation=row["token_ids"][boundary:])
                    if action in seen and (row["token_ids"], row["rounds"]) != seen[action]:
                        raise ValueError("Live-KV action repeat changed trajectory")
                    seen[action] = row["token_ids"], row["rounds"]
                    append(args.output/"branches.jsonl", row)
                    counts["branches"] += 1
            info.update(counts=counts, compiler=engine._regional.summary())
            write(args.output/"provenance.json", info)
            print("completed", record["id"], counts, flush=True)
        cache_bytes = sum(p.stat().st_size for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
                          for p in Path(os.environ[name]).rglob("*") if p.is_file())
        if cache_bytes > 4*2**30:
            raise ValueError("Allocation compilation cache exceeded budget")
        info.update(status="complete", compiler=engine._regional.summary(), cache_bytes=cache_bytes)
    except Exception as error:
        info.update(status="failed", error=type(error).__name__+": "+str(error))
        raise
    finally:
        info["original_unchanged"] = before == protected(ROOT)
        write(args.output/"provenance.json", info)
        if engine is not None:
            engine.close()
        if not info["original_unchanged"]:
            raise ValueError("Previous evidence changed")


if __name__ == "__main__":
    main()
