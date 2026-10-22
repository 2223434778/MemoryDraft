"""Natural-EOS strict candidate trees and one-shot suffix repair, fixed FP32."""
from dataclasses import asdict, replace
import argparse
import json
import os
from pathlib import Path
import random
from unittest.mock import patch

from memorydraft.backends.candidate_repair import SuffixRepair
from memorydraft.backends.dflash_reuse import ReuseConfig
from memorydraft.backends.draft_tree import TreeConfig
from memorydraft.evaluation.asd_suite import load_suite, request_for, score
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_allocation_probe import references
from scripts.run_asd_budget_probe import protected
from scripts.run_asd_execution_probe import selected_records
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("native", "packed-linear", "tree32", "tree48", "repair4", "repair8")
REPEATS, LIMIT = 3, 4096


def execute(engine, record, method, limit=LIMIT):
    if method not in METHODS:
        raise ValueError("Unknown candidate method")
    from memorydraft.runtime import engine as module
    native, detail = module.generate_reuse, None
    def hook(*args, **kwargs):
        nonlocal detail
        kwargs["incremental_stop"] = True
        if method.startswith("repair"):
            kwargs.update(matcher=SuffixRepair(int(method[6:])), config=ReuseConfig(cooldown_rounds=0))
        elif method != "native":
            kwargs["tree_config"] = {"packed-linear": TreeConfig(0), "tree32": TreeConfig(4, 4),
                                     "tree48": TreeConfig(4, 8)}[method]
        detail = native(*args, **kwargs)
        return detail
    with patch.object(module, "generate_reuse", hook):
        result = engine.generate(request_for(record, limit), "DFlash")
    engine.synchronize()
    return dict(asdict(result), benchmark=record["benchmark"], candidate=method,
                rounds=detail.rounds, max_new_tokens=limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("pilot", "expanded"), default="pilot")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods) or "native" not in args.methods:
        raise ValueError("Candidate comparison needs unique methods including native")
    if args.scope == "pilot" and tuple(args.methods) != METHODS:
        raise ValueError("Pilot method matrix is frozen")
    records = selected_records() if args.scope == "pilot" else load_suite(ROOT/"data/research/asd-suite", "development")
    records = [r for i, r in enumerate(records) if i % 2 == args.shard]
    old, before = references(), protected(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    info = dict(status="started", scope=args.scope, split="development", task_ids=[r["id"] for r in records],
                methods=args.methods, shard=args.shard, repeats=REPEATS, limit=LIMIT, original_sha256=before)
    write(args.output/"provenance.json", info)
    engine = None
    try:
        engine = load_engine(replace(RuntimeConfig.load(args.config), db=None, max_new_tokens=LIMIT))
        if engine.draft.block_size != 16 or engine.config.execution_profile != "rmsnorm-inductor":
            raise ValueError("Expected frozen native16 RMSNorm runtime")
        info.update(runtime=provenance(engine), stop_token_ids=engine.stops)
        warm = dict(benchmark="gsm8k", id="warmup", question="What is 3 plus 4?", answer="#### 7")
        for method in args.methods:
            append(args.output/"warmup.jsonl", execute(engine, warm, method, 32))
        info["compiler_after_warmup"] = engine._regional.summary()
        count = 0
        for record in records:
            expected = old[record["id"]]
            quality_cache = {}
            for repeat in range(REPEATS):
                methods = list(args.methods)
                random.Random(f"candidate:{args.scope}:{record['id']}:{repeat}").shuffle(methods)
                for method in methods:
                    row = execute(engine, record, method)
                    if (row["prompt_sha256"] != expected["prompt_sha256"] or row["snapshot_id"] != expected["snapshot_id"]):
                        raise ValueError("Frozen candidate input changed: "+record["id"])
                    if row["text"] not in quality_cache:
                        quality_cache[row["text"]] = score(record, row["text"])
                    row.update(repeat=repeat, scope=args.scope, quality=quality_cache[row["text"]],
                               token_match=row["token_ids"] == expected["token_ids"],
                               text_match=row["text"] == expected["text"],
                               finish_match=row["finish_reason"] == expected["finish_reason"],
                               full_reference_quality=expected["quality"])
                    append(args.output/"rows.jsonl", row)
                    count += 1
            info.update(count=count, compiler=engine._regional.summary())
            write(args.output/"provenance.json", info)
            print("completed", record["id"], count, flush=True)
        cache_bytes = sum(p.stat().st_size for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
                          for p in Path(os.environ[name]).rglob("*") if p.is_file())
        if cache_bytes > 4*2**30:
            raise ValueError("Candidate compilation cache exceeded budget")
        info.update(status="complete", count=count, compiler=engine._regional.summary(), cache_bytes=cache_bytes)
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
