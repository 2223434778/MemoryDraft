"""Full development natural-EOS comparison; validation/reserve are not run."""
from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import random
import time

import torch

from memorydraft.backends.regional_compile import RegionalRMSNorm
from memorydraft.evaluation.asd_suite import canonical, score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_block_suite import records_and_positions
from scripts.run_asd_budget_probe import protected
from scripts.run_asd_compile_probe import METHODS, execute, trajectory_comparison
from scripts.run_asd_execution_probe import require_same
from scripts.run_asd_rmsnorm_probe import PROFILES
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 4096
REPEATS = 3
WORKER_TIMEOUT = 5400


def previous_rows():
    return {(r["request_id"], r["method"]): r for i in (0, 1)
            for r in read_rows(ROOT/f"results/asd-block-suite/part-{i}/rows.jsonl")
            if r["stage"] == "natural" and r["repeat"] == 0 and r["experimental_block_size"] == 16}


def timing_order(q, repeat, methods=METHODS):
    rng = random.Random(f"rmsnorm-suite:{q}:{repeat}")
    methods = list(methods)
    rng.shuffle(methods)
    order = []
    for method in methods:
        pair = list(PROFILES)
        rng.shuffle(pair)
        order.extend((method, p) for p in pair)
    return order


def run_question(engine, compiler, record, previous, output):
    """First timed repetition is also scored; its online costs stay in estimates."""
    q = record["id"]
    seen, quality_cache = {}, {}

    def finish(row, stage, repeat, slot, eager):
        key = row["method"], row["profile"]
        if repeat == 0:
            if previous is None:
                trajectory_comparison(eager, row)
            elif row["profile"] == "eager":
                require_same(row, previous[q, row["method"]])
            else:
                trajectory_comparison(previous[q, row["method"]], row)
            if row["text"] not in quality_cache:
                quality_cache[row["text"]] = score(record, row["text"])
            row["quality"] = quality_cache[row["text"]]
            if previous is not None and row["profile"] == "eager" and row["quality"] != previous[q, row["method"]]["quality"]:
                raise ValueError("Eager natural baseline quality changed")
            seen[key] = row
        else:
            require_same(row, seen[key])
            row["quality"] = seen[key]["quality"]
        row.update(stage=stage, repeat=repeat, slot=slot)
        append(output/("references.jsonl" if stage == "reference" else "rows.jsonl"), row)

    # Scoring happens after BOTH calls of a profile pair, outside engine timing.
    for repeat, methods, stage in ((0, ("AR",), "reference"),
                                   *((i, METHODS, "natural") for i in range(REPEATS))):
        order = timing_order(q, -1 if stage == "reference" else repeat, methods)
        for index in range(0, len(order), 2):
            pair = [(slot, execute(engine, compiler, record, method, profile, LIMIT))
                    for slot, (method, profile) in enumerate(order[index:index+2], index)]
            eager = next(row for _, row in pair if row["profile"] == "eager")
            for slot, row in pair:
                finish(row, stage, repeat, slot, eager)
    return dict(reference=2, timed=2*len(METHODS)*REPEATS, quality_rows=2*(len(METHODS)+1),
                unique_texts_scored=len(quality_cache))


def cache_bytes(directories):
    size = sum(p.stat().st_size for folder in directories.values() for p in Path(folder).rglob("*") if p.is_file())
    if size > 4*2**30:
        raise RuntimeError("RMSNorm caches exceeded the frozen 4 GiB budget")
    return size


def main(*, record_loader=records_and_positions, prior_loader=previous_rows, split="development",
         protocol="docs/ASD_RMSNORM_SUITE_PROTOCOL.md", worker_timeout=WORKER_TIMEOUT,
         entrypoint=None, precheck=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    records, positions = record_loader()
    if split not in {"development", "validation"} or len(records) != 48 or any(r["split"] != split for r in records):
        raise ValueError("Expected exactly the frozen development or validation split")
    records = [r for r in records if positions[r["id"]]%2 == args.shard]
    previous = prior_loader() if prior_loader is not None else None
    before = protected(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    info = dict(status="started", split=split, shard=args.shard, task_ids=[r["id"] for r in records],
                original_sha256=before, max_new_tokens=LIMIT, repeats=REPEATS,
                runner_sha256=file_hash(Path(entrypoint or __file__)), data_manifest_sha256=file_hash(ROOT/"data/research/asd-suite/manifest.json"),
                protocol_sha256=file_hash(ROOT/protocol),
                backend_graph_budget=8, backend_seconds_budget=300, worker_timeout_seconds=worker_timeout,
                cache_dirs={k: os.environ[k] for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")})
    counts = dict(warmup=0, reference=0, timed=0, quality_rows=0, unique_texts_scored=0)
    compiler = None
    write(args.output/"provenance.json", info)
    try:
        if precheck is not None:
            info["initial_isolation"] = precheck(ROOT)
            write(args.output/"provenance.json", info)
        checks = [dict(request_id=r["id"], quality=score(r, canonical(r))) for r in records]
        write(args.output/"canonical-preflight.json", checks)
        if not all(c["quality"]["correct"] for c in checks):
            raise ValueError("Reference solution failed grading")
        engine = load_engine(replace(RuntimeConfig.load(args.config), db=None, max_new_tokens=LIMIT))
        if engine.draft.block_size != 16 or engine.target.dtype != torch.float32:
            raise ValueError("Expected frozen FP32 native16 models")
        info.update(runtime=provenance(engine), stop_token_ids=engine.stops)
        started = time.perf_counter()
        compiler = RegionalRMSNorm(engine.target, engine.draft)
        info["controller_setup_seconds"] = time.perf_counter()-started
        warmup = dict(benchmark="gsm8k", id="warmup", question="What is 3 plus 4?", answer="#### 7")
        started = time.perf_counter()
        for profile in PROFILES:
            for method in ("AR", *METHODS):
                row = execute(engine, compiler, warmup, method, profile, 32)
                row.update(stage="warmup", repeat=0)
                append(args.output/"warmup.jsonl", row)
                counts["warmup"] += 1
        info.update(warmup_wall_seconds=time.perf_counter()-started,
                    compiler_after_warmup=compiler.summary(), cache_bytes_after_warmup=cache_bytes(info["cache_dirs"]))
        for record in records:
            added = run_question(engine, compiler, record, previous, args.output)
            counts = {k: counts[k]+added.get(k, 0) for k in counts}
            info.update(counts=counts, compiler=compiler.summary(), cache_bytes=cache_bytes(info["cache_dirs"]))
            write(args.output/"provenance.json", info)
            print("completed", record["id"], counts, "graphs", len(compiler.backend.events), flush=True)
        info.update(status="complete")
    except Exception as error:
        info.update(status="failed", error=type(error).__name__+": "+str(error))
        raise
    finally:
        info.update(counts=counts, original_unchanged=before == protected(ROOT))
        if compiler is not None:
            info["compiler"] = compiler.summary()
        write(args.output/"provenance.json", info)
        if not info["original_unchanged"]:
            raise ValueError("Previous evidence changed")


if __name__ == "__main__":
    main()
