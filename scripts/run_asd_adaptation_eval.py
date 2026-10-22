"""Matched baseline/ordinary/ASD-prefix dense drafters, natural and equal-prefix."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random
import time
from unittest.mock import patch

import torch

from memorydraft.backends.asd import ASDVerifier
from memorydraft.evaluation.adaptation import ARMS, AdapterBank, digest, parameter_guard
from memorydraft.evaluation.asd_suite import BENCHMARKS, load_suite, request_for, score
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import POLICIES, protected
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
VARIANTS, POLICIES_USED = ("base", *ARMS), ("DFlash", "ASD")


def records():
    rows = load_suite(ROOT/"data/research/asd-suite", "development")
    return [r for b in BENCHMARKS for r in [r for r in rows if r["benchmark"] == b][:(8 if b == "gsm8k" else 2)]]


def references():
    return {(r["request_id"], r["method"]): r for i in (0, 1) for r in read_rows(
        ROOT/f"results/asd-rmsnorm-suite/part-{i}/rows.jsonl") if r["profile"] == "rmsnorm-inductor"
        and r["method"] in POLICIES_USED and r["repeat"] == 0}


def execute(engine, record, policy_name, limit):
    from memorydraft.runtime import engine as module
    native, detail = module.generate_reuse, None
    policy = ASDVerifier(POLICIES[policy_name])
    def hook(*args, **kwargs):
        nonlocal detail
        kwargs["incremental_stop"] = True
        if policy_name == "ASD":
            kwargs["verifier"] = policy
        detail = native(*args, **kwargs)
        return detail
    with patch.object(module, "generate_reuse", hook):
        result = engine.generate(request_for(record, limit), "DFlash")
    return dict(asdict(result), benchmark=record["benchmark"], policy=policy_name,
                asd=policy.summary() if policy_name == "ASD" else None, rounds=detail.rounds, max_new_tokens=limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    from safetensors.torch import load_file
    out = ROOT/f"results/raw/asd-adaptation/eval-{args.shard}"
    out.mkdir(parents=True, exist_ok=False)
    before = protected(ROOT)
    info = dict(status="started", shard=args.shard, original_sha256=before, split="development", natural_repeats=3, prefix_repeats=2)
    write(out/"provenance.json", info)
    engine = None
    try:
        states, adapters = {}, {}
        for arm in ARMS:
            meta = json.loads((ROOT/f"results/raw/asd-adaptation/train-{arm}/provenance.json").read_text(encoding="utf-8"))
            if meta["status"] != "complete":
                raise ValueError("Cannot evaluate incomplete adaptation")
            path = ROOT/meta["adapter_path"]
            if digest(path.read_bytes()) != meta["adapter_sha256"]:
                raise ValueError("Selected adapter changed")
            states[arm] = load_file(str(path))
            adapters[arm] = {k:meta[k] for k in ("adapter_sha256", "selected_step", "selected_calibration_loss")}
        engine = load_engine(replace(RuntimeConfig.load(ROOT/"configs/runtime-dflash-rmsnorm.json"), db=None, max_new_tokens=4096))
        engine.target.requires_grad_(False)
        engine.draft.requires_grad_(False)
        guard = parameter_guard(engine.target)
        bank = AdapterBank(engine.draft)
        info.update(runtime=provenance(engine), adapters=adapters)
        setup_seconds = 0.
        def select(variant):
            nonlocal setup_seconds
            engine.synchronize()
            started = time.perf_counter()
            bank.select(states.get(variant))
            engine.synchronize()
            elapsed = time.perf_counter()-started
            setup_seconds += elapsed
            return elapsed
        warm = dict(benchmark="gsm8k", id="warmup", question="What is 3 plus 4?", answer="#### 7")
        for variant in VARIANTS:
            for policy in POLICIES_USED:
                elapsed = select(variant)
                append(out/"warmup.jsonl", dict(execute(engine, warm, policy, 32), variant=variant, offline_merge_seconds=elapsed))
        info["compiler_after_warmup"] = engine._regional.summary()
        old, count, peak = references(), 0, 0
        selected = [r for i, r in enumerate(records()) if i % 2 == args.shard]
        info["task_ids"] = [r["id"] for r in selected]
        for record in selected:
            quality_cache = {}
            for stage, limit, repeats in (("natural", 4096, 3), ("prefix", 128, 2)):
                for repeat in range(repeats):
                    jobs = [(v, p) for v in VARIANTS for p in POLICIES_USED]
                    random.Random(f"adaptation:{record['id']}:{stage}:{repeat}").shuffle(jobs)
                    for variant, policy in jobs:
                        elapsed = select(variant)
                        row = execute(engine, record, policy, limit)
                        reference = old[(record["id"], "DFlash" if policy == "DFlash" else "ASD")]
                        strict_reference = old[(record["id"], "DFlash")]
                        same_input = row["prompt_sha256"] == reference["prompt_sha256"] and row["snapshot_id"] == reference["snapshot_id"]
                        row.update(variant=variant, stage=stage, repeat=repeat, offline_merge_seconds=elapsed, same_input=same_input,
                                   strict_token_match=(row["token_ids"] == strict_reference["token_ids"][:limit]
                                       and (stage != "natural" or (row["text"] == strict_reference["text"] and row["finish_reason"] == strict_reference["finish_reason"]))),
                                   base_reference_match=(row["token_ids"] == reference["token_ids"] and row["text"] == reference["text"] and row["finish_reason"] == reference["finish_reason"]) if stage == "natural" and variant == "base" else None)
                        if stage == "natural":
                            if row["text"] not in quality_cache:
                                quality_cache[row["text"]] = score(record, row["text"])
                            row["quality"] = quality_cache[row["text"]]
                        append(out/"rows.jsonl", row)
                        count += 1
                        peak = max(peak, row["peak_gpu_bytes"])
            info.update(count=count, peak_gpu_bytes=peak, compiler=engine._regional.summary())
            write(out/"provenance.json", info)
            print("evaluated", record["id"], count, flush=True)
        select("base")
        if any(not torch.equal(engine.draft.get_submodule(n).weight.detach().cpu(), w) for n, w in bank.original.items()):
            raise ValueError("Dense merge did not restore base draft")
        if parameter_guard(engine.target) != guard:
            raise ValueError("Target changed while evaluating adapters")
        cache_bytes = sum(p.stat().st_size for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
                          for p in Path(os.environ[name]).rglob("*") if p.is_file())
        if cache_bytes > 4*2**30 or not 0 < peak <= 46*2**30:
            raise ValueError("Evaluation resource budget exceeded")
        info.update(status="complete", count=count, compiler=engine._regional.summary(), cache_bytes=cache_bytes,
                    offline_merge_seconds=setup_seconds, base_restored=True, teacher_parameter_versions_unchanged=True)
    except Exception as error:
        info.update(status="failed", error=type(error).__name__+": "+str(error))
        raise
    finally:
        info["original_unchanged"] = before == protected(ROOT)
        write(out/"provenance.json", info)
        if engine is not None:
            engine.close()
        if not info["original_unchanged"]:
            raise ValueError("Previous evidence changed")


if __name__ == "__main__":
    main()
