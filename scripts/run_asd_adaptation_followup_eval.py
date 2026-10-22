"""Four fixed dense drafters on old and independently frozen GSM questions."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random
import time

import torch

from memorydraft.evaluation.adaptation import AdapterBank, parameter_guard
from memorydraft.evaluation.adaptation_followup import STUDY, POLICIES, VARIANTS, adapter_paths, records
from memorydraft.evaluation.asd_suite import request_for, score
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_adaptation_eval import execute
from scripts.run_asd_budget_probe import protected
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    from safetensors.torch import load_file
    out = ROOT/f"results/raw/{STUDY}/eval-{args.shard}"
    out.mkdir(parents=True, exist_ok=False)
    before = protected(ROOT)
    info = dict(status="started", shard=args.shard, natural_repeats=3, prefix_repeats=2)
    engine = None
    try:
        paths = adapter_paths(ROOT)
        states = {a:load_file(str(path)) for a, (path, _) in paths.items()}
        engine = load_engine(replace(RuntimeConfig.load(ROOT/"configs/runtime-dflash-rmsnorm.json"), db=None, max_new_tokens=4096))
        engine.target.requires_grad_(False)
        engine.draft.requires_grad_(False)
        guard, bank = parameter_guard(engine.target), AdapterBank(engine.draft)
        info.update(runtime=provenance(engine), adapters={a:p[1] for a,p in paths.items()})
        setup_seconds = 0.
        def select(variant):
            nonlocal setup_seconds
            engine.synchronize()
            started = time.perf_counter()
            bank.select(states.get(variant))
            engine.synchronize()
            seconds = time.perf_counter()-started
            setup_seconds += seconds
            return seconds
        warm = dict(benchmark="gsm8k", id="warmup", question="What is 3 plus 4?", answer="#### 7")
        append(out/"warmup.jsonl", dict(asdict(engine.generate(request_for(warm, 32), "AR")), variant="base", policy="AR"))
        for variant in VARIANTS:
            for policy in POLICIES:
                elapsed = select(variant)
                append(out/"warmup.jsonl", dict(execute(engine, warm, policy, 32), variant=variant, offline_merge_seconds=elapsed))
        info["compiler_after_warmup"] = engine._regional.summary()
        old = {(r["request_id"],r["policy"]):r for i in (0,1) for r in read_rows(ROOT/f"results/asd-adaptation/eval-{i}/rows.jsonl")
               if r["variant"] == "base" and r["stage"] == "natural" and r["repeat"] == 0}
        selected = [r for i,r in enumerate(records(ROOT)) if i % 2 == args.shard]
        info["task_ids"] = [r["id"] for r in selected]
        count, peak = 0, 0
        for record in selected:
            select("base")
            reference = asdict(engine.generate(request_for(record, 4096), "AR"))
            reference.update(benchmark="gsm8k", study_split=record["study_split"], quality=score(record, reference["text"]))
            append(out/"references.jsonl", reference)
            peak = max(peak, reference["peak_gpu_bytes"])
            quality_cache = {reference["text"]:reference["quality"]}
            for stage, limit, repeats in (("natural",4096,3),("prefix",128,2)):
                for repeat in range(repeats):
                    jobs = [(v,p) for v in VARIANTS for p in POLICIES]
                    random.Random(f"adaptation-followup:{record['id']}:{stage}:{repeat}").shuffle(jobs)
                    for variant, policy in jobs:
                        elapsed = select(variant)
                        row = execute(engine, record, policy, limit)
                        same_input = (row["prompt_sha256"],row["snapshot_id"]) == (reference["prompt_sha256"],reference["snapshot_id"])
                        old_match = None
                        if variant == "base" and stage == "natural" and record["study_split"] == "observed":
                            previous = old[record["id"],policy]
                            old_match = (row["token_ids"],row["text"],row["finish_reason"]) == (previous["token_ids"],previous["text"],previous["finish_reason"])
                        row.update(variant=variant, stage=stage, repeat=repeat, study_split=record["study_split"],
                                   offline_merge_seconds=elapsed, same_input=same_input, base_reference_match=old_match,
                                   strict_token_match=(row["token_ids"] == reference["token_ids"][:limit] and
                                       (stage != "natural" or (row["text"],row["finish_reason"]) == (reference["text"],reference["finish_reason"]))),
                                   adapter_sha256=paths[variant][1]["adapter_sha256"] if variant != "base" else None)
                        if stage == "natural":
                            if row["text"] not in quality_cache:
                                quality_cache[row["text"]] = score(record,row["text"])
                            row["quality"] = quality_cache[row["text"]]
                        append(out/"rows.jsonl",row)
                        count += 1
                        peak = max(peak,row["peak_gpu_bytes"])
            info.update(count=count,peak_gpu_bytes=peak,compiler=engine._regional.summary())
            write(out/"provenance.json",info)
            print("evaluated",record["study_split"],record["id"],count,flush=True)
        select("base")
        if any(not torch.equal(engine.draft.get_submodule(n).weight.detach().cpu(),w) for n,w in bank.original.items()):
            raise ValueError("Dense base draft not restored")
        if parameter_guard(engine.target) != guard:
            raise ValueError("Target changed during evaluation")
        cache_bytes = sum(p.stat().st_size for name in ("TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR")
                          for p in Path(os.environ[name]).rglob("*") if p.is_file())
        if cache_bytes > 4*2**30 or not 0 < peak <= 46*2**30:
            raise ValueError("Evaluation resource budget exceeded")
        info.update(status="complete", count=count, compiler=engine._regional.summary(), cache_bytes=cache_bytes,
                    offline_merge_seconds=setup_seconds, base_restored=True,teacher_parameter_versions_unchanged=True)
    except Exception as error:
        info.update(status="failed",error=type(error).__name__+": "+str(error))
        raise
    finally:
        info["original_unchanged"] = before == protected(ROOT)
        write(out/"provenance.json",info)
        if engine is not None:
            engine.close()
        if not info["original_unchanged"]:
            raise ValueError("Previous evidence changed")


if __name__ == "__main__":
    main()
