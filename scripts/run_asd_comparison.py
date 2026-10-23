"""Same-card comparison of frozen acceptance policies and dense draft adapters."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random
import time
from unittest.mock import patch

import torch

from memorydraft.backends.asd_confidence import ConfidenceUtilityVerifier
from memorydraft.backends.asd_cpu import CPUASDVerifier
from memorydraft.backends.draft_tree import TreeConfig
from memorydraft.evaluation.adaptation import AdapterBank, digest, parameter_guard
from memorydraft.evaluation.asd_comparison import ADAPTERS, ARMS, REPEATS, STUDY, arms_for, records
from memorydraft.evaluation.asd_suite import canonical, request_for, score
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import POLICIES, protected
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]


def execute(engine, record, arm, limit):
    if arm not in ARMS:
        raise ValueError("Unknown frozen arm")
    engine.synchronize()
    started = time.perf_counter()
    _, method = arm.split("/")
    verifier = None
    if method == "Confidence-Utility":
        verifier = ConfidenceUtilityVerifier(POLICIES["ASD-Utility"], alpha=.5)
    elif method not in ("DFlash", "tree32"):
        verifier = CPUASDVerifier(POLICIES[method])
    from memorydraft.runtime import engine as module
    native, detail = module.generate_reuse, None
    def hook(*args, **kwargs):
        nonlocal detail
        kwargs["incremental_stop"] = True
        if verifier is not None:
            kwargs["verifier"] = verifier
        if method == "tree32":
            kwargs["tree_config"] = TreeConfig(4, 4)
        detail = native(*args, **kwargs)
        return detail
    with patch.object(module, "generate_reuse", hook):
        result = engine.generate(request_for(record, limit), "DFlash")
    elapsed = time.perf_counter()-started
    row = asdict(result)
    row.update(engine_latency_seconds=row["latency_seconds"], latency_seconds=elapsed)
    return dict(row, benchmark=record["benchmark"], arm=arm,
                asd=verifier.summary() if verifier else None, rounds=detail.rounds, max_new_tokens=limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=REPEATS, required=True)
    parser.add_argument("--shard", type=int, choices=(0,1), required=True)
    args = parser.parse_args()
    from safetensors.torch import load_file
    out = ROOT/f"results/raw/{STUDY}/{args.phase}-{args.shard}"
    out.mkdir(parents=True, exist_ok=False)
    selection_path = ROOT/f"results/raw/{STUDY}/selection.json"
    selection = json.loads(selection_path.read_text()) if args.phase == "confirm" else None
    before, engine = protected(ROOT), None
    info = dict(status="started", phase=args.phase, shard=args.shard, selection=selection,
                repeats=REPEATS[args.phase], selection_sha256=digest(selection_path.read_bytes()) if selection else None)
    write(out/"provenance.json", info)
    try:
        paths = {v: ROOT.parent/n for v,(n,_) in ADAPTERS.items()}
        for v,path in paths.items():
            if digest(path.read_bytes()) != ADAPTERS[v][1]:
                raise ValueError("Frozen adapter changed: "+v)
        states = {v: load_file(str(p)) for v,p in paths.items()}
        engine = load_engine(replace(RuntimeConfig.load(ROOT/"configs/runtime-dflash-rmsnorm.json"), db=None, max_new_tokens=4096))
        engine.target.requires_grad_(False)
        engine.draft.requires_grad_(False)
        bank, guard = AdapterBank(engine.draft), parameter_guard(engine.target)
        if engine.draft.block_size != 16:
            raise ValueError("Comparison requires native16")
        merge_seconds = 0.
        current = None
        def select(variant):
            nonlocal merge_seconds, current
            if current == variant:
                return 0.
            engine.synchronize()
            started = time.perf_counter()
            bank.select(states.get(variant))
            engine.synchronize()
            elapsed = time.perf_counter()-started
            merge_seconds += elapsed
            current = variant
            return elapsed
        warm = dict(benchmark="gsm8k", id="warmup", question="What is 3 plus 4?", answer="#### 7")
        append(out/"warmup.jsonl", asdict(engine.generate(request_for(warm,32), "AR")))
        for arm in ARMS:
            select(arm.split("/")[0])
            append(out/"warmup.jsonl", execute(engine,warm,arm,32))
        info.update(runtime=provenance(engine), compiler_after_warmup=engine._regional.summary(), adapters={v:h for v,(_,h) in ADAPTERS.items()})
        selected = [r for b in ("gsm8k","math500","mbpp","humaneval","squad","json")
                    for i,r in enumerate(r for r in records(ROOT,args.phase) if r["benchmark"] == b) if i%2 == args.shard]
        info["task_ids"] = [r["id"] for r in selected]
        preflight = [{"request_id":r["id"], "quality":score(r,canonical(r))} for r in selected]
        write(out/"canonical-preflight.json", preflight)
        if not all(r["quality"]["correct"] for r in preflight):
            raise ValueError("Reference scoring preflight failed")
        count, peak = 0, 0
        for record in selected:
            select("base")
            reference = asdict(engine.generate(request_for(record,4096),"AR"))
            reference.update(benchmark=record["benchmark"], quality=score(record,reference["text"]))
            append(out/"references.jsonl",reference)
            quality_cache = {reference["text"]:reference["quality"]}
            seen = {}
            for stage, repeats in REPEATS[args.phase].items():
                limit = 4096 if stage == "natural" else 128
                for repeat in range(repeats):
                    jobs = list(arms_for(record["benchmark"],args.phase,selection))
                    random.Random(f"comparison:{args.phase}:{record['id']}:{stage}:{repeat}").shuffle(jobs)
                    for arm in jobs:
                        elapsed = select(arm.split("/")[0])
                        row = execute(engine,record,arm,limit)
                        row.update(phase=args.phase,stage=stage,repeat=repeat,offline_merge_seconds=elapsed,
                            same_input=(row["prompt_sha256"],row["snapshot_id"]) == (reference["prompt_sha256"],reference["snapshot_id"]),
                            strict_token_match=row["token_ids"] == reference["token_ids"][:limit] and
                                (stage != "natural" or (row["text"],row["finish_reason"]) == (reference["text"],reference["finish_reason"])))
                        signature = (row["token_ids"],row["rounds"],row["asd"],row["finish_reason"])
                        key = stage, arm
                        if key in seen and seen[key] != signature:
                            raise ValueError("Repeated trajectory changed")
                        seen[key] = signature
                        if not row["same_input"] or (arm.endswith(("/DFlash","/tree32")) and not row["strict_token_match"]):
                            raise ValueError("Strict output or input mismatch")
                        if row["asd"] and not 0 <= row["asd"]["spent"] <= 8:
                            raise ValueError("Request budget exceeded")
                        if stage == "natural":
                            if row["text"] not in quality_cache:
                                quality_cache[row["text"]] = score(record,row["text"])
                            row["quality"] = quality_cache[row["text"]]
                        append(out/"rows.jsonl",row)
                        count += 1
                        peak = max(peak,row["peak_gpu_bytes"])
            info.update(count=count, peak_gpu_bytes=peak, compiler=engine._regional.summary())
            write(out/"provenance.json",info)
            print("completed",args.phase,record["benchmark"],record["id"],count,flush=True)
        select("base")
        if parameter_guard(engine.target) != guard:
            raise ValueError("Frozen Target changed")
        if any(not torch.equal(engine.draft.get_submodule(n).weight.detach().cpu(),w) for n,w in bank.original.items()):
            raise ValueError("Base Draft not restored")
        for v,path in paths.items():
            if digest(path.read_bytes()) != ADAPTERS[v][1]:
                raise ValueError("Adapter changed during evaluation")
        cache_bytes = sum(p.stat().st_size for key in ("TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR") for p in Path(os.environ[key]).rglob("*") if p.is_file())
        if not 0 < peak <= 46*2**30 or cache_bytes > 4*2**30:
            raise ValueError("Comparison resource limit exceeded")
        info.update(status="complete", count=count, compiler=engine._regional.summary(),
                    offline_merge_seconds=merge_seconds, cache_bytes=cache_bytes, base_restored=True)
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
