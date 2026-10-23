"""Resource admission and timed requests on the unchanged explicit runtime."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import torch

from memorydraft.evaluation.asd_suite import score
from memorydraft.evaluation.long_context import LEVELS, NATURAL_REPEATS, STUDY, check_input, load_data, request_for
from memorydraft.runtime.benchmark import provenance, write_json
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.run_asd_execution_probe import profile_summary
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]


def diagnostic(engine,item,method,reference):
    original,first = engine.draft.forward,True
    def forward(*args,**kwargs):
        nonlocal first
        name = "draft_first_forward" if first else "draft_later_forward"
        first = False
        with torch.profiler.record_function("memorydraft::"+name):return original(*args,**kwargs)
    with patch.object(engine.draft,"forward",forward), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
        row = asdict(engine.generate(request_for(item,128),method,profile=True))
    if (row["token_ids"],row["counters"]["rounds"],row["prompt_sha256"]) != (
            reference["token_ids"],reference["counters"]["rounds"],reference["prompt_sha256"]):
        raise ValueError("Profiler changed output/decisions")
    return dict(instrumented=True,not_a_timing_sample=True,request_id=row["request_id"],
                method=method,level=item["level"],**profile_summary(profiler.events()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=("base","position"),required=True)
    parser.add_argument("--shard",type=int,choices=(0,1),required=True)
    parser.add_argument("--mode",choices=("preflight","eval"),required=True)
    parser.add_argument("--levels",type=int,nargs="+",default=list(LEVELS))
    args = parser.parse_args()
    if not args.levels or not set(args.levels) <= set(LEVELS):raise ValueError("Unknown context lengths")
    output = ROOT/f"results/raw/{STUDY}/{args.mode}-{args.variant}-{args.shard}"
    output.mkdir(parents=True,exist_ok=False)
    tasks,items,data_info = load_data(ROOT)
    chosen = [r["id"] for i,r in enumerate(tasks) if i%2 == args.shard]
    tasks = {r["id"]:r for r in tasks}
    items = [r for r in items if r["task_id"] in chosen and r["level"] in args.levels]
    config_path = ROOT/("configs/runtime-dflash-asd-position.json" if args.variant == "position" else "configs/runtime-dflash-asd.json")
    config = RuntimeConfig.load(config_path)
    methods = ("DFlash","ASD") if args.variant == "base" else ("ASD",)
    info = dict(status="started",variant=args.variant,shard=args.shard,mode=args.mode,levels=args.levels,
                tasks=chosen,data_files=data_info["files"],rows=0,admitted=[],peak_gpu_bytes=0)
    engine = None
    diagnostic_jobs = []
    try:
        started = time.perf_counter()
        engine = load_engine(config)
        info.update(runtime=provenance(engine),model_setup_seconds=time.perf_counter()-started,
                    load_peak_gpu_bytes=torch.cuda.max_memory_allocated(engine.target.device))
        # All shapes are exercised outside formal timing; cold costs remain in warmup.jsonl.
        for level in sorted(args.levels):
            item = next(r for r in items if r["level"] == level)
            try:
                for method in ("AR",*methods):
                    row = asdict(engine.generate(request_for(item,16),method))
                    check_input(row,item)
                    append(output/"warmup.jsonl",dict(row,level=level))
                    info["peak_gpu_bytes"] = max(info["peak_gpu_bytes"],row["peak_gpu_bytes"])
                if info["peak_gpu_bytes"] > 46*2**30:raise torch.cuda.OutOfMemoryError("Allocated resource budget exceeded")
                info["admitted"].append(level)
            except torch.cuda.OutOfMemoryError as error:
                if args.mode != "preflight":raise
                info["resource_exception"] = dict(level=level,error=str(error)[:300])
                info["resource_exception"]["peak_gpu_bytes"] = torch.cuda.max_memory_allocated(engine.target.device)
                break
        info["compiler_after_warmup"] = engine._regional.summary()
        if args.mode == "eval":
            for item in items:
                reference = None
                if args.variant == "base":
                    reference = asdict(engine.generate(request_for(item),"AR"))
                    check_input(reference,item)
                    reference.update(task_id=item["task_id"],level=item["level"],quality=score(tasks[item["task_id"]],reference["text"]))
                    append(output/"references.jsonl",reference)
                    info["peak_gpu_bytes"] = max(info["peak_gpu_bytes"],reference["peak_gpu_bytes"])
                jobs = [(stage,repeat,method,limit) for stage,limit,repeats in (("natural",4096,NATURAL_REPEATS),("prefix",128,1))
                        for repeat in range(repeats) for method in methods]
                random.Random("long-context:"+item["request"]["id"]).shuffle(jobs)
                prefix_rows = {}
                for stage,repeat,method,limit in jobs:
                    row = asdict(engine.generate(request_for(item,limit),method))
                    check_input(row,item)
                    row.update(task_id=item["task_id"],level=item["level"],variant=args.variant,
                        arm="position-ASD" if args.variant == "position" else method,stage=stage,repeat=repeat,
                        quality=score(tasks[item["task_id"]],row["text"]) if stage == "natural" else None,
                        strict_token_match=(row["token_ids"] == reference["token_ids"][:limit]) if method == "DFlash" else None)
                    append(output/"rows.jsonl",row)
                    info["rows"] += 1
                    info["peak_gpu_bytes"] = max(info["peak_gpu_bytes"],row["peak_gpu_bytes"])
                    if stage == "prefix":prefix_rows[method] = row
                if args.shard == 0 and item["task_id"] == chosen[0]:
                    diagnostic_jobs.extend((item,method,prefix_rows[method]) for method in methods)
                write_json(output/"provenance.json",info)
                print("evaluated",args.variant,args.shard,item["task_id"],item["level"],info["rows"],flush=True)
        for item,method,reference in diagnostic_jobs:
            try:result = diagnostic(engine,item,method,reference)
            except ValueError as error:result = dict(status="failed",error=str(error),instrumented=True,level=item["level"],method=method)
            result.update(variant=args.variant,shard=args.shard)
            append(output/"diagnostics.jsonl",result)
        if info["load_peak_gpu_bytes"] > 46*2**30 or args.mode == "eval" and info["peak_gpu_bytes"] > 46*2**30:
            raise ValueError("GPU resource budget exceeded")
        import os
        cache_bytes = sum(p.stat().st_size for name in ("TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR")
                          for p in Path(os.environ[name]).rglob("*") if p.is_file())
        if cache_bytes > 4*2**30:raise ValueError("Worker compiler cache budget exceeded")
        info.update(status="complete",compiler=engine._regional.summary(),cache_bytes=cache_bytes)
    except Exception as error:
        info.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if engine is not None:engine.close()
        write_json(output/"provenance.json",info)
    print(json.dumps(dict(status=info["status"],variant=args.variant,shard=args.shard,mode=args.mode,admitted=info["admitted"],rows=info["rows"])),flush=True)


if __name__ == "__main__":
    main()
