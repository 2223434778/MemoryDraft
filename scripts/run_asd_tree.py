"""Six-task cost-controlled trees with fixed Target and ASD B8/g.25/m2."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random
import statistics
import time
from unittest.mock import patch

import torch
from transformers import DynamicCache

from memorydraft.backends.asd import ASDConfig
from memorydraft.backends.asd_cpu import CPUASDVerifier
from memorydraft.backends.budget_tree import BudgetTreeConfig, TreeASDVerifier, TreeCosts, build_budget_tree
from memorydraft.backends.draft_tree import DraftTree
from memorydraft.evaluation.asd_suite import BENCHMARKS, canonical, request_for, score
from memorydraft.evaluation.asd_tree import ARMS, REPEATS, STUDY, aggregate, records
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import protected
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]


def execute(engine, record, arm, limit, tree_config, *, zero=False):
    if arm not in ARMS:
        raise ValueError("Unknown tree comparison arm")
    from memorydraft.runtime import engine as module
    native, detail = module.generate_reuse, None
    engine.synchronize()
    started = time.perf_counter()
    verifier = (CPUASDVerifier(ASDConfig(budget=8)) if arm == "ASD" else
                TreeASDVerifier(ASDConfig(budget=0 if zero else 8)) if arm == "BudgetTree-ASD" else None)
    def hook(*args, **kwargs):
        nonlocal detail
        kwargs["incremental_stop"] = True
        if arm.startswith("BudgetTree"):
            kwargs["tree_config"] = tree_config
        if verifier is not None:
            kwargs["verifier"] = verifier
        detail = native(*args, **kwargs)
        return detail
    with patch.object(module, "generate_reuse", hook):
        result = engine.generate(request_for(record, limit), "DFlash")
    engine.synchronize()
    row = asdict(result)
    row.update(engine_latency_seconds=row["latency_seconds"], latency_seconds=time.perf_counter()-started)
    row.update(engine_method=row["method"],method=arm,approximate=verifier is not None and verifier.config.budget > 0)
    return dict(row, benchmark=record["benchmark"], arm=arm, rounds=detail.rounds,
                asd=verifier.summary() if verifier else None, max_new_tokens=limit,
                experimental_approximate=verifier is not None and verifier.config.budget > 0)


@torch.inference_mode()
def calibrate(engine, warm, out):
    """Offline synthetic-prefix replay; no benchmark answers or quality tuning."""
    from memorydraft.runtime import engine as module
    native = module.generate_reuse
    captured, draft_times = [], []
    forward = engine.draft.forward
    def measured(*args, **kwargs):
        engine.synchronize()
        started = time.perf_counter()
        result = forward(*args, **kwargs)
        engine.synchronize()
        draft_times.append(time.perf_counter()-started)
        return result
    def hook(*args, **kwargs):
        kwargs["on_draft"] = lambda confirmed,tokens,logits,budget:captured.append(
            (int(confirmed[-1]), list(tokens), logits.detach().clone())) if not captured else None
        return native(*args, **kwargs)
    with patch.object(module, "generate_reuse", hook), patch.object(engine.draft, "forward", measured):
        row = asdict(engine.generate(request_for(warm, 128), "DFlash"))
        append(out/"calibration-generations.jsonl",row)
    contexts, nodes = (128, 1024), (8, 12, 16, 20, 24, 32)
    all_rows, seconds = [], []
    text_ids = engine.tokenizer.encode("Calculate carefully. Use the supplied facts. "*256, add_special_tokens=False)
    for context in contexts:
        ids = torch.tensor([text_ids[:context]],device=engine.target.device)
        if ids.shape[1] != context:
            raise ValueError("Synthetic calibration prefix too short")
        cache = DynamicCache()
        initial = engine.target(ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
        anchor = int(initial.logits[0,-1].argmax())
        times = []
        for count in nodes:
            parents = (-1,)+tuple((i-1)//2 for i in range(1,count))
            depths = [0]
            for parent in parents[1:]:depths.append(depths[parent]+1)
            tree = DraftTree(ids.new_full((1,count),anchor),parents,tuple(depths),1)
            positions = ids.new_tensor([depths])+context
            measured_times = []
            for repeat in range(4):
                cache.crop(context)
                engine.synchronize()
                started = time.perf_counter()
                mask = tree.mask(context,engine.target.dtype)
                result = engine.target(tree.tokens,position_ids=positions,past_key_values=cache,
                    attention_mask=mask,use_cache=True,output_hidden_states=True)
                result.logits.argmax(-1).tolist()
                engine.synchronize()
                elapsed = time.perf_counter()-started
                all_rows.append(dict(context=context,nodes=count,repeat=repeat,seconds=elapsed,warmup=repeat==0))
                if repeat:measured_times.append(elapsed)
                del result
            times.append(statistics.median(measured_times))
        seconds.append(tuple(times))
        del initial, cache
    costs = TreeCosts(contexts,nodes,tuple(seconds),statistics.median(draft_times))
    anchor,tokens,logits = captured[0]
    block = torch.tensor([[anchor,*tokens]],device=logits.device)
    auxiliary = []
    for _ in range(4):
        engine.synchronize()
        started = time.perf_counter()
        build_budget_tree(block,logits.unsqueeze(0),BudgetTreeConfig(costs),128,stop_token_ids=engine.stops)
        engine.synchronize()
        auxiliary.append(time.perf_counter()-started)
    costs = replace(costs,auxiliary_seconds=statistics.median(auxiliary[1:]))
    write(out/"calibration.json",dict(costs=asdict(costs),replays=all_rows,draft_seconds=draft_times,
        auxiliary_seconds=auxiliary,method="synthetic128/1024; median3; measured table; no roofline fit"))
    return BudgetTreeConfig(costs)


def main(*, study=STUDY, arms=ARMS, tree_factory=calibrate, execution=execute, legacy_check=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard",type=int,choices=(0,1),required=True)
    args = parser.parse_args()
    out = ROOT/f"results/raw/{study}/part-{args.shard}"
    out.mkdir(parents=True,exist_ok=False)
    before,engine = protected(ROOT),None
    pool = records(ROOT)
    selected = [r for half in (0,1) for b in BENCHMARKS
                for i,r in enumerate(r for r in pool if r["benchmark"]==b)
                if i%2 == args.shard and int(i>=4)==half]
    info = dict(status="started",task_ids=[r["id"] for r in selected],shard=args.shard,repeats=REPEATS,
        arms=arms,split="previously_observed_development",limit=4096)
    write(out/"provenance.json",info)
    try:
        started = time.perf_counter()
        engine = load_engine(replace(RuntimeConfig.load(ROOT/"configs/runtime-dflash-rmsnorm.json"),
            db=None,max_new_tokens=4096))
        info["load_seconds"] = time.perf_counter()-started
        if engine.draft.block_size != 16:
            raise ValueError("Expected native16 DFlash")
        warm = dict(benchmark="gsm8k",id="warmup",question="What is 3 plus 4?",answer="#### 7")
        append(out/"warmup.jsonl",asdict(engine.generate(request_for(warm,32),"AR")))
        append(out/"warmup.jsonl",asdict(engine.generate(request_for(warm,32),"DFlash")))
        started = time.perf_counter()
        tree_config = tree_factory(engine,warm,out)
        info.update(calibration_seconds=time.perf_counter()-started,tree_config=asdict(tree_config))
        for arm in arms:
            append(out/"warmup.jsonl",execution(engine,warm,arm,64,tree_config))
        zero = execution(engine,warm,"BudgetTree-ASD",64,tree_config,zero=True)
        strict = execution(engine,warm,"BudgetTree",64,tree_config)
        append(out/"warmup.jsonl",zero)
        append(out/"warmup.jsonl",strict)
        if zero["token_ids"] != strict["token_ids"] or zero["asd"]["spent"] != 0:
            raise ValueError("GPU zero-budget tree mismatch")
        info.update(runtime=provenance(engine),compiler_after_warmup=engine._regional.summary())
        preflight = [{"request_id":r["id"],"quality":score(r,canonical(r))} for r in selected]
        write(out/"canonical-preflight.json",preflight)
        if not all(r["quality"]["correct"] for r in preflight):
            raise ValueError("Reference scoring preflight failed")
        count, peak = 0,0
        for index,record in enumerate(selected):
            if index == 12:
                write(out/"pilot-ready.json",dict(count=count,requests=index))
                def barrier(name):
                    started = time.monotonic()
                    while not all((out.parent/f"part-{i}/{name}.json").exists() for i in (0,1)):
                        for i in (0,1):
                            path = out.parent/f"part-{i}/provenance.json"
                            if path.exists():
                                try:
                                    status = json.loads(path.read_text())["status"]
                                except json.JSONDecodeError:
                                    continue  # The other worker may be replacing its progress JSON.
                                if status == "failed":
                                    raise ValueError("Other tree worker failed at pilot barrier")
                        if time.monotonic()-started > 3600:
                            raise TimeoutError("Tree pilot barrier timed out")
                        time.sleep(.5)
                barrier("pilot-ready")
                def partial(name):
                    return [json.loads(line) for i in (0,1)
                        for line in (out.parent/f"part-{i}/{name}.jsonl").read_text().splitlines()]
                pilot = aggregate(partial("rows"),partial("references"),arms=arms)
                pilot_count = 24*len(arms)*sum(REPEATS.values())
                if (pilot["timed_generations"] != pilot_count or pilot["ar_generations"] != 24
                        or any(v["ASD"]["natural"]["requests"] != 4 for v in pilot["families"].values())):
                    raise ValueError("Pilot24 coverage incomplete")
                write(out/"pilot-check.json",dict(status="passed",timed=pilot_count,ar=24,checks="coverage/AR/input/repeats/budget"))
                barrier("pilot-check")
                info["completed_phase"] = "pilot24"
            reference = asdict(engine.generate(request_for(record,4096),"AR"))
            reference.update(benchmark=record["benchmark"],quality=score(record,reference["text"]))
            append(out/"references.jsonl",reference)
            quality_cache, seen = {reference["text"]:reference["quality"]},{}
            for stage,repeats in REPEATS.items():
                limit = 4096 if stage == "natural" else 128
                for repeat in range(repeats):
                    jobs = list(arms)
                    random.Random(f"budget-tree:{record['id']}:{stage}:{repeat}").shuffle(jobs)
                    for arm in jobs:
                        row = execution(engine,record,arm,limit,tree_config)
                        row.update(stage=stage,repeat=repeat)
                        if (row["prompt_sha256"],row["snapshot_id"]) != (reference["prompt_sha256"],reference["snapshot_id"]):
                            raise ValueError("Target input changed")
                        if arm in ("DFlash","BudgetTree") and (row["token_ids"] != reference["token_ids"][:limit]
                                or (stage == "natural" and (row["text"],row["finish_reason"]) != (reference["text"],reference["finish_reason"]))):
                            raise ValueError("Strict tree mismatch: "+record["id"])
                        signature = (row["token_ids"],row["rounds"],row["asd"],row["finish_reason"])
                        key = stage,arm
                        if key in seen and seen[key] != signature:
                            raise ValueError("Repeated tree trajectory changed")
                        seen[key] = signature
                        if row["asd"] and not 0 <= row["asd"]["spent"] <= 8:
                            raise ValueError("ASD tree budget exceeded")
                        if stage == "natural":
                            if row["text"] not in quality_cache:quality_cache[row["text"]]=score(record,row["text"])
                            row["quality"] = quality_cache[row["text"]]
                        if legacy_check is not None:
                            legacy_check(row)
                        append(out/"rows.jsonl",row)
                        count += 1
                        peak = max(peak,row["peak_gpu_bytes"])
            info.update(count=count,peak_gpu_bytes=peak,compiler=engine._regional.summary())
            write(out/"provenance.json",info)
            print("completed",record["benchmark"],record["id"],count,flush=True)
        cache_bytes = sum(p.stat().st_size for name in ("TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR")
                          for p in Path(os.environ[name]).rglob("*") if p.is_file())
        if peak > 46*2**30 or cache_bytes > 4*2**30:
            raise ValueError("Resource bounds exceeded")
        info.update(status="complete",count=count,cache_bytes=cache_bytes,compiler=engine._regional.summary(),
                    legacy_checked_rows=getattr(legacy_check,"checked",None))
    except Exception as error:
        info.update(status="failed",error=type(error).__name__+": "+str(error))
        raise
    finally:
        info["original_unchanged"] = before == protected(ROOT)
        write(out/"provenance.json",info)
        if engine is not None:engine.close()
        if not info["original_unchanged"]:raise ValueError("Previous evidence changed")


if __name__ == "__main__":main()
