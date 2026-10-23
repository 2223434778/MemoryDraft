"""Matched public-prefix cache check of the confirmed GSM candidate and controls."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import statistics
from unittest.mock import patch

from memorydraft.evaluation.adaptation import AdapterBank, digest
from memorydraft.evaluation.asd_comparison import ADAPTERS, BASELINES
from memorydraft.evaluation.asd_suite import score
from memorydraft.evaluation.long_context import load_data, request_for
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts import run_asd_comparison as comparison
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import protected
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]


def candidate(analysis):
    family = analysis["confirm"]["families"]["gsm8k"]
    choices = [a for a in family if a not in BASELINES]
    if len(choices) != 2:
        raise ValueError("Expected two confirmed GSM candidates")
    def key(arm):
        value = family[arm]["natural"]["base/ASD"]
        return value["lost"],value["new_truncations"],value["ratio"],arm
    return min(choices,key=key)


def analyze(rows,arms):
    identities = {r["request_id"] for r in rows}
    if not identities:
        raise ValueError("Empty engineering matrix")
    expected = {(q,a,s,i) for q in identities for a in arms for s in ("off","cold","hit") for i in range(2)}
    actual = [(r["request_id"],r["arm"],r["cache_state"],r["repeat"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete engineering matrix")
    samples, deltas = [],[]
    for identity in sorted(identities):
        case = {}
        for arm in arms:
            group = {s:[r for r in rows if (r["request_id"],r["arm"],r["cache_state"]) == (identity,arm,s)] for s in ("off","cold","hit")}
            reference = group["off"][0]
            stable = all((r["token_ids"],r["text"],r["finish_reason"]) ==
                (reference["token_ids"],reference["text"],reference["finish_reason"]) for rs in group.values() for r in rs)
            for state,rs in group.items():
                for r in rs:
                    if (not math.isfinite(r["latency_seconds"]) or r["latency_seconds"] <= 0
                            or not 0 <= r["ttft_seconds"] <= r["latency_seconds"]):
                        raise ValueError("Invalid engineering timing")
                    if not r["same_input"] or not r["strict_token_match"]:
                        raise ValueError("Engineering strict output/input mismatch")
                    if r["asd"] and not 0 <= r["asd"]["spent"] <= 8:
                        raise ValueError("Engineering budget exceeded")
                    if state != "off" and r["counters"]["prefix_cache_hit"] != int(state == "hit"):
                        raise ValueError("Unexpected cache state")
                    if r["asd"]:
                        deltas.append(abs(r["asd"]["spent"]-reference["asd"]["spent"]))
            costs = {s:statistics.mean(r["latency_seconds"] for r in rs) for s,rs in group.items()}
            case[arm] = dict(seconds=costs,ttft={s:statistics.mean(r["ttft_seconds"] for r in rs) for s,rs in group.items()},
                tokens={s:len(rs[0]["token_ids"]) for s,rs in group.items()},outputs_equal=stable,
                quality={s:rs[0]["quality"] for s,rs in group.items()})
        samples.append(dict(request_id=identity,arms=case))
    summary = {}
    for arm in arms:
        values = [s["arms"][arm] for s in samples]
        summary[arm] = dict(requests=len(values),all_outputs_equal=all(v["outputs_equal"] for v in values),
            hit_off_ratio=statistics.mean(v["seconds"]["hit"]/v["seconds"]["off"] for v in values),
            cold_off_ratio=statistics.mean(v["seconds"]["cold"]/v["seconds"]["off"] for v in values),
            ttft={state:statistics.mean(v["ttft"][state] for v in values) for state in ("off","cold","hit")},
            correct={state:sum(v["quality"][state]["correct"] for v in values) for state in ("off","cold","hit")},
            vs_asd={state:statistics.mean(s["arms"][arm]["seconds"][state]/s["arms"]["base/ASD"]["seconds"][state] for s in samples) for state in ("off","cold","hit")},
            vs_dflash={state:statistics.mean(s["arms"][arm]["seconds"][state]/s["arms"]["base/DFlash"]["seconds"][state] for s in samples) for state in ("off","cold","hit")})
    return dict(rows=len(rows),requests=len(samples),summary=summary,samples=samples,max_spent_abs_delta=max(deltas,default=0.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard",type=int,choices=(0,1),required=True)
    parser.add_argument("--comparison",type=Path,required=True)
    args = parser.parse_args()
    raw = (args.comparison/"analysis.json").read_bytes()
    arm = candidate(json.loads(raw))
    arms = (*BASELINES,arm)
    out = ROOT/f"results/raw/asd-comparison-engineering/eval-{args.shard}"
    out.mkdir(parents=True,exist_ok=False)
    info = dict(status="started",arms=arms,comparison_analysis_sha256=digest(raw),shard=args.shard)
    before,engine = protected(ROOT),None
    try:
        from safetensors.torch import load_file
        states = {}
        for variant in {a.split("/")[0] for a in arms}-{ "base" }:
            path = ROOT.parent/ADAPTERS[variant][0]
            if digest(path.read_bytes()) != ADAPTERS[variant][1]:raise ValueError("Engineering adapter changed")
            states[variant] = load_file(str(path))
        engine = load_engine(replace(RuntimeConfig.load(ROOT/"configs/runtime-dflash-asd-shared-prefix.json"),asd={}))
        obj,bank = engine._prefix_cache,AdapterBank(engine.draft)
        tasks,items,_ = load_data(ROOT)
        chosen = {r["id"] for i,r in enumerate(tasks) if i%2 == args.shard}
        tasks = {r["id"]:r for r in tasks}
        items = [dict(r,benchmark="gsm8k") for r in items if r["task_id"] in chosen and r["level"] == 4096]
        if len(items) != 4:raise ValueError("Unexpected engineering task coverage")
        def generate(item,arm,limit,state):
            context = patch.object(engine,"_prefix_cache",None) if state == "off" else nullcontext()
            with context,patch.object(comparison,"request_for",request_for):
                row = comparison.execute(engine,item,arm,limit)
            return row
        for a in arms:
            bank.select(states.get(a.split("/")[0]))
            for state in ("off","cold","hit"):
                if state == "cold":obj.clear()
                append(out/"warmup.jsonl",generate(items[0],a,16,state))
        info["compiler_after_warmup"] = engine._regional.summary()
        count = 0
        for item in items:
            bank.select()
            reference = asdict(engine.generate(request_for(item,4096),"AR"))
            append(out/"references.jsonl",reference)
            cache = {}
            for repeat in range(2):
                jobs = list(arms)
                random.Random(f"comparison-engineering:{item['task_id']}:{repeat}").shuffle(jobs)
                for a in jobs:
                    bank.select(states.get(a.split("/")[0]))
                    obj.clear()
                    order = ("off","cold","hit") if repeat == 0 else ("cold","hit","off")
                    for state in order:
                        row = generate(item,a,4096,state)
                        row.update(cache_state=state,repeat=repeat,task_id=item["task_id"],
                            same_input=row["prompt_sha256"] == item["prompt_sha256"] and row["snapshot_id"] == reference["snapshot_id"],
                            strict_token_match=not a.endswith(("/DFlash","/tree32")) or
                                (row["token_ids"],row["text"],row["finish_reason"]) == (reference["token_ids"],reference["text"],reference["finish_reason"]))
                        if row["text"] not in cache:cache[row["text"]] = score(tasks[item["task_id"]],row["text"])
                        row["quality"] = cache[row["text"]]
                        append(out/"rows.jsonl",row)
                        count += 1
            info.update(count=count,compiler=engine._regional.summary())
            write(out/"provenance.json",info)
            print("completed engineering",item["task_id"],count,flush=True)
        bank.select()
        rows = read_rows(out/"rows.jsonl")
        result = analyze(rows,arms)
        if max(r["peak_gpu_bytes"] for r in rows) > 46*2**30:raise ValueError("Engineering memory limit")
        if any(r["counters"]["compile_graphs_after"] != r["counters"]["compile_graphs_before"] for r in rows):
            raise ValueError("Engineering formal compile")
        write(out/"analysis.json",result)
        info.update(status="complete",count=count,compiler=engine._regional.summary())
    except Exception as error:
        info.update(status="failed",error=type(error).__name__+": "+str(error))
        raise
    finally:
        info["original_unchanged"] = before == protected(ROOT)
        write(out/"provenance.json",info)
        if engine is not None:engine.close()


if __name__ == "__main__":
    main()
