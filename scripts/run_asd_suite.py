"""First-stage cross-task baselines. Validation/reserve cannot be selected here."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
from unittest.mock import patch

from memorydraft.backends.asd import ASDVerifier
from memorydraft.evaluation.asd_suite import BENCHMARKS, canonical, load_suite, request_for, score
from memorydraft.evaluation.runtime_migration import file_hash
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import POLICIES, protected
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/research/asd-suite"
METHODS = ("AR", *POLICIES)


def prefix_agreement(full, prefix):
    # Only the final length-cropped block can change remaining-length ASD gates.
    affected = min((r["generated_before"] for r in prefix["rounds"] if r["proposed"] < 15),default=len(prefix["token_ids"]))
    required = min(affected,len(prefix["token_ids"]))
    common = next((i for i,(a,b) in enumerate(zip(full["token_ids"],prefix["token_ids"])) if a!=b),
                  min(len(full["token_ids"]),len(prefix["token_ids"])))
    return dict(required=required,common=common,agrees=common>=required)


def execute(engine, record, method, limit, *, mechanism=False, profile=False):
    if method=="AR":
        result=engine.generate(request_for(record,limit),"AR",profile=profile)
        return {**asdict(result),"benchmark":record["benchmark"],"max_new_tokens":limit,"asd":None,"rounds":[]}
    policy = ASDVerifier(POLICIES[method],record=mechanism)
    detail = None
    from memorydraft.runtime import engine as module
    native = module.generate_reuse

    def hook(*args,**kwargs):
        nonlocal detail
        if method!="DFlash" or mechanism:
            kwargs["verifier"] = policy
        detail = native(*args,**kwargs)
        return detail

    with patch.object(module,"generate_reuse",hook):
        result = engine.generate(request_for(record,limit),"DFlash",profile=profile)
    if method=="DFlash" and not mechanism:
        policy.blocks=len(detail.rounds)
    row={**asdict(result),"method":method,"benchmark":record["benchmark"],"max_new_tokens":limit,
         "asd":policy.summary(),"rounds":detail.rounds}
    if mechanism:
        row["events"]=policy.events
        row["instrumented_latency_seconds"]=row.pop("latency_seconds")
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--benchmarks",nargs="+",choices=BENCHMARKS,required=True)
    parser.add_argument("--shard",type=int,choices=(0,1),required=True)
    args=parser.parse_args()
    if len(set(args.benchmarks))!=len(args.benchmarks):
        parser.error("Duplicate benchmarks")
    all_records=[r for r in load_suite(DATA,"development") if r["benchmark"] in args.benchmarks]
    positions={r["id"]:i for b in args.benchmarks for i,r in enumerate([r for r in all_records if r["benchmark"]==b])}
    records=[r for r in all_records if positions[r["id"]]%2==args.shard]
    before=protected(ROOT)
    args.output.mkdir(parents=True,exist_ok=False)
    info=dict(status="started",benchmarks=args.benchmarks,task_ids=[r["id"] for r in records],split="development",shard=args.shard,
              runner_sha256=file_hash(Path(__file__)),data_manifest_sha256=file_hash(DATA/"manifest.json"),
              protocol_sha256=file_hash(ROOT/"docs/ASD_SUITE_PROTOCOL.md"),original_sha256=before,
              natural_repeats=3,prefix_repeats=2,natural_limit=4096,prefix_limit=128)
    write(args.output/"provenance.json",info)
    try:
        engine=load_engine(replace(RuntimeConfig.load(args.config),db=None,max_new_tokens=4096))
        if engine.draft.block_size!=16:
            raise ValueError("Frozen baseline requires the audited native block size 16")
        info.update(runtime=provenance(engine),stop_token_ids=engine.stops)
        write(args.output/"provenance.json",info)
        # Test expected scoring before any model outcome; keep all tasks and fail
        # the harness rather than silently excluding a task on an evaluator error.
        preflight=[]
        for record in records:
            quality=score(record,canonical(record))
            preflight.append(dict(request_id=record["id"],quality=quality))
        write(args.output/"canonical-preflight.json",preflight)
        if not all(r["quality"]["correct"] for r in preflight):
            raise ValueError("Reference solution failed grading")
        for method in METHODS:
            # Distinct fresh KV per request, including warmups.
            execute(engine,dict(benchmark="gsm8k",id="warmup",question="What is 3 plus 4?",answer="#### 7"),method,32)
        by_benchmark={b:[r for r in records if r["benchmark"]==b] for b in args.benchmarks}
        counts=dict(main=0,mechanism=0)
        for b,selected in by_benchmark.items():
            for index,record in enumerate(selected):
                seen={}
                for repeat in range(3):
                    order=list(METHODS); random.Random(f"suite:{record['id']}:natural:{repeat}").shuffle(order)
                    for method in order:
                        row=execute(engine,record,method,4096)
                        key="natural",method
                        if key in seen:
                            if (row["token_ids"],row["asd"],row["rounds"])!=(seen[key]["token_ids"],seen[key]["asd"],seen[key]["rounds"]):
                                raise ValueError("Natural repeat changed trajectory/ledger")
                            row["quality"]=seen[key]["quality"]
                        else:
                            row["quality"]=score(record,row["text"])
                            seen[key]=row
                        row.update(stage="natural",repeat=repeat)
                        append(args.output/"rows.jsonl",row); counts["main"]+=1
                for repeat in range(2):
                    order=list(METHODS); random.Random(f"suite:{record['id']}:prefix:{repeat}").shuffle(order)
                    for method in order:
                        row=execute(engine,record,method,128)
                        key="prefix",method
                        if key in seen and (row["token_ids"],row["asd"],row["rounds"])!=(seen[key]["token_ids"],seen[key]["asd"],seen[key]["rounds"]):
                            raise ValueError("Prefix repeat changed trajectory/ledger")
                        seen[key]=row
                        row.update(stage="prefix",repeat=repeat,prefix_agreement=prefix_agreement(seen["natural",method],row))
                        append(args.output/"rows.jsonl",row); counts["main"]+=1
                if positions[record["id"]]<2:
                    for method in POLICIES:
                        row=execute(engine,record,method,4096,mechanism=True)
                        reference=seen["natural",method]
                        if (row["token_ids"],row["asd"])!=(reference["token_ids"],reference["asd"]):
                            raise ValueError("Mechanism changed timed trajectory")
                        row["quality"]=reference["quality"]
                        append(args.output/"mechanism.jsonl",row); counts["mechanism"]+=1
                print("completed",b,record["id"],json.dumps(counts),flush=True)
                info.update(counts=counts)
                write(args.output/"provenance.json",info)
        info.update(status="complete",counts=counts)
    except Exception as error:
        info.update(status="failed",error=type(error).__name__+": "+str(error)); raise
    finally:
        info["original_unchanged"]=before==protected(ROOT)
        write(args.output/"provenance.json",info)
        if not info["original_unchanged"]:
            raise ValueError("Previous evidence changed")
    print("SUITE COMPLETE",flush=True)


if __name__=="__main__":
    main()
