"""Research-only extrapolation of the b16 checkpoint to real 24/32-token drafts."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import random
from unittest.mock import patch

from memorydraft.evaluation.asd_suite import BENCHMARKS, load_suite, score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import POLICIES, protected
from scripts.run_asd_suite import execute
from scripts.run_runtime_migration import append

ROOT=Path(__file__).resolve().parents[1]
SIZES=(16,24,32)
ARMS=tuple((m,size) for size in SIZES for m in POLICIES)


@contextmanager
def experimental_block(draft,size):
    """Override only the audited decode attribute; never rewrite training config.

    The pinned DFlash forward accepts noise position lengths independently of
    this attribute. Its native generate and our adapter read block_size at call
    entry. This is checkpoint extrapolation, isolated from production config.
    """
    if type(size) is not int or size not in SIZES:
        raise ValueError("Unknown experimental block size")
    if draft.block_size!=16 or draft.config.block_size!=16:
        raise ValueError("Expected the frozen b16 checkpoint")
    with patch.object(draft,"block_size",size):
        yield


def selected_records():
    records=load_suite(ROOT/"data/research/asd-suite")
    return [next(r for r in records if r["benchmark"]==b) for b in BENCHMARKS]


def run(engine,record,method,size,limit):
    with experimental_block(engine.draft,size):
        row=execute(engine,record,method,limit)
    if any(r["proposed"]>size-1 for r in row["rounds"]):
        raise ValueError("Unexpected proposal length")
    row.update(training_block_size=16,experimental_block_size=size)
    return row


def check_native(row,old):
    keys=("token_ids","prompt_sha256","snapshot_id","asd","rounds","finish_reason")
    if any(row[k]!=old[k] for k in keys) or row.get("quality")!=old.get("quality"):
        raise ValueError("Native b16 baseline changed")


def prefix_check(full,prefix,size):
    affected=min((r["generated_before"] for r in prefix["rounds"] if r["proposed"]<size-1),
                 default=len(prefix["token_ids"]))
    required=min(affected,len(prefix["token_ids"]))
    common=next((i for i,(a,b) in enumerate(zip(full["token_ids"],prefix["token_ids"])) if a!=b),
                min(len(full["token_ids"]),len(prefix["token_ids"])))
    return dict(required=required,common=common,agrees=common>=required)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--shard",type=int,choices=(0,1),required=True)
    args=parser.parse_args()
    records=selected_records()[args.shard::2]
    previous={}
    for i in (0,1):
        for row in read_rows(ROOT/f"results/asd-suite-stage1/part-{i}/rows.jsonl"):
            if row["repeat"]==0:
                previous[row["request_id"],row["method"],row["stage"]]=row
    args.output.mkdir(parents=True,exist_ok=False)
    before=protected(ROOT)
    info=dict(status="started",task_ids=[r["id"] for r in records],shard=args.shard,
              sizes=SIZES,methods=list(POLICIES),runner_sha256=file_hash(Path(__file__)),
              original_sha256=before,training_block_size=16,protocol_sha256=file_hash(ROOT/"docs/ASD_BLOCK_PROTOCOL.md"))
    write(args.output/"provenance.json",info)
    try:
        engine=load_engine(replace(RuntimeConfig.load(args.config),db=None,max_new_tokens=4096))
        info.update(runtime=provenance(engine),stop_token_ids=engine.stops)
        write(args.output/"provenance.json",info)
        for size in SIZES:
            for m in POLICIES:
                run(engine,dict(benchmark="gsm8k",id="warmup",question="What is 3 plus 4?",answer="#### 7"),m,size,32)
        count=0
        for record in records:
            seen={}
            for stage,limit,repeats in (("natural",4096,1),("prefix",128,2)):
                for rep in range(repeats):
                    arms=list(ARMS)
                    if rep==0:arms.append(("AR",16))
                    random.Random(f"block-preflight:{record['id']}:{stage}:{rep}").shuffle(arms)
                    for method,size in arms:
                        row=(execute(engine,record,"AR",limit) if method=="AR" else run(engine,record,method,size,limit))
                        row.update(stage=stage,repeat=rep,training_block_size=16,experimental_block_size=size)
                        if stage=="natural":row["quality"]=score(record,row["text"])
                        if stage=="prefix":row["prefix_agreement"]=prefix_check(seen["natural",method,size],row,size)
                        if size==16:check_native(row,previous[record["id"],method,stage])
                        key=stage,method,size
                        if key in seen and (row["token_ids"],row["asd"],row["rounds"])!=(seen[key]["token_ids"],seen[key]["asd"],seen[key]["rounds"]):
                            raise ValueError("Block probe repeat changed")
                        seen[key]=row
                        append(args.output/"rows.jsonl",row);count+=1
            print("completed",record["id"],count,flush=True)
            info.update(rows=count)
            write(args.output/"provenance.json",info)
        info.update(status="complete")
    except Exception as error:
        info.update(status="failed",error=type(error).__name__+": "+str(error));raise
    finally:
        info.update(original_unchanged=before==protected(ROOT))
        write(args.output/"provenance.json",info)
        if not info["original_unchanged"]:raise ValueError("Previous evidence changed")


if __name__=="__main__":
    main()
