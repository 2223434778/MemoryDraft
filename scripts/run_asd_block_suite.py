"""Full development comparison of real b16/b24/b32 drafts with unchanged policies."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
from unittest.mock import patch

from memorydraft.backends.asd import ASDVerifier
from memorydraft.evaluation.asd_suite import BENCHMARKS, load_suite, score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_block_probe import ARMS, check_native, experimental_block, prefix_check, run
from scripts.run_asd_budget_probe import protected
from scripts.run_asd_suite import execute
from scripts.run_runtime_migration import append

ROOT=Path(__file__).resolve().parents[1]
ALL_ARMS=(*ARMS,("AR",16))
REPEATS={"natural":3,"prefix":2}
LIMITS={"natural":4096,"prefix":128}


class RecordedVerifier(ASDVerifier):
    """Add candidate identities only during the separately timed mechanism run."""

    def decide(self,tokens,logits,*,stop_token_ids=()):
        decision=super().decide(tokens,logits,stop_token_ids=stop_token_ids)
        if self.record:
            self.events[-1].update(proposal_token_ids=tokens.tolist(),target_top_token_ids=logits.argmax(-1).tolist())
        return decision


def records_and_positions():
    records=load_suite(ROOT/"data/research/asd-suite","development")
    positions={r["id"]:i for b in BENCHMARKS for i,r in enumerate(r for r in records if r["benchmark"]==b)}
    return records,positions


def previous_rows():
    rows={}
    for i in (0,1):
        for row in read_rows(ROOT/f"results/asd-suite-stage1/part-{i}/rows.jsonl"):
            if row["repeat"]==0:rows[row["request_id"],row["method"],row["stage"]]=row
    return rows


def record_mechanism(engine,record,method,size):
    from scripts import run_asd_suite as source
    with experimental_block(engine.draft,size),patch.object(source,"ASDVerifier",RecordedVerifier):
        row=execute(engine,record,method,4096,mechanism=True)
    row.update(training_block_size=16,experimental_block_size=size)
    return row


def run_question(engine,record,position,previous,output):
    seen={};counts=dict(main=0,mechanism=0)
    for stage,repeats in REPEATS.items():
        for rep in range(repeats):
            arms=list(ALL_ARMS)
            random.Random(f"block-suite:{record['id']}:{stage}:{rep}").shuffle(arms)
            for method,size in arms:
                row=(execute(engine,record,method,LIMITS[stage]) if method=="AR" else run(engine,record,method,size,LIMITS[stage]))
                key=stage,method,size
                if key in seen:
                    if (row["token_ids"],row["asd"],row["rounds"])!=(seen[key]["token_ids"],seen[key]["asd"],seen[key]["rounds"]):
                        raise ValueError("Block suite repeat changed trajectory or ledger")
                if stage=="natural":row["quality"]=seen[key]["quality"] if key in seen else score(record,row["text"])
                if stage=="prefix":row["prefix_agreement"]=prefix_check(seen["natural",method,size],row,size)
                row.update(stage=stage,repeat=rep,training_block_size=16,experimental_block_size=size)
                if size==16:check_native(row,previous[record["id"],method,stage])
                seen[key]=row
                append(output/"rows.jsonl",row);counts["main"]+=1
    if position<2:
        for method,size in ARMS:
            row=record_mechanism(engine,record,method,size)
            baseline=seen["natural",method,size]
            if (row["token_ids"],row["asd"])!=(baseline["token_ids"],baseline["asd"]):
                raise ValueError("Mechanism changed block suite trajectory or ledger")
            row["quality"]=baseline["quality"]
            append(output/"mechanism.jsonl",row);counts["mechanism"]+=1
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--shard",type=int,choices=(0,1),required=True)
    args=parser.parse_args()
    records,positions=records_and_positions()
    records=[r for r in records if positions[r["id"]]%2==args.shard]
    previous=previous_rows();before=protected(ROOT)
    args.output.mkdir(parents=True,exist_ok=False)
    info=dict(status="started",task_ids=[r["id"] for r in records],split="development",shard=args.shard,
              training_block_size=16,arms=ALL_ARMS,repeats=REPEATS,limits=LIMITS,
              runner_sha256=file_hash(Path(__file__)),protocol_sha256=file_hash(ROOT/"docs/ASD_BLOCK_SUITE_PROTOCOL.md"),
              data_manifest_sha256=file_hash(ROOT/"data/research/asd-suite/manifest.json"),original_sha256=before)
    write(args.output/"provenance.json",info)
    try:
        engine=load_engine(replace(RuntimeConfig.load(args.config),db=None,max_new_tokens=4096))
        info.update(runtime=provenance(engine),stop_token_ids=engine.stops)
        write(args.output/"provenance.json",info)
        warmup=dict(benchmark="gsm8k",id="warmup",question="What is 3 plus 4?",answer="#### 7")
        for method,size in ALL_ARMS:
            if method=="AR":execute(engine,warmup,method,32)
            else:run(engine,warmup,method,size,32)
        counts=dict(main=0,mechanism=0)
        for record in records:
            added=run_question(engine,record,positions[record["id"]],previous,args.output)
            counts={k:counts[k]+added[k] for k in counts}
            info.update(counts=counts)
            write(args.output/"provenance.json",info)
            print("completed",record["id"],json.dumps(counts),flush=True)
        info.update(status="complete")
    except Exception as error:
        info.update(status="failed",error=type(error).__name__+": "+str(error));raise
    finally:
        info["original_unchanged"]=before==protected(ROOT)
        write(args.output/"provenance.json",info)
        if not info["original_unchanged"]:raise ValueError("Previous evidence changed")


if __name__=="__main__":main()
