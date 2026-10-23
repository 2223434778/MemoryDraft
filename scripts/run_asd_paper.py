"""Frozen five-arm comparison: unused math first, then the other paper tasks."""
import argparse
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import random
import time

import torch

from memorydraft.evaluation.asd_joint_tree import frozen_config
from memorydraft.evaluation.asd_paper import (ARMS,BENCHMARKS,REPEATS,STUDY,canonical,
    conversation_record,load_records,request_for,score,stages)
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig,load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import protected
from scripts.run_asd_joint_tree import execute
from scripts.run_runtime_migration import append

ROOT=Path(__file__).resolve().parents[1]


def signature(row):return row['token_ids'],row['rounds'],row['asd'],row['finish_reason'],row['text']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--shard',type=int,choices=(0,1),required=True)
    args=parser.parse_args();out=ROOT/f'results/raw/{STUDY}/part-{args.shard}';out.mkdir(parents=True,exist_ok=False)
    before=protected(ROOT);engine=None
    pool=load_records(ROOT)
    selected=[r for b in BENCHMARKS for i,r in enumerate(r for r in pool if r['benchmark']==b) if i%2==args.shard]
    info=dict(status='started',shard=args.shard,task_ids=[r['id'] for r in selected],arms=ARMS,repeats=REPEATS,
        observed='unused source IDs selected before new generation',max_new_tokens=4096)
    write(out/'provenance.json',info)
    try:
        start=time.perf_counter()
        engine=load_engine(replace(RuntimeConfig.load(ROOT/'configs/runtime-dflash-rmsnorm.json'),db=None,max_new_tokens=4096))
        info['load_seconds']=time.perf_counter()-start
        tree_config,source=frozen_config(ROOT,args.shard);write(out/'cost-table.json',source)
        if engine.draft.block_size!=16:raise ValueError('Native16 required')
        warm=dict(id='warmup',benchmark='gsm8k',question='What is 3 plus 4?',answer='#### 7')
        append(out/'warmup.jsonl',asdict(engine.generate(request_for(warm,32),'AR')))
        for arm in ARMS:append(out/'warmup.jsonl',execute(engine,warm,arm,64,tree_config,request_factory=request_for))
        append(out/'warmup.jsonl',execute(engine,warm,'JointTree-ASD',256,tree_config,request_factory=request_for,fixed=True))
        zero=execute(engine,warm,'JointTree-ASD',64,tree_config,request_factory=request_for,zero=True)
        strict=execute(engine,warm,'BudgetTree',64,tree_config,request_factory=request_for)
        append(out/'warmup.jsonl',zero);append(out/'warmup.jsonl',strict)
        if zero['token_ids']!=strict['token_ids'] or zero['asd']['spent']!=0:raise ValueError('GPU zero-budget mismatch')
        info.update(runtime=provenance(engine),compiler_after_warmup=engine._regional.summary())
        preflight=[]
        for record in selected:
            expected=canonical(record)
            if expected is not None:preflight.append(dict(request_id=record['id'],quality=score(record,expected)))
        write(out/'canonical-preflight.json',preflight)
        # A bad reference test is evidence about the source/scorer, not grounds to replace a task.
        info['canonical_failed_ids']=[r['request_id'] for r in preflight if not r['quality']['correct']]
        count,peak=0,0
        for index,task in enumerate(selected):
            peer=out.parent/f'part-{1-args.shard}/provenance.json'
            if peer.exists():
                try:peer_status=json.loads(peer.read_text())['status']
                except json.JSONDecodeError:peer_status='writing'
                if peer_status=='failed':raise RuntimeError('Other owned paper worker failed')
            first_reference=None;first_outputs={}
            turns=(1,2) if task['benchmark']=='mtbench' else (1,)
            for turn in turns:
                record=conversation_record(task,turn,first_reference) if task['benchmark']=='mtbench' else task
                ref=asdict(engine.generate(request_for(record,4096),'AR'))
                ref.update(benchmark=record['benchmark'],quality=score(record,ref['text']),
                    context_policy='strict-AR first-turn history' if turn==2 else 'original request')
                append(out/'references.jsonl',ref)
                if turn==1:first_reference=ref['text']
                seen={};quality_cache={ref['text']:ref['quality']}
                for stage in stages(record):
                    limit=4096 if stage=='natural' else 128 if stage=='prefix' else 256
                    base_order=list(ARMS);random.Random('paper:'+record['id']+':'+stage).shuffle(base_order)
                    fixed_strict=None
                    for repeat in range(REPEATS[stage]):
                        jobs=base_order if repeat==0 else list(reversed(base_order))
                        for arm in jobs:
                            row=execute(engine,record,arm,limit,tree_config,request_factory=request_for,fixed=stage=='fixed')
                            row.update(stage=stage,repeat=repeat,stop_policy='ignore-eos' if stage=='fixed' else 'natural-eos',
                                context_policy=ref['context_policy'])
                            if (row['prompt_sha256'],row['snapshot_id'])!=(ref['prompt_sha256'],ref['snapshot_id']):
                                raise ValueError('Changed matched Target input')
                            if stage!='fixed' and arm in ('DFlash','BudgetTree'):
                                if row['token_ids']!=ref['token_ids'][:limit]:raise ValueError('Strict AR mismatch '+record['id'])
                                if stage=='natural' and (row['text'],row['finish_reason'])!=(ref['text'],ref['finish_reason']):
                                    raise ValueError('Strict EOS mismatch '+record['id'])
                            if stage=='fixed':
                                if len(row['token_ids'])!=256:raise ValueError('Fixed completion length mismatch')
                                if arm in ('DFlash','BudgetTree'):
                                    if fixed_strict is not None and row['token_ids']!=fixed_strict:raise ValueError('Strict fixed mismatch')
                                    fixed_strict=row['token_ids']
                            if row['asd'] and not 0<=row['asd']['spent']<=8:raise ValueError('Budget exceeded')
                            key=stage,arm
                            if key in seen and seen[key]!=signature(row):raise ValueError('Unstable repeated trajectory')
                            seen[key]=signature(row)
                            if stage=='natural':
                                if row['text'] not in quality_cache:quality_cache[row['text']]=score(record,row['text'])
                                row['quality']=quality_cache[row['text']]
                                if repeat==0 and task['benchmark'] in ('mtbench','alpaca'):
                                    first_outputs.setdefault(arm,[]).append(row)
                            append(out/'rows.jsonl',row);count+=1;peak=max(peak,row['peak_gpu_bytes'])
            if task['benchmark']=='mtbench':
                # Secondary full dialogue follows EACH arm's actual first answer. Its input hashes
                # intentionally differ and are never mixed with the matched-history performance table.
                for arm in ARMS:
                    initial=first_outputs[arm][0]
                    record=conversation_record(task,2,initial['text'])
                    row=execute(engine,record,arm,4096,tree_config,request_factory=request_for)
                    row.update(stage='rollout',repeat=0,conversation_id=task['id'],turn=2,
                        first_token_ids=initial['token_ids'],first_text=initial['text'],
                        first_latency_seconds=initial['latency_seconds'],context_policy='own realized first-turn history')
                    append(out/'rollout.jsonl',row);peak=max(peak,row['peak_gpu_bytes'])
            info.update(count=count,completed_tasks=index+1,peak_gpu_bytes=peak,compiler=engine._regional.summary())
            write(out/'provenance.json',info)
            print('completed',task['benchmark'],task['id'],count,flush=True)
            if index+1==sum(r['benchmark'] in ('gsm8k','math500') for r in selected):
                write(out/'math-complete.json',dict(count=count,status='complete',parameter_changes=0))
        cache_bytes=sum(p.stat().st_size for name in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')
            for p in Path(os.environ[name]).rglob('*') if p.is_file())
        if peak>46*2**30 or cache_bytes>4*2**30:raise ValueError('Paper resource bounds exceeded')
        if engine._regional.summary()['graph_count']!=info['compiler_after_warmup']['graph_count']:
            raise ValueError('New formal compiler graph')
        info.update(status='complete',cache_bytes=cache_bytes,compiler=engine._regional.summary())
    except Exception as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        info['original_unchanged']=before==protected(ROOT);write(out/'provenance.json',info)
        if engine is not None:engine.close()
        if not info['original_unchanged']:raise ValueError('Previous published evidence changed')


if __name__=='__main__':main()
