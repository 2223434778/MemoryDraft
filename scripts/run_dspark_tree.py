"""Eight-arm, paired replay with immutable DFlash assets and a DSpark adapter."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import statistics
import time
from unittest.mock import patch

import torch
from torch import nn

from memorydraft.backends.asd import ASDConfig
from memorydraft.backends.asd_cpu import CPUASDVerifier
from memorydraft.backends.budget_tree import BudgetTreeConfig, TreeASDVerifier
from memorydraft.backends.dspark import load_dspark
from memorydraft.backends.dspark_reuse import generate_dspark
from memorydraft.backends.dspark_tree import build_dspark_tree
from memorydraft.backends.regional_compile import runtime_rmsnorm
from memorydraft.evaluation.asd_paper import BENCHMARKS, conversation_record, request_for, score, canonical
from memorydraft.evaluation.asd_paper_extension import load_records, validate_policy
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.run_asd_tree import execute as execute_dflash, calibrate
from scripts.run_runtime_migration import append
from scripts.diagnose_transport_online import write

ROOT=Path(__file__).resolve().parents[1]
ARMS=('DFlash','DFlash-ASD','DFlash-Tree','DFlash-Tree-ASD','DSpark','DSpark-ASD','DSpark-Tree','DSpark-Tree-ASD')
REPEATS={'natural':2,'prefix':1,'fixed':1}
MAPPING={'DFlash':'DFlash','DFlash-ASD':'ASD','DFlash-Tree':'BudgetTree','DFlash-Tree-ASD':'BudgetTree-ASD'}


class DraftPair(nn.ModuleDict):
    @property
    def device(self):return next(self.parameters()).device

    @property
    def dtype(self):return next(self.parameters()).dtype


def execute(engine,dspark,record,arm,limit,trees,*,fixed=False):
    if arm in MAPPING:
        row=execute_dflash(engine,record,MAPPING[arm],limit,trees['DFlash'],fixed=fixed,request_factory=request_for)
    else:
        from memorydraft.runtime import engine as module
        verifier=(TreeASDVerifier(ASDConfig(budget=8)) if arm=='DSpark-Tree-ASD' else
                  CPUASDVerifier(ASDConfig(budget=8)) if arm=='DSpark-ASD' else None)
        detail=None
        def hook(target,unused,ids,**kwargs):
            nonlocal detail
            detail=generate_dspark(target,dspark,ids,max_new_tokens=kwargs['max_new_tokens'],
                stop_token_ids=() if fixed else kwargs['stop_token_ids'],verifier=verifier,
                tree_config=trees['DSpark'] if '-Tree' in arm else None,
                on_tokens=kwargs.get('on_tokens'),should_stop=kwargs.get('should_stop'))
            return detail
        engine.synchronize();started=time.perf_counter()
        with patch.object(module,'generate_reuse',hook):result=engine.generate(request_for(record,limit),'DFlash')
        engine.synchronize()
        row=asdict(result)
        row.update(engine_latency_seconds=row['latency_seconds'],latency_seconds=time.perf_counter()-started,
                   engine_method='DFlash',rounds=detail.rounds,asd=verifier.summary() if verifier else None)
    row.update(method=arm,arm=arm,benchmark=record['benchmark'],max_new_tokens=limit,
               approximate=arm.endswith('-ASD'),experimental_approximate=arm.endswith('-ASD'))
    return row


@torch.inference_mode()
def calibrate_dspark(engine,draft,warm,out,original):
    captured=[];times=[];forward=draft.forward
    def measured(**kwargs):
        engine.synchronize();started=time.perf_counter();result=forward(**kwargs);engine.synchronize()
        times.append(time.perf_counter()-started);return result
    ids=engine.tokenizer.apply_chat_template(list(request_for(warm,128).messages),add_generation_prompt=True,
        return_tensors='pt',**engine.config.template_options).to(engine.target.device)
    with patch.object(draft,'forward',measured):
        result=generate_dspark(engine.target,draft,ids,max_new_tokens=128,stop_token_ids=engine.stops,
            on_draft=lambda anchor,base:captured.append((anchor.clone(),base.clone())) if not captured else None)
    costs=replace(original.costs,draft_seconds=statistics.median(times),auxiliary_seconds=0.)
    config=BudgetTreeConfig(costs)
    auxiliary=[]
    for _ in range(4):
        engine.synchronize();started=time.perf_counter()
        build_dspark_tree(*captured[0],draft,config,128,stop_token_ids=engine.stops)
        engine.synchronize();auxiliary.append(time.perf_counter()-started)
    config=replace(config,costs=replace(costs,auxiliary_seconds=statistics.median(auxiliary[1:])))
    write(out/'dspark-calibration.json',dict(config=asdict(config),draft_seconds=times,
        auxiliary_seconds=auxiliary,warmup_tokens=result.sequences.shape[1]-ids.shape[1],
        target_verify_table='shared per-card Target table; separately measured DSpark draft/tree cost'))
    return config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,choices=(0,1),required=True)
    args=parser.parse_args();out=ROOT/f'results/raw/dspark-tree/part-{args.shard}'
    out.mkdir(parents=True,exist_ok=False)
    records=load_records(ROOT)
    selected=[r for b in BENCHMARKS for i,r in enumerate([x for x in records if x['benchmark']==b][:4]) if i%2==args.shard]
    info=dict(status='started',shard=args.shard,task_ids=[r['id'] for r in selected],arms=ARMS,repeats=REPEATS,
        split='previously_observed_extension_replay',native_proposals={'DFlash':15,'DSpark':7},confidence_pruning=False)
    write(out/'provenance.json',info)
    engine=None
    try:
        models=ROOT.parent/'models'
        started=time.perf_counter()
        config=RuntimeConfig(backend='dflash',target=str(models/'qwen3-8b'),draft=str(models/'dflash-qwen3-8b'),
            target_revision='b968826d9c46dd6066d109eabc6255188de91218',max_new_tokens=4096)
        engine=load_engine(config)
        dspark=load_dspark(models/'dspark-qwen3-8b-block7',engine.target,engine.draft)
        from scripts.check_dspark_backbone import check
        write(out/'upstream-backbone-check.json',check(engine,dspark,ROOT/'artifacts/dspark-upstream'))
        regional=runtime_rmsnorm(engine.target,DraftPair({'dflash':engine.draft,'dspark':dspark}))
        engine._regional=regional;engine._regional_context=regional.activate('rmsnorm-inductor')
        engine._regional_context.__enter__();engine.execution_profile='rmsnorm-inductor'
        engine.config=replace(config,execution_profile='rmsnorm-inductor')
        info.update(load_seconds=time.perf_counter()-started,target_config=engine.target.config.to_dict(),
            runtime=engine.describe(),shared_target_weights_verified=True,
            draft_parameters={k:sum(p.numel() for p in v.parameters()) for k,v in [('DFlash',engine.draft),('DSpark',dspark)]})
        warm=dict(benchmark='gsm8k',id='warmup',question='What is 3 plus 4?',answer='#### 7')
        append(out/'warmup.jsonl',asdict(engine.generate(request_for(warm,32),'AR')))
        trees={'DFlash':calibrate(engine,warm,out)}
        trees['DSpark']=calibrate_dspark(engine,dspark,warm,out,trees['DFlash'])
        for arm in ARMS:append(out/'warmup.jsonl',execute(engine,dspark,warm,arm,64,trees))
        info.update(compiler_after_warmup=regional.summary(),tree_configs={k:asdict(v) for k,v in trees.items()})
        graphs=len(regional.backend.events)
        preflight=[dict(id=r['id'],quality=score(r,canonical(r))) for r in selected if canonical(r) is not None]
        if any(not item['quality']['correct'] for item in preflight):raise ValueError('Canonical scorer failed')
        write(out/'canonical-preflight.json',preflight)
        write(out/'provenance.json',info)
        count=0;differences=[]
        for original in selected:
            first_answer=None
            for turn in ((1,2) if original['benchmark']=='mtbench' else (0,)):
                record=conversation_record(original,turn,first_answer) if turn else original
                ref=asdict(engine.generate(request_for(record,4096),'AR'))
                ref.update(benchmark=record['benchmark'],quality=score(record,ref['text']),messages=list(request_for(record,4096).messages))
                append(out/'references.jsonl',ref)
                if turn==1:first_answer=ref['text']
                order=list(ARMS);random.Random('dspark-tree-v1/'+record['id']).shuffle(order)
                signatures={}
                for stage in REPEATS:
                    for repeat in range(REPEATS[stage]):
                        methods=order if repeat==0 else list(reversed(order))
                        for arm in methods:
                            row=execute(engine,dspark,record,arm,4096 if stage=='natural' else 128 if stage=='prefix' else 256,
                                        trees,fixed=stage=='fixed')
                            row.update(stage=stage,repeat=repeat,quality=score(record,row['text']) if stage=='natural' else None,
                                       stop_policy='ignore-eos' if stage=='fixed' else 'eos')
                            if (row['prompt_sha256'],row['snapshot_id'])!=(ref['prompt_sha256'],ref['snapshot_id']):
                                raise ValueError('Target input/snapshot changed')
                            if row['asd']:validate_policy(row,ASDConfig(budget=8))
                            if stage=='fixed' and len(row['token_ids'])!=256:raise ValueError('Incomplete fixed work')
                            signature=(row['token_ids'],row['rounds'],row['asd'],row['finish_reason'])
                            key=stage,arm
                            if key in signatures and signatures[key]!=signature:raise ValueError('Unstable repeat: '+arm)
                            signatures[key]=signature
                            if row['counters']['compile_graphs_after']!=graphs:raise ValueError('Formal compilation occurred')
                            if stage=='natural' and repeat==0 and not arm.endswith('-ASD') and row['token_ids']!=ref['token_ids']:
                                index=next((i for i,(x,y) in enumerate(zip(row['token_ids'],ref['token_ids'])) if x!=y),min(len(row['token_ids']),len(ref['token_ids'])))
                                differences.append(dict(request_id=record['id'],arm=arm,index=index,
                                    tokens=len(row['token_ids']),ar_tokens=len(ref['token_ids'])))
                            append(out/'rows.jsonl',row)
                count+=1
                info.update(completed_requests=count,identity_differences=differences)
                write(out/'provenance.json',info)
                print(json.dumps(dict(completed=count,request_id=record['id'],benchmark=record['benchmark'])),flush=True)
        info.update(status='complete',compiler_end=regional.summary())
    except Exception as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        write(out/'provenance.json',info)
        if engine is not None:engine.close()


if __name__=='__main__':main()
