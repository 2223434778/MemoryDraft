"""Independent cross-task confirmation with adjacent strict fixed-work controls."""
import argparse
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import random
import time

from memorydraft.evaluation.asd_joint_tree import frozen_config
from memorydraft.evaluation.asd_paper import BENCHMARKS,canonical,conversation_record,request_for,score
from memorydraft.evaluation.asd_paper_extension import (ARMS,CONTROLS,REPEATS,STUDY,frontier_ids,
    load_records,signature,validate_policy)
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig,load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import protected
from scripts.run_asd_joint_tree import execute
from scripts.run_runtime_migration import append

ROOT=Path(__file__).resolve().parents[1]


def main(*,continuation=False):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard',type=int,choices=(0,1),required=True);args=parser.parse_args()
    out=ROOT/f'results/raw/{STUDY}/part-{args.shard}';out.mkdir(parents=True,exist_ok=False)
    previous=protected(ROOT);engine=None;pool=load_records(ROOT);frontier=frontier_ids(pool)
    selected=[r for b in BENCHMARKS for i,r in enumerate(r for r in pool if r['benchmark']==b) if i%2==args.shard]
    info=dict(status='started',shard=args.shard,task_ids=[r['id'] for r in selected],arms=ARMS,
        repeats=REPEATS,controls={k:asdict(v) for k,v in CONTROLS.items()},limit=4096,
        selection='Frozen unused public source IDs; no outcome-based exclusions')
    write(out/'provenance.json',info)
    try:
        reuse=None
        if continuation:
            from memorydraft.evaluation.asd_extension_resume import resume_completed
            reuse=resume_completed(ROOT,ROOT.parent/'asd-paper-extension',out,selected,args.shard)
            info.update(reused_completed_tasks=reuse['completed_tasks'],
                strict_identity_policy='observe-forward-geometry',strict_AR_bit_identity_claim=False)
        def reject(message,row):
            append(out/'rejected.jsonl',dict(row,rejection=message))
            raise ValueError(message)
        def identity(left,right,comparison):
            from memorydraft.evaluation.asd_extension_resume import identity_difference
            difference=identity_difference(left,right,comparison)
            if difference:
                append(out/'geometry-differences.jsonl',difference)
                if not continuation:reject('Strict token identity mismatch: '+comparison,left)
        started=time.perf_counter()
        engine=load_engine(replace(RuntimeConfig.load(ROOT/'configs/runtime-dflash-rmsnorm.json'),db=None,max_new_tokens=4096))
        info['load_seconds']=time.perf_counter()-started
        started=time.perf_counter();tree_config,cost_source=frozen_config(ROOT,args.shard)
        info['cost_table_read_seconds']=time.perf_counter()-started;write(out/'cost-table.json',cost_source)
        if engine.draft.block_size!=16:raise ValueError('Expected native16')
        warm=dict(id='warmup',benchmark='gsm8k',question='What is 3 plus 4?',answer='#### 7')
        append(out/'warmup.jsonl',asdict(engine.generate(request_for(warm,32),'AR')))
        for arm in ARMS:append(out/'warmup.jsonl',execute(engine,warm,arm,64,tree_config,request_factory=request_for))
        append(out/'warmup.jsonl',execute(engine,warm,'BudgetTree-ASD',256,tree_config,request_factory=request_for,fixed=True))
        zero=execute(engine,warm,'BudgetTree-ASD',64,tree_config,request_factory=request_for,zero=True)
        strict=execute(engine,warm,'BudgetTree',64,tree_config,request_factory=request_for)
        append(out/'warmup.jsonl',zero);append(out/'warmup.jsonl',strict)
        if zero['token_ids']!=strict['token_ids'] or zero['asd']['spent']!=0:raise ValueError('Zero-budget mismatch')
        info.update(runtime=provenance(engine),compiler_after_warmup=engine._regional.summary())
        preflight=[]
        for task in selected:
            reference=canonical(task)
            if reference is not None:preflight.append(dict(request_id=task['id'],quality=score(task,reference)))
        write(out/'canonical-preflight.json',preflight)
        info['canonical_failed_ids']=[r['request_id'] for r in preflight if not r['quality']['correct']]
        old=reuse['original_provenance'] if reuse else {}
        count=old.get('count',0);control_count=old.get('frontier_count',0)
        rollout_count=old.get('rollout_count',0);peak=old.get('peak_gpu_bytes',0)
        for index,task in enumerate(selected):
            if reuse and index<reuse['completed_tasks']:continue
            peer=out.parent/f'part-{1-args.shard}/provenance.json'
            if peer.exists():
                try:peer_status=json.loads(peer.read_bytes())['status']
                except json.JSONDecodeError:peer_status='writing'
                if peer_status=='failed':raise RuntimeError('Other owned extension worker failed')
            first_reference=None;first_outputs={}
            for turn in ((1,2) if task['benchmark']=='mtbench' else (1,)):
                record=conversation_record(task,turn,first_reference) if task['benchmark']=='mtbench' else task
                ref=asdict(engine.generate(request_for(record,4096),'AR'))
                ref.update(benchmark=task['benchmark'],quality=score(record,ref['text']),
                    context_policy='strict-AR first-turn history' if turn==2 else 'original request')
                append(out/'references.jsonl',ref)
                if turn==1:first_reference=ref['text']
                quality_cache={ref['text']:ref['quality']};seen={}
                def run(arm,limit,*,stage,repeat=0,control=None,role=None):
                    config=CONTROLS[control] if control else None
                    row=execute(engine,record,arm,limit,tree_config,request_factory=request_for,
                        fixed=stage=='fixed',asd_config=config)
                    row.update(stage=stage,repeat=repeat,stop_policy='ignore-eos' if stage=='fixed' else 'natural-eos',
                        context_policy=ref['context_policy'])
                    # Keep the actual last output if a later row check fails.
                    write(out/'last-generated.json',row)
                    if (row['prompt_sha256'],row['snapshot_id'])!=(ref['prompt_sha256'],ref['snapshot_id']):
                        raise ValueError('Changed matched Target input')
                    if control:
                        row.update(control=control,asd_config=asdict(config));validate_policy(row,config)
                    elif row['asd'] and not 0<=row['asd']['spent']<=8:raise ValueError('Main budget exceeded')
                    if stage=='fixed':
                        if len(row['token_ids'])!=256:raise ValueError('Wrong fixed-work completion count')
                        if role:row['role']=role
                    elif arm in ('DFlash','BudgetTree'):
                        identity(row,dict(ref,token_ids=ref['token_ids'][:limit]),'strict-vs-AR')
                        if stage=='natural' and row['token_ids']==ref['token_ids'] and (row['text'],row['finish_reason'])!=(ref['text'],ref['finish_reason']):
                            raise ValueError('Strict natural EOS mismatch')
                    if stage=='natural':
                        if row['text'] not in quality_cache:quality_cache[row['text']]=score(record,row['text'])
                        row['quality']=quality_cache[row['text']]
                    return row
                for stage,limit in (('natural',4096),('prefix',128)):
                    order=list(ARMS);random.Random('extension:'+record['id']+':'+stage).shuffle(order)
                    for repeat in range(REPEATS[stage]):
                        for arm in (order if repeat==0 else list(reversed(order))):
                            row=run(arm,limit,stage=stage,repeat=repeat)
                            if (stage,arm) in seen and seen[stage,arm]!=signature(row):reject('Repeated trajectory changed',row)
                            seen[stage,arm]=signature(row);append(out/'rows.jsonl',row);count+=1
                            peak=max(peak,row['peak_gpu_bytes'])
                            if turn==1 and stage=='natural' and repeat==0:first_outputs[arm]=row
                pairs=[('DFlash','ASD'),('BudgetTree','BudgetTree-ASD')]
                random.Random('extension-fixed:'+record['id']).shuffle(pairs);fixed_strict=None
                for strict_arm,candidate in pairs:
                    before=run(strict_arm,256,stage='fixed',role='before')
                    append(out/'rows.jsonl',before);count+=1
                    row=run(candidate,256,stage='fixed',role='candidate')
                    append(out/'rows.jsonl',row);count+=1
                    after=run(strict_arm,256,stage='fixed',repeat=1,role='after')
                    append(out/'rows.jsonl',after);count+=1
                    if signature(before)!=signature(after):reject('Strict fixed repeat mismatch',after)
                    if fixed_strict is not None:identity(before,fixed_strict,'fixed-tree-vs-linear')
                    fixed_strict=before
                    identity(dict(before,token_ids=before['token_ids'][:len(ref['token_ids'][:256])]),
                        dict(ref,token_ids=ref['token_ids'][:256]),'fixed-prefix-vs-AR')
                    peak=max(peak,*[r['peak_gpu_bytes'] for r in (before,row,after)])
                if task['id'] in frontier:
                    controls=list(CONTROLS);random.Random('extension-frontier:'+record['id']).shuffle(controls)
                    for control in controls:
                        for stage,limit in (('natural',4096),('fixed',256)):
                            arms=('ASD','BudgetTree-ASD') if controls.index(control)%2==0 else ('BudgetTree-ASD','ASD')
                            for arm in arms:
                                row=run(arm,limit,stage=stage,control=control)
                                append(out/'frontier.jsonl',row);control_count+=1;peak=max(peak,row['peak_gpu_bytes'])
            if task['benchmark']=='mtbench':
                strict_rollout=None
                for arm in ARMS:
                    initial=first_outputs[arm];record=conversation_record(task,2,initial['text'])
                    row=execute(engine,record,arm,4096,tree_config,request_factory=request_for)
                    row.update(stage='rollout',repeat=0,conversation_id=task['id'],turn=2,
                        first_token_ids=initial['token_ids'],first_text=initial['text'],
                        first_latency_seconds=initial['latency_seconds'],context_policy='own realized first-turn history')
                    if arm in ('DFlash','BudgetTree'):
                        if strict_rollout is not None:
                            if row['prompt_sha256']==strict_rollout['prompt_sha256']:identity(row,strict_rollout,'own-history-tree-vs-linear')
                            elif not continuation:reject('Strict own-history input mismatch',row)
                            else:append(out/'geometry-differences.jsonl',dict(request_id=row['request_id'],arm=arm,
                                stage='rollout',comparison='different-own-history-inputs',
                                left_prompt_sha256=row['prompt_sha256'],right_prompt_sha256=strict_rollout['prompt_sha256'],
                                interpretation='Different realized first turns; not a matched-input comparison'))
                        strict_rollout=row
                    append(out/'rollout.jsonl',row);rollout_count+=1;peak=max(peak,row['peak_gpu_bytes'])
            if peak>46*2**30:raise ValueError('GPU memory limit exceeded')
            info.update(count=count,frontier_count=control_count,rollout_count=rollout_count,completed_tasks=index+1,
                peak_gpu_bytes=peak,compiler=engine._regional.summary())
            write(out/'provenance.json',info)
            print('completed',task['benchmark'],task['id'],count,control_count,flush=True)
        cache_bytes=sum(p.stat().st_size for key in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')
            for p in Path(os.environ[key]).rglob('*') if p.is_file())
        if cache_bytes>4*2**30:raise ValueError('Compiler cache limit exceeded')
        if engine._regional.summary()['graph_count']!=info['compiler_after_warmup']['graph_count']:
            raise ValueError('New formal compiler graph')
        if reuse:
            import hashlib
            directory=Path(reuse['original_root'])/f'results/raw/{STUDY}/part-{args.shard}'
            for name,digest in reuse['original_payload_sha256'].items():
                path=directory/(name+'.jsonl');data=path.read_bytes() if path.exists() else b''
                if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Original interrupted payload changed')
            info['reuse_payloads_unchanged']=True
        info.update(status='complete',cache_bytes=cache_bytes)
    except Exception as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        info['original_unchanged']=previous==protected(ROOT);write(out/'provenance.json',info)
        if engine is not None:engine.close()
        if not info['original_unchanged']:raise ValueError('Old published evidence changed')


if __name__=='__main__':main()
