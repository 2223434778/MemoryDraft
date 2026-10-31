"""Matched AR, strict cost tree and official BASTION on frozen new workloads."""
from dataclasses import asdict,replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import time

from memorydraft.backends.budget_tree import BudgetTreeConfig,TreeCosts
from memorydraft.evaluation.expanded_suite import load_records,request_for,score
from memorydraft.memory.store import Snapshot
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig,load_engine
from scripts import run_asd_tree as common
from scripts import run_expanded_suite as previous
from scripts.diagnose_transport_online import write
from scripts.run_external_trees import external_execute,load_upstreams
from scripts.run_runtime_migration import append

ROOT=Path(__file__).resolve().parents[1]
ARMS=('AR','DFlash','BudgetTree','BASTION','BufferedKV-ASD')
STRICT=('DFlash','BudgetTree','BASTION')


def frozen_inputs(root=ROOT):
    files=('results/expanded-suite/rows.jsonl.gz','results/expanded-suite/calibration.json',
           'results/expanded-suite/quality-v3-canonical-preflight.json',
           'results/external-trees/bastion-calibration.json','results/external-trees/bastion-profile.json')
    for folder in ('expanded-suite','external-trees'):
        manifest=json.loads((root/'results'/folder/'publication.json').read_bytes())
        for path in files:
            p=root/path
            if p.parent.name==folder and hashlib.sha256(p.read_bytes()).hexdigest()!=manifest['payloads'][p.name]:
                raise ValueError('Changed frozen reference: '+path)
    prior=[json.loads(s) for s in gzip.decompress((root/files[0]).read_bytes()).decode().splitlines()]
    costs=json.loads((root/files[1]).read_bytes())['costs']
    costs=dict(costs,contexts=tuple(costs['contexts']),nodes=tuple(costs['nodes']),seconds=tuple(map(tuple,costs['seconds'])))
    canonical=json.loads((root/files[2]).read_bytes())
    unavailable={r['id'] for r in canonical if r['canonical'].get('correct') is not True}
    if unavailable!={'canitedit:78_llm_inference'}:raise ValueError('Changed canonical scoring mask')
    return prior,BudgetTreeConfig(TreeCosts(**costs)),unavailable,{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}


def prepared_records(engine,prior):
    refs={r['request_id']:r for r in prior if r['arm']=='DFlash'}
    records=[previous.clipped(r,engine.tokenizer) for r in load_records(ROOT)]
    if len(records)!=len(refs):raise ValueError('Changed frozen request coverage')
    for r in records:
        request=request_for(r,r['limit'])
        ids=engine.tokenizer.apply_chat_template(list(request.messages),add_generation_prompt=True,**engine.config.template_options)
        signature=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),Snapshot(request.user_id,request.order,()).snapshot_id
        ref=refs[r['id']]
        if len(ids)>4096 or signature!=(ref['prompt_sha256'],ref['snapshot_id']) or r['limit']!=ref['max_new_tokens']:
            raise ValueError('Changed frozen input or output limit: '+r['id'])
    return records


def execute(engine,record,arm,tree,dd,bastion,profile):
    if arm=='AR':
        r=asdict(engine.generate(request_for(record,record['limit']),'AR'))
        return dict(r,arm=arm,benchmark=record['benchmark'],rounds=[],max_new_tokens=record['limit'])
    if arm=='BASTION':
        return external_execute(engine,record,arm,record['limit'],dd,bastion,profile,request_factory=request_for)
    return previous.execute(engine,record,arm,tree)


def main():
    out=ROOT/'results/raw/expanded-baselines';out.mkdir(parents=True,exist_ok=False)
    prior,tree,unavailable,hashes=frozen_inputs()
    dd,bastion,_,_,upstreams=load_upstreams(ROOT)
    write(out/'upstreams.json',upstreams)
    write(out/'frozen-inputs.json',hashes)
    profile=json.loads((ROOT/'results/external-trees/bastion-profile.json').read_bytes())
    calibration=ROOT/'results/external-trees/bastion-calibration.json'
    os.environ['CALIBRATION_JSON_PATH']=str(calibration)
    write(out/'tree-costs.json',asdict(tree))
    write(out/'bastion-profile.json',profile)
    write(out/'bastion-calibration.json',json.loads(calibration.read_bytes()))
    info=dict(status='started',arms=ARMS,expected=182*len(ARMS),dtype='float32',attention='eager',temperature=0,
        max_input_tokens=4096,cost_policy='Frozen same-GPU synthetic costs from expanded-suite and external-trees; no answer tuning',
        scoring_policy='Same v3 scorer and canonical mask as expanded-suite; all online inference costs included')
    write(out/'provenance.json',info)
    engine=None
    try:
        started=time.perf_counter()
        engine=load_engine(replace(RuntimeConfig.load(ROOT/'configs/runtime-dflash-rmsnorm.json'),db=None,max_new_tokens=4096,max_input_tokens=4096))
        info['load_seconds']=time.perf_counter()-started
        pool=prepared_records(engine,prior)
        write(out/'tasks.json',pool)
        warm=dict(id='expanded-baselines-warm',benchmark='ifeval',prompt='What is 3 plus 4?',limit=64)
        for arm in ARMS:append(out/'warmup.jsonl',execute(engine,warm,arm,tree,dd,bastion,profile))
        info.update(runtime=provenance(engine),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        write(out/'provenance.json',info)
        completed=0
        for record in pool:
            order=list(ARMS);random.Random('expanded-baselines:'+record['id']).shuffle(order)
            cache={}
            for arm in order:
                row=execute(engine,record,arm,tree,dd,bastion,profile)
                ref=next(r for r in prior if r['request_id']==record['id'] and r['arm']=='DFlash')
                if (row['prompt_sha256'],row['snapshot_id'])!=(ref['prompt_sha256'],ref['snapshot_id']):
                    raise ValueError('Changed execution input: '+record['id'])
                if row['text'] not in cache:
                    cache[row['text']]=dict(correct=None,evaluation_error='canonical tests unavailable') if record['id'] in unavailable else score(record,row['text'])
                row.update(stage='natural',repeat=0,quality=cache[row['text']],quality_revision=3)
                append(out/'rows.jsonl',row);completed+=1
                info.update(timed_generations=completed,last_task=record['id'],last_arm=arm)
                write(out/'provenance.json',info)
            print(json.dumps(dict(completed=completed,expected=info['expected'],task=record['id'])),flush=True)
        info.update(status='completed',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    except BaseException as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        if engine is not None:info['compiler']=engine._regional.summary();engine.close()
        write(out/'provenance.json',info)


if __name__=='__main__':main()
