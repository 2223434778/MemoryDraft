"""Five new workload families with only three necessary same-machine arms."""
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

from memorydraft.evaluation.expanded_suite import ARMS,load_records,request_for,score
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig,load_engine
from scripts import run_asd_tree as common
from scripts import run_tree_execution as current
from scripts.diagnose_transport_online import write
from scripts.run_runtime_migration import append

ROOT=Path(__file__).resolve().parents[1]


def clipped(record,tokenizer):
    if 'clip_input_tokens' not in record:return dict(record)
    ids=tokenizer.encode(record['prompt'],add_special_tokens=False)
    cap=record['clip_input_tokens']
    prompt=record['prompt'] if len(ids)<=cap else tokenizer.decode(ids[:cap//2],skip_special_tokens=True)+tokenizer.decode(ids[-cap//2:],skip_special_tokens=True)
    return dict(record,prompt=prompt,original_prompt_tokens=len(ids),input_clipped=len(ids)>cap)


def execute(engine,record,arm,tree):
    if arm=='BufferedKV-ASD':
        with patch.object(current,'request_for',request_for):
            return current.execute(engine,record,arm,record['limit'],tree)
    return common.execute(engine,record,arm,record['limit'],tree,request_factory=request_for)


def main():
    out=ROOT/'results/raw/expanded-suite';out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'data/research/expanded-suite/manifest.json').read_bytes())
    for upstream in manifest['upstream'].values():
        for path,digest in upstream['files'].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed scorer: '+path)
    info=dict(status='started',arms=ARMS,expected=manifest['count']*len(ARMS),
        baseline_policy='DFlash and ASD only rerun for matched FP32/A6000/new prompts; external EAGLE/DDTree/BASTION retained as cited literature, not mixed timing',
        dtype='float32',attention='eager',temperature=0,max_input_tokens=4096,source_manifest=manifest)
    write(out/'provenance.json',info)
    engine,rows=None,[]
    try:
        start=time.perf_counter()
        engine=load_engine(replace(RuntimeConfig.load(ROOT/'configs/runtime-dflash-rmsnorm.json'),db=None,max_new_tokens=4096,max_input_tokens=4096))
        info['load_seconds']=time.perf_counter()-start
        prepared=[clipped(r,engine.tokenizer) for r in load_records(ROOT)]
        pool=[];excluded=[]
        for record in prepared:
            ids=engine.tokenizer.apply_chat_template(list(request_for(record,record['limit']).messages),
                add_generation_prompt=True,**engine.config.template_options)
            if len(ids)>4096:excluded.append(dict(id=record['id'],prompt_tokens=len(ids),reason='input admission limit'))
            else:pool.append(record)
        write(out/'admission.json',dict(selected=len(prepared),admitted=len(pool),excluded=excluded))
        info['expected']=len(pool)*len(ARMS)
        write(out/'tasks.json',pool)
        # Dependency/test-artifact preflight, with no generated model answers.
        preflight=[]
        for r in pool:
            if r['benchmark']=='canitedit':
                preflight.append(dict(id=r['id'],canonical=score(r,r['canonical']),before=score(r,r['before'])))
        write(out/'canonical-preflight.json',preflight)
        unavailable={v['id'] for v in preflight if v['canonical'].get('correct') is not True}
        warm=dict(id='warmup',benchmark='gsm8k',question='What is 3 plus 4?',answer='#### 7')
        started=time.perf_counter()
        tree=common.calibrate(engine,warm,out)
        info['calibration_seconds']=time.perf_counter()-started
        warm_record=dict(id='expanded-warm',benchmark='ifeval',prompt='What is 3 plus 4?',limit=64)
        for arm in ARMS:append(out/'warmup.jsonl',execute(engine,warm_record,arm,tree))
        info.update(runtime=provenance(engine),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        write(out/'provenance.json',info)
        for index,record in enumerate(pool):
            order=list(ARMS);random.Random('expanded:'+record['id']).shuffle(order)
            cached={};identity=None
            for arm in order:
                row=execute(engine,record,arm,tree)
                signature=row['prompt_sha256'],row['snapshot_id']
                if identity is not None and identity!=signature:raise ValueError('Unequal method inputs')
                identity=signature
                if row['text'] not in cached:
                    cached[row['text']]=(dict(correct=None,evaluation_error='canonical hidden tests unavailable; excluded uniformly') if record['id'] in unavailable else score(record,row['text']))
                row.update(stage='natural',repeat=0,quality=cached[row['text']])
                rows.append(row);append(out/'rows.jsonl',row)
                info.update(timed_generations=len(rows),last_task=record['id'],last_arm=arm)
                write(out/'provenance.json',info)
            print(json.dumps(dict(completed=index+1,total=len(pool),benchmark=record['benchmark'],rows=len(rows))),flush=True)
        info.update(status='completed',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            scored_errors=sum('evaluation_error' in r['quality'] for r in rows))
    except BaseException as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        if engine is not None:info['compiler']=engine._regional.summary();engine.close()
        write(out/'provenance.json',info)


if __name__=='__main__':main()
