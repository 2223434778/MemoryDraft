"""Same-input fixed-work AR bracketed by strict DFlash; no policy changes."""
import argparse
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import time

from memorydraft.evaluation.asd_paper import request_for
from memorydraft.evaluation.asd_paper_metrics import ROLES,STUDY,inputs,read_rows,validate_triplet
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig,load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import protected
from scripts.run_asd_tree import execute
from scripts.run_runtime_migration import append

ROOT=Path(__file__).resolve().parents[1]


def fixed_ar(engine,record,limit=256):
    stops=engine.stops
    engine.synchronize();started=time.perf_counter()
    try:
        engine.stops=[]
        row=asdict(engine.generate(request_for(record,limit),'AR'))
    finally:engine.stops=stops
    engine.synchronize()
    row.update(engine_latency_seconds=row['latency_seconds'],latency_seconds=time.perf_counter()-started)
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,choices=(0,1),required=True)
    args=parser.parse_args();out=ROOT/f'results/raw/{STUDY}/part-{args.shard}';out.mkdir(parents=True,exist_ok=False)
    before=protected(ROOT);engine=None
    records,refs=inputs(ROOT,args.shard)
    old={r['request_id']:r for r in read_rows(ROOT,'rows') if r['stage']=='fixed' and r['arm']=='DFlash'}
    info=dict(status='started',shard=args.shard,requests=len(records),roles=ROLES,source_study='asd-paper',count=0,peak_gpu_bytes=0)
    write(out/'provenance.json',info)
    try:
        start=time.perf_counter()
        engine=load_engine(replace(RuntimeConfig.load(ROOT/'configs/runtime-dflash-rmsnorm.json'),db=None,max_new_tokens=4096))
        info['load_seconds']=time.perf_counter()-start
        warm=dict(id='metrics-warmup',benchmark='gsm8k',question='What is 3 plus 4?',answer='#### 7')
        append(out/'warmup.jsonl',fixed_ar(engine,warm))
        append(out/'warmup.jsonl',execute(engine,warm,'DFlash',256,None,request_factory=request_for,fixed=True))
        info.update(runtime=provenance(engine),compiler_after_warmup=engine._regional.summary())
        for index,record in enumerate(records):
            triplet=[]
            for role in ROLES:
                row=fixed_ar(engine,record) if role=='AR' else execute(engine,record,'DFlash',256,None,request_factory=request_for,fixed=True)
                row.update(role=role,benchmark=record['benchmark'],stop_policy='ignore-eos')
                append(out/'rows.jsonl',row);triplet.append(row);info['count']+=1
                info['peak_gpu_bytes']=max(info['peak_gpu_bytes'],row['peak_gpu_bytes'])
                if info['peak_gpu_bytes']>46*2**30:raise ValueError('GPU memory bound')
            check=validate_triplet(triplet,old[record['id']]);append(out/'triplets.jsonl',check)
            info['completed_requests']=index+1;write(out/'provenance.json',info)
            print('completed',record['id'],info['count'],flush=True)
        compiler=engine._regional.summary()
        if compiler['graph_count']!=info['compiler_after_warmup']['graph_count']:raise ValueError('New formal graph')
        cache=sum(p.stat().st_size for k in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR') for p in Path(os.environ[k]).rglob('*') if p.is_file())
        if cache>4*2**30:raise ValueError('Compiler cache bound')
        info.update(status='complete',compiler=compiler,cache_bytes=cache)
    except Exception as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        info['original_unchanged']=protected(ROOT)==before;write(out/'provenance.json',info)
        if engine is not None:engine.close()
        if not info['original_unchanged']:raise ValueError('Previous evidence changed')


if __name__=='__main__':main()
