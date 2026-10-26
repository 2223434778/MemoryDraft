"""Bounded real-GPU acceptance of the deployed cost-tree API, not a new paper study."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient

from memorydraft.evaluation.asd_paper import conversation_record, request_for
from memorydraft.evaluation.asd_paper_extension import load_records
from memorydraft.runtime.benchmark import benchmark, provenance, write_json
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest, RequestCancelled
from memorydraft.serving.app import create_app
from scripts.run_asd_tree import execute as research_execute

ROOT=Path(__file__).resolve().parents[1]
METHODS=('AR','DFlash','ASD','BudgetTree','BudgetTree-ASD')


def main():
    out=ROOT/'results/raw/infra-tree-v04/acceptance'
    out.mkdir(parents=True,exist_ok=False)
    models=ROOT.parent/'models'
    base=json.loads((ROOT/'configs/runtime-dflash-tree-base.json').read_bytes())
    base.update(target=str(models/'qwen3-8b'),draft=str(models/'dflash-qwen3-8b'))
    write_json(out/'base.json',base)
    with (out/'calibration.log').open('x') as log:
        subprocess.run([sys.executable,'-m','memorydraft','calibrate-tree','--config',str(out/'base.json'),
            '--output',str(out/'costs.json'),'--runtime-config',str(out/'runtime.json')],
            check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    config=RuntimeConfig.load(out/'runtime.json');engine=load_engine(config)
    info=dict(status='started',purpose='deployment acceptance; quality not evaluated; observed public development inputs',
              methods=METHODS,limit=128,repeats=2)
    try:
        records=load_records(ROOT);selected=[]
        for benchmark_name,count in (('gsm8k',2),('humaneval',1),('mtbench',1),('alpaca',2)):
            subset=[r for r in records if r['benchmark']==benchmark_name][:count]
            selected.extend(conversation_record(r,1,None) if benchmark_name=='mtbench' else r for r in subset)
        requests=[request_for(r,128) for r in selected]
        warm=GenerationRequest('__warmup__','__warmup__',0,({'role':'user','content':'List the numbers one through twenty.'},),64)
        for method in METHODS:engine.generate(warm,method)
        graphs=len(engine._regional.backend.events)
        info.update(provenance=provenance(engine),requests=[asdict(r) for r in requests],warmup_graphs=graphs)
        summary=benchmark(engine,requests,METHODS,out/'benchmark',repeats=2,warmup=False,allow_approximate=True)
        timed=[json.loads(line) for line in (out/'benchmark/rows.jsonl').read_text().splitlines()]
        assert len(timed)==60 and all(r['counters']['compile_graphs_after']==graphs for r in timed)
        assert summary['methods']['BudgetTree']['strict_correct']
        equivalent=[]
        for record,request in zip(selected,requests):
            for method in ('BudgetTree','BudgetTree-ASD'):
                expected=research_execute(engine,record,method,128,engine._tree_config,request_factory=request_for)
                tokens=[];actual=engine.generate(request,method,on_tokens=tokens.extend)
                assert actual.token_ids==tokens==expected['token_ids'] and actual.text==expected['text']
                assert actual.counters['rounds']==len(expected['rounds'])
                if actual.asd:
                    assert {k:v for k,v in actual.asd.items() if k not in {'gate','cap'}}==expected['asd']
                equivalent.append(dict(request_id=request.id,method=method,exact=True,rounds=actual.counters['rounds'],
                    output_sha256=hashlib.sha256(json.dumps(actual.token_ids).encode()).hexdigest()))
        write_json(out/'research-equivalence.json',equivalent)
        for method in ('BudgetTree','BudgetTree-ASD'):
            reference=engine.generate(requests[-1],method);tokens=[]
            try:
                engine.generate(requests[-1],method,on_tokens=tokens.extend,should_stop=lambda:len(tokens)>1)
            except RequestCancelled:
                pass
            else:
                raise AssertionError('Cancellation did not trigger')
            recovered=engine.generate(requests[-1],method)
            assert recovered.token_ids==reference.token_ids and recovered.asd==reference.asd and not engine.lock.locked()
        body={**asdict(requests[-1]),'method':'BudgetTree-ASD','timeout_seconds':120}
        # Service owns this engine from here and closes it on exit.
        with TestClient(create_app(lambda:engine,event_capacity=2,queue_capacity=2)) as client:
            health=client.get('/healthz').json();assert set(METHODS)<=set(health['methods'])
            response=client.post('/v1/generate',json=body);assert response.status_code==200
            complete=response.json();assert complete['approximate'] and complete['tree']['sha256']==config.cost_tree['sha256']
            stream=client.post('/v1/generate',json={**body,'stream':True});assert stream.status_code==200
            events=[(lines[0][7:],json.loads(lines[1][6:])) for block in stream.text.split('\n\n') if block for lines in [block.splitlines()]]
            assert events[-1][0]=='complete' and events[0][1]['approximate']
            final=events[-1][1]
            assert final['token_ids']==complete['token_ids']==[t for name,data in events if name=='tokens' for t in data['token_ids']]
            assert final['asd']==complete['asd'] and final['tree']==complete['tree']
            write_json(out/'http.json',dict(health=health,json=complete,sse_complete=final,stream_exact=True,metrics=client.get('/metrics').json()))
        info.update(status='complete',timed_rows=60,research_equivalence=len(equivalent),
                    cancellation_recovery_methods=['BudgetTree','BudgetTree-ASD'],http_json_sse_exact=True,
                    formal_new_graphs=0,summary=summary)
    except Exception as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        engine.close();write_json(out/'verification.json',info)
    print(json.dumps({k:info[k] for k in ('status','timed_rows','research_equivalence','http_json_sse_exact')}))


if __name__=='__main__':main()
