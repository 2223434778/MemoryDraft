"""Bounded deployment acceptance of the v0.5 high-fidelity runtime."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient

from memorydraft.evaluation.asd_paper import conversation_record, request_for
from memorydraft.runtime.benchmark import benchmark, provenance, write_json
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import RequestCancelled
from memorydraft.serving.app import create_app
from scripts.run_branch_drift_optimization import select_records
from scripts.run_tree_execution import execute as research_execute

ROOT = Path(__file__).resolve().parents[1]


def stable(value):
    if isinstance(value, dict):
        return {k:stable(v) for k,v in value.items() if not k.endswith('_seconds')}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def main():
    out = ROOT/'results/raw/infra-fidelity-v05'
    out.mkdir(parents=True, exist_ok=False)
    base = json.loads((ROOT/'configs/runtime-dflash-fidelity-base.json').read_bytes())
    base.update(target=str(ROOT.parent/'models/qwen3-8b'), draft=str(ROOT.parent/'models/dflash-qwen3-8b'))
    write_json(out/'base.json', base)
    with (out/'calibration.log').open('x') as log:
        subprocess.run([sys.executable,'-m','memorydraft','calibrate-tree','--config',str(out/'base.json'),
            '--output',str(out/'costs.json'),'--runtime-config',str(out/'runtime.json')],
            check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    config = RuntimeConfig.load(out/'runtime.json')
    records = select_records(ROOT)
    selected = []
    for task in ('gsm8k','math500','humaneval','mbpp','mmlu','mtbench','alpaca'):
        record = next(r for r in records if r['benchmark']==task)
        selected.append(conversation_record(record,1,None) if task=='mtbench' else record)
    requests = [request_for(r,128) for r in selected]
    request_path = out/'request.json'
    write_json(request_path,asdict(requests[-1]))
    # Run the independent CLI before retaining a parent engine on the GPU.
    with (out/'cli.log').open('x') as log:
        subprocess.run([sys.executable,'-m','memorydraft','infer','--config',str(out/'runtime.json'),
            '--request',str(request_path),'--method','BudgetTree-ASD','--output',str(out/'cli-answer.json')],
            check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    cli = json.loads((out/'cli-answer.json').read_bytes())
    engine = load_engine(config)
    info = dict(status='started', purpose='infra deployment acceptance, not a new quality/performance benchmark', limit=128)
    try:
        equivalent = []
        for record, request in zip(selected, requests):
            expected = research_execute(engine,record,'BufferedKV-ASD',128,engine._tree_config)
            tokens = []
            actual = engine.generate(request,'BudgetTree-ASD',on_tokens=tokens.extend)
            assert actual.token_ids == expected['token_ids'] == tokens
            assert actual.prompt_sha256 == expected['prompt_sha256'] and actual.snapshot_id == expected['snapshot_id']
            assert actual.counters['rounds'] == len(expected['rounds'])
            assert stable({k:v for k,v in actual.asd.items() if k not in ('gate','cap')}) == stable(expected['asd'])
            assert actual.tree['verification'] == engine._tree_policy.describe()
            assert actual.counters['kv_allocations'] > 0
            equivalent.append(dict(request_id=request.id, exact=True, result=asdict(actual)))
        write_json(out/'research-equivalence.json', equivalent)
        methods = ['AR','DFlash','BudgetTree','ASD','BudgetTree-ASD']
        summary = benchmark(engine,requests[:1],methods,out/'benchmark',repeats=1,warmup=False,allow_approximate=True)
        assert summary['methods']['BudgetTree']['strict_correct'] and summary['methods']['DFlash']['strict_correct']
        reference = engine.generate(requests[-1],'BudgetTree-ASD')
        assert cli['token_ids'] == reference.token_ids and stable(cli['asd']) == stable(reference.asd)
        tokens = []
        try:
            engine.generate(requests[-1],'BudgetTree-ASD',on_tokens=tokens.extend,should_stop=lambda:len(tokens)>1)
        except RequestCancelled:
            pass
        else:
            raise AssertionError('Cancellation did not trigger')
        recovered = engine.generate(requests[-1],'BudgetTree-ASD')
        assert recovered.token_ids == reference.token_ids and stable(recovered.asd)==stable(reference.asd)
        assert not engine.lock.locked()
        info.update(research_equivalence=7, cancellation_recovery=True, benchmark=summary,
                    provenance=provenance(engine), direct_reference=asdict(reference))
        body = {**asdict(requests[-1]),'method':'BudgetTree-ASD','timeout_seconds':120}
        with TestClient(create_app(lambda:engine,event_capacity=8,queue_capacity=2)) as client:
            health = client.get('/healthz').json()
            assert health['tree_policy'] == engine._tree_policy.describe()
            response = client.post('/v1/generate',json=body)
            assert response.status_code == 200
            complete = response.json()
            response = client.post('/v1/generate',json={**body,'stream':True,'visualize':True})
            assert response.status_code == 200
            events = [(lines[0][7:],json.loads(lines[1][6:])) for block in response.text.split('\n\n') if block
                      for lines in [block.splitlines()]]
            final = events[-1][1]
            assert final['token_ids'] == complete['token_ids'] == reference.token_ids
            assert stable(final['asd']) == stable(complete['asd']) == stable(reference.asd)
            assert final['tree'] == complete['tree']
            assert [t for name,data in events if name=='tokens' for t in data['token_ids']] == reference.token_ids
            assert any(name=='speculation' for name,data in events)
            assert events[0][1]['tree_policy']['buffered_kv']
            assert client.post('/v1/generate',json={**body,'tree_policy':{}}).status_code==422
            write_json(out/'http.json',dict(health=health,json=complete,sse_complete=final,
                speculation_events=sum(name=='speculation' for name,data in events),stream_exact=True))
        info.update(status='completed', http_json_sse_exact=True, cli_exact=True)
    except BaseException as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error))
        raise
    finally:
        engine.close()
        write_json(out/'verification.json',info)
    print(json.dumps(dict(status=info['status'],research_equivalence=7,http_json_sse_exact=True,cli_exact=True)),flush=True)


if __name__ == '__main__':
    main()
