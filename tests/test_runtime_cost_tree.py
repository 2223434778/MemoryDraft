from dataclasses import asdict, replace
import hashlib
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from memorydraft.backends.asd import ASDConfig
from memorydraft.runtime.benchmark import benchmark, check_regression
from memorydraft.runtime.config import RuntimeConfig
from memorydraft.runtime.contracts import RequestCancelled
from memorydraft.runtime.cost_tree import CostTreeSpec, cost_environment
from memorydraft.runtime.engine import InferenceEngine
from memorydraft.runtime.tree_calibration import calibrate_tree
from memorydraft.serving.app import create_app
from test_runtime import engine, request


def asset(engine, directory, *, modify=None):
    data = dict(schema=1, environment=cost_environment(engine), costs=dict(contexts=[1, 100], nodes=[2, 32],
        seconds=[[.01, .03], [.01, .03]], draft_seconds=.02, auxiliary_seconds=0.))
    if modify:
        modify(data)
    path = directory/"costs.json"
    raw = (json.dumps(data)+"\n").encode();path.write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), min_nodes=2, max_nodes=12, top_k=4)


@pytest.fixture
def tree_engine(engine, tmp_path):
    engine.config = replace(engine.config, backend="dflash", draft="fixture", target_revision="fixture",
                            asd=dict(budget=8, gate=.25, cap=2))
    spec = asset(engine, tmp_path);engine.close()
    result = InferenceEngine(engine.target, engine.tokenizer, replace(engine.config, cost_tree=spec), engine.draft)
    try:
        yield result
    finally:
        result.close()


def test_cost_tree_is_explicit_and_requires_dflash():
    assert not RuntimeConfig(target="fixture").cost_tree
    for invalid in (None, [], dict(path="missing"), dict(path="file", sha256="a"*64, top_k=True),
                    dict(path="file", sha256="a"*64, unexpected=1)):
        with pytest.raises(ValueError):
            RuntimeConfig(backend="dflash", target="fixture", draft="fixture", cost_tree=invalid)
    with pytest.raises(ValueError, match="dflash"):
        RuntimeConfig(target="fixture", cost_tree=dict(path="file", sha256="a"*64))


def test_cost_paths_resolve_and_configuration_round_trips(tree_engine, tmp_path):
    value = asdict(tree_engine.config);value['cost_tree']['path'] = 'costs.json'
    path = tmp_path/'runtime.json';path.write_text(json.dumps(value))
    loaded = RuntimeConfig.load(path)
    assert loaded.cost_tree['path'] == str(tmp_path/'costs.json')
    assert RuntimeConfig(**asdict(loaded)) == loaded


@pytest.mark.parametrize('bad', ['hash', 'schema', 'environment', 'nan', 'bounds', 'dimensions'])
def test_invalid_cost_assets_fail_before_inference(engine, tmp_path, bad):
    engine.config = replace(engine.config, backend='dflash', draft='fixture', target_revision='fixture')
    def modify(data):
        if bad=='schema':data['schema']=True
        elif bad=='environment':data['environment']['gpu']='other-gpu'
        elif bad=='nan':data['costs']['seconds'][0][0]=float('nan')
        elif bad=='dimensions':data['costs']['nodes'][0]=True
    spec = asset(engine,tmp_path,modify=modify)
    if bad=='hash':spec['sha256']='0'*64
    if bad=='bounds':spec['max_nodes']=64
    engine.close()
    with patch.object(engine.target,'forward',side_effect=AssertionError('must not execute Target')):
        with pytest.raises(ValueError):
            InferenceEngine(engine.target,engine.tokenizer,replace(engine.config,cost_tree=spec),engine.draft)


@pytest.mark.parametrize('method',['BudgetTree','BudgetTree-ASD'])
@pytest.mark.parametrize('limit',[1,2,19])
def test_tree_stream_output_metadata_and_strict_identity(tree_engine,method,limit):
    req=request(max_new_tokens=limit);commits=[]
    result=tree_engine.generate(req,method,on_tokens=commits.extend)
    assert commits==result.token_ids and result.tree['sha256']==tree_engine.config.cost_tree['sha256']
    assert result.counters['tree_built_nodes']>=result.counters['tree_verification_nodes']
    assert result.counters['tree_rounds']==result.counters['rounds']
    if method=='BudgetTree':
        assert not result.approximate and result.asd is None
        assert result.token_ids==tree_engine.generate(req,'AR').token_ids
    else:
        assert result.approximate and 0<=result.asd['spent']<=8


def test_tree_policy_is_fresh_cancel_recovers_and_original_methods_stay_strict(tree_engine,monkeypatch):
    from memorydraft.runtime import engine as module
    native=module.generate_reuse;policies=[]
    def inspect(*args,**kwargs):
        verifier=kwargs.get('verifier')
        if verifier is not None:
            assert verifier.spent==verifier.blocks==verifier.relaxed==0
            policies.append(verifier)
        return native(*args,**kwargs)
    monkeypatch.setattr(module,'generate_reuse',inspect)
    baseline=tree_engine.generate(request(),'BudgetTree-ASD');commits=[]
    with pytest.raises(RequestCancelled):
        tree_engine.generate(request(),'BudgetTree-ASD',on_tokens=commits.extend,should_stop=lambda:len(commits)>3)
    recovered=tree_engine.generate(request(),'BudgetTree-ASD')
    assert recovered.token_ids==baseline.token_ids and recovered.asd==baseline.asd
    assert len(policies)==len({id(p) for p in policies})==3
    assert not tree_engine.lock.locked()
    ar=tree_engine.generate(request(),'AR');linear=tree_engine.generate(request(),'DFlash')
    assert ar.token_ids==linear.token_ids and linear.tree is None and not linear.approximate


def test_tree_eos_and_mutation_require_reload(tree_engine):
    reference=tree_engine.generate(request(),'BudgetTree-ASD')
    tree_engine.stops=[reference.token_ids[0]];tokens=[]
    stopped=tree_engine.generate(request(),'BudgetTree-ASD',on_tokens=tokens.extend)
    assert tokens==stopped.token_ids==reference.token_ids[:1] and stopped.finish_reason=='eos'
    tree_engine.config=replace(tree_engine.config,cost_tree={**tree_engine.config.cost_tree,'max_nodes':8})
    with pytest.raises(ValueError,match='Reload'):
        tree_engine.generate(request(),'BudgetTree')
    assert not tree_engine.lock.locked()


def test_runtime_matches_research_tree_without_patching_production(tree_engine):
    from scripts.run_asd_tree import execute
    record=dict(benchmark='gsm8k',id='fixture',question='unused',answer='#### 1')
    for method in ('BudgetTree','BudgetTree-ASD'):
        expected=execute(tree_engine,record,method,19,tree_engine._tree_config,request_factory=lambda record,limit:request(max_new_tokens=limit))
        actual=tree_engine.generate(request(),method)
        assert actual.token_ids==expected['token_ids'] and actual.text==expected['text']
        assert actual.counters['rounds']==len(expected['rounds'])
        if actual.asd:
            assert {k:v for k,v in actual.asd.items() if k not in {'gate','cap'}}==expected['asd']


def test_http_tree_json_and_sse_use_the_real_runtime(tree_engine):
    body={**asdict(request()),'method':'BudgetTree-ASD'}
    with TestClient(create_app(lambda:tree_engine)) as client:
        health=client.get('/healthz').json()
        assert 'BudgetTree-ASD' in health['approximate_methods'] and 'BudgetTree' in health['methods']
        first=client.post('/v1/generate',json=body)
        assert first.status_code==200 and first.json()['tree']['sha256']==tree_engine._tree_spec.sha256
        response=client.post('/v1/generate',json={**body,'stream':True})
        events=[(lines[0][7:],json.loads(lines[1][6:])) for b in response.text.split('\n\n') if b for lines in [b.splitlines()]]
        complete=events[-1][1]
        assert complete['tree']==first.json()['tree'] and complete['asd']==first.json()['asd']
        assert [t for name,data in events if name=='tokens' for t in data['token_ids']]==complete['token_ids']==first.json()['token_ids']
        assert events[0][1]['approximate']


def test_approximate_benchmark_does_not_claim_token_identity_or_task_accuracy(tree_engine,tmp_path):
    with pytest.raises(ValueError,match='approximate'):
        benchmark(tree_engine,[request()],['BudgetTree-ASD'],tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()
    result=benchmark(tree_engine,[request()],['AR','DFlash','BudgetTree','ASD','BudgetTree-ASD'],
        tmp_path/'comparison',repeats=1,allow_approximate=True)
    assert result['quality_evaluated'] is False
    assert result['methods']['BudgetTree-ASD']['strict_correct'] is None
    assert result['methods']['BudgetTree']['strict_correct']
    rows=[json.loads(line) for line in (tmp_path/'comparison/rows.jsonl').read_text().splitlines()]
    assert all(r['correct'] is None for r in rows if r['approximate'])
    with pytest.raises(ValueError,match='strictly'):
        check_regression(tmp_path/'comparison',tmp_path/'comparison')


def test_synthetic_calibration_is_loadable_preserves_engine_and_refuses_overwrite(engine,tmp_path):
    engine.config=replace(engine.config,backend='dflash',draft='fixture',target_revision='fixture')
    path=tmp_path/'measured.json'
    result=calibrate_tree(engine,path,contexts=(16,32),nodes=(2,8),repeats=2)
    spec=CostTreeSpec(str(path),result['sha256'],min_nodes=2,max_nodes=8)
    config=spec.load(engine)
    assert config.costs.contexts==(16,32) and not engine.lock.locked()
    assert engine.generate(request(),'DFlash').token_ids==engine.generate(request(),'AR').token_ids
    with pytest.raises(FileExistsError):calibrate_tree(engine,path)


def test_cli_dispatches_calibration_and_writes_a_bound_config(engine,tmp_path,monkeypatch):
    from memorydraft.cli import main
    from memorydraft.runtime import cli
    engine.config=replace(engine.config,backend='dflash',draft='fixture',target_revision='fixture')
    base=tmp_path/'base.json';base.write_text(json.dumps(asdict(engine.config)))
    monkeypatch.setattr(cli,'load_engine',lambda config:engine)
    import memorydraft.runtime.tree_calibration as module
    native=module.calibrate_tree
    monkeypatch.setattr(module,'calibrate_tree',lambda engine,output:native(engine,output,contexts=(16,32),nodes=(8,32),repeats=2))
    config_path=tmp_path/'bound.json';costs_path=tmp_path/'asset.json'
    main(['calibrate-tree','--config',str(base),'--output',str(costs_path),'--runtime-config',str(config_path)])
    config=RuntimeConfig.load(config_path)
    assert config.cost_tree['sha256']==hashlib.sha256(costs_path.read_bytes()).hexdigest()
    assert config.cost_tree['path']==str(costs_path.resolve())
    assert engine._closed


def test_approximate_benchmark_preserves_bad_budget_evidence(tree_engine,tmp_path,monkeypatch):
    native=tree_engine.generate
    def broken(req,method,**kwargs):
        result=native(req,method,**kwargs)
        if method=='BudgetTree-ASD':result.asd['spent']=9
        return result
    monkeypatch.setattr(tree_engine,'generate',broken)
    with pytest.raises(ValueError,match='budget'):
        benchmark(tree_engine,[request()],['BudgetTree-ASD'],tmp_path/'failed',repeats=1,
                  warmup=False,allow_approximate=True)
    assert (tmp_path/'failed/failure.json').exists()
    assert not (tmp_path/'failed/summary.json').exists()
