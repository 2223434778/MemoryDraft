from dataclasses import asdict, replace
import json

from fastapi.testclient import TestClient
import pytest

from memorydraft.runtime.config import RuntimeConfig
from memorydraft.runtime.contracts import RequestCancelled
from memorydraft.runtime.engine import InferenceEngine
from memorydraft.runtime.tree_policy import TreePolicySpec
from memorydraft.serving.app import create_app
from test_runtime import engine, request
from test_runtime_cost_tree import asset


def stable(value):
    if isinstance(value, dict):
        return {k:stable(v) for k,v in value.items() if not k.endswith('_seconds')}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


@pytest.fixture
def fidelity_engine(engine, tmp_path):
    engine.config = replace(engine.config, backend='dflash', draft='fixture', target_revision='fixture',
                            asd=dict(budget=8, gate=.25, cap=2), tree_policy=dict(mode='min-gain-feedback'))
    spec = asset(engine, tmp_path)
    engine.close()
    result = InferenceEngine(engine.target, engine.tokenizer, replace(engine.config, cost_tree=spec), engine.draft)
    try:
        yield result
    finally:
        result.close()


@pytest.mark.parametrize('invalid', [None, [], dict(mode='unknown'), dict(unexpected=1),
    dict(buffered_kv=1), dict(cache_block_tokens=0), dict(min_extra_tokens=True), dict(max_js=2)])
def test_policy_rejects_invalid_fields(invalid):
    with pytest.raises(ValueError):
        RuntimeConfig(backend='dflash', target='fixture', draft='fixture', asd=dict(budget=8), tree_policy=invalid)


def test_policy_requires_explicit_asd_and_is_optional():
    assert RuntimeConfig(target='fixture').tree_policy == {}
    with pytest.raises(ValueError, match='ASD'):
        RuntimeConfig(backend='dflash', target='fixture', draft='fixture', tree_policy=dict(mode='min-gain-feedback'))


def test_native_runtime_matches_research_and_keeps_strict_modes(fidelity_engine):
    from scripts.run_tree_execution import execute
    from memorydraft.evaluation.asd_paper import request_for
    record = dict(benchmark='gsm8k', id='fixture', question='w1 w2 w3', answer='#### 1')
    req = request_for(record,19)
    expected = execute(fidelity_engine, record, 'BufferedKV-ASD', 19, fidelity_engine._tree_config)
    tokens, events = [], []
    actual = fidelity_engine.generate(req, 'BudgetTree-ASD', on_tokens=tokens.extend, on_event=events.append)
    assert actual.token_ids == expected['token_ids'] == tokens
    assert stable({k:v for k,v in actual.asd.items() if k not in ('gate','cap')}) == stable(expected['asd'])
    assert actual.tree['verification']['min_extra_tokens'] == 2
    assert actual.counters['kv_allocations'] > 0
    assert events and fidelity_engine.describe()['tree_policy']['buffered_kv']
    for method in ('DFlash', 'BudgetTree'):
        strict = fidelity_engine.generate(req, method)
        assert strict.token_ids == fidelity_engine.generate(req,'AR').token_ids and not strict.approximate
        assert 'kv_allocations' not in strict.counters
        if strict.tree:
            assert 'verification' not in strict.tree


def test_request_isolation_cancel_recovery_and_policy_reload(fidelity_engine):
    reference = fidelity_engine.generate(request(), 'BudgetTree-ASD')
    tokens = []
    with pytest.raises(RequestCancelled):
        fidelity_engine.generate(request(), 'BudgetTree-ASD', on_tokens=tokens.extend, should_stop=lambda:len(tokens)>2)
    recovered = fidelity_engine.generate(request(), 'BudgetTree-ASD')
    assert recovered.token_ids == reference.token_ids and stable(recovered.asd) == stable(reference.asd)
    assert not fidelity_engine.lock.locked()
    fidelity_engine.config = replace(fidelity_engine.config, tree_policy=dict(mode='min-gain-feedback', max_js=.05))
    with pytest.raises(ValueError, match='Reload'):
        fidelity_engine.generate(request(), 'BudgetTree-ASD')


def test_cost_environment_binds_policy(fidelity_engine):
    from memorydraft.runtime.cost_tree import CostTreeSpec
    fidelity_engine.config = replace(fidelity_engine.config, tree_policy=dict(mode='min-gain-feedback', buffered_kv=False))
    with pytest.raises(ValueError, match='tree_policy'):
        CostTreeSpec(**fidelity_engine.config.cost_tree).load(fidelity_engine)


def test_http_json_sse_visualization_and_client_cannot_change_policy(fidelity_engine):
    body = {**asdict(request()), 'method':'BudgetTree-ASD'}
    with TestClient(create_app(lambda:fidelity_engine)) as client:
        assert client.get('/healthz').json()['tree_policy'] == TreePolicySpec().describe()
        result = client.post('/v1/generate',json=body).json()
        response = client.post('/v1/generate',json={**body,'stream':True,'visualize':True})
        assert response.status_code == 200
        events = [(lines[0][7:],json.loads(lines[1][6:])) for b in response.text.split('\n\n') if b
                  for lines in [b.splitlines()]]
        complete = events[-1][1]
        assert complete['token_ids'] == result['token_ids']
        assert complete['tree'] == result['tree'] and stable(complete['asd']) == stable(result['asd'])
        assert any(name=='speculation' for name,data in events)
        assert events[0][1]['tree_policy']['min_extra_tokens'] == 2
        assert client.post('/v1/generate',json={**body,'tree_policy':{}}).status_code == 422
