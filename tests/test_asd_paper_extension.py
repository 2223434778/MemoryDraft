from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from memorydraft.evaluation.asd_paper import BENCHMARKS,expanded_ids
from memorydraft.evaluation.asd_paper_extension import (ARMS,CONTROLS,COUNTS,aggregate,load_records,
    normalize_fixed,validate_policy)

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    records=[dict(id=b+':fixture',benchmark=b) for b in BENCHMARKS]
    refs=[];rows=[];frontier=[]
    for identity in expanded_ids(records):
        benchmark=identity.split(':')[0];quality=dict(correct=None if benchmark in ('mtbench','alpaca') else True)
        ref=dict(request_id=identity,benchmark=benchmark,token_ids=[1,2],text='answer',finish_reason='eos',
            prompt_sha256='fixed-'+identity,snapshot_id='empty',quality=quality)
        refs.append(ref)
        def row(arm,stage,repeat=0):
            return dict(deepcopy(ref),arm=arm,method=arm,stage=stage,repeat=repeat,
                token_ids=list(range(256)) if stage=='fixed' else [1,2],rounds=[],
                asd=dict(spent=0.,relaxed_tokens=0) if arm in ('ASD','BudgetTree-ASD') else None,
                latency_seconds=.9 if arm=='BudgetTree-ASD' else 1.,ttft_seconds=.1,
                stop_policy='ignore-eos' if stage=='fixed' else 'natural-eos')
        for stage,count in (('natural',2),('prefix',1)):
            for arm in ARMS:
                rows.extend(row(arm,stage,i) for i in range(count))
        for arm in ARMS:
            if arm in ('DFlash','BudgetTree'):
                rows.extend(dict(row(arm,'fixed',i),role=role) for i,role in enumerate(('before','after')))
            else:rows.append(dict(row(arm,'fixed'),role='candidate'))
        for control,config in CONTROLS.items():
            for stage in ('natural','fixed'):
                for arm in ('ASD','BudgetTree-ASD'):
                    frontier.append(dict(row(arm,stage),control=control,asd_config=asdict(config)))
    return rows,refs,records,frontier


def test_new_seven_task_selection_is_disjoint_balanced_and_frozen():
    records=load_records(ROOT)
    old=json.loads((ROOT/'data/research/asd-paper/manifest.json').read_bytes())
    previous={identity for identities in old['ids'].values() for identity in identities}
    assert len(records)==514 and len(expanded_ids(records))==530
    assert not {r['id'] for r in records}&previous
    assert {b:sum(r['benchmark']==b for r in records) for b in COUNTS}==COUNTS
    mmlu=[r for r in records if r['benchmark']=='mmlu']
    assert len({r['subject'] for r in mmlu})==57
    assert all(sum(r['subject']==s for r in mmlu)==2 for s in {r['subject'] for r in mmlu})
    mt=[r for r in records if r['benchmark']=='mtbench']
    assert len({r['category'] for r in mt})==8


def test_adjacent_strict_roles_are_averaged_without_discarding_drift():
    rows,_,_,_=fixture()
    target=next(r for r in rows if r['stage']=='fixed' and r['arm']=='DFlash' and r['role']=='after')
    target['latency_seconds']=2.
    normalized,drifts=normalize_fixed(rows)
    derived=next(r for r in normalized if r['stage']=='fixed' and r['arm']=='DFlash' and r['request_id']==target['request_id'])
    assert derived['latency_seconds']==1.5
    assert any(not d['within_3pct'] for d in drifts)
    assert target['latency_seconds']==2.


def test_complete_extension_matrix_and_budget_frontier_are_separate():
    result=aggregate(*fixture())
    assert result['raw_timed_generations']==144 and result['timed_generations']==128
    assert result['frontier_generations']==160 and result['frontier_requests']==8
    assert result['families']['gsm8k']['BudgetTree-ASD']['natural']['ASD']['ratio']==pytest.approx(.9)
    assert len(result['frontier'])==7*7*2
    assert all(v['tau_emitted'] is None for v in result['acceptance'])


@pytest.mark.parametrize('change',['fixed-role','fixed-hash','fixed-input','fixed-time','missing-main',
    'duplicate-main','missing-control','duplicate-control','control-budget','control-config','control-ledger'])
def test_incomplete_or_changed_evidence_is_rejected(change):
    rows,refs,records,frontier=fixture()
    after=next(r for r in rows if r['stage']=='fixed' and r['role']=='after')
    if change=='fixed-role':rows.remove(after)
    if change=='fixed-hash':after['token_ids'][0]=999
    if change=='fixed-input':after['prompt_sha256']='changed'
    if change=='fixed-time':after['latency_seconds']=float('nan')
    if change=='missing-main':rows.pop(0)
    if change=='duplicate-main':rows.append(deepcopy(rows[0]))
    if change=='missing-control':frontier.pop()
    if change=='duplicate-control':frontier.append(deepcopy(frontier[0]))
    if change=='control-budget':frontier[0]['asd']['spent']=99
    if change=='control-config':frontier[0]['asd_config']['gate']=99
    if change=='control-ledger':frontier[0]['asd']['spent']=1
    with pytest.raises(ValueError):aggregate(rows,refs,records,frontier)


def test_explicit_controls_reach_both_real_verifiers_and_zero_reduces_to_strict(engine,monkeypatch):
    from memorydraft.backends.asd import ASDConfig
    from memorydraft.backends.budget_tree import BudgetTreeConfig
    from scripts.run_asd_joint_tree import execute
    from scripts import run_asd_tree
    from tests.test_budget_tree import costs
    from memorydraft.evaluation.asd_paper import request_for
    record=dict(id='gsm8k:fixture',benchmark='gsm8k',question='What is 2 plus 2?',answer='#### 4')
    strict=engine.generate(request_for(record,8),'AR')
    constructed=[]
    for name in ('CPUASDVerifier','TreeASDVerifier'):
        native=getattr(run_asd_tree,name)
        def factory(config,_native=native):
            constructed.append(config)
            return _native(config)
        monkeypatch.setattr(run_asd_tree,name,factory)
    for arm in ('ASD','BudgetTree-ASD'):
        config=ASDConfig(budget=0)
        output=execute(engine,record,arm,8,BudgetTreeConfig(costs()),request_factory=request_for,asd_config=config)
        assert output['token_ids']==strict.token_ids and output['asd']['spent']==0
        assert not output['experimental_approximate']
        validate_policy(output,config)
        control=ASDConfig(budget=4,gate=.5,cap=1)
        output=execute(engine,record,arm,8,BudgetTreeConfig(costs()),request_factory=request_for,asd_config=control)
        assert constructed[-1]==control
        validate_policy(output,control)
    with pytest.raises(ValueError):
        execute(engine,record,'BudgetTree',8,BudgetTreeConfig(costs()),request_factory=request_for,asd_config=ASDConfig())


from test_runtime import engine  # noqa: E402,F401
