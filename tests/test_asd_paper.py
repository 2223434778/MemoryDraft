from copy import deepcopy
from pathlib import Path
import json

import pytest

from memorydraft.evaluation.asd_paper import (ARMS,COUNTS,REPEATS,aggregate,conversation_record,
    expanded_ids,load_records,mbpp_interface,request_for,score,stages)

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    task=dict(id='gsm8k:fixture',benchmark='gsm8k')
    reference=dict(request_id=task['id'],benchmark='gsm8k',token_ids=[1,2],text='answer',
        finish_reason='eos',prompt_sha256='fixed',snapshot_id='empty',quality=dict(correct=True))
    rows=[]
    for stage,repeats in REPEATS.items():
        for arm in ARMS:
            for repeat in range(repeats):
                rows.append(dict(reference,stage=stage,arm=arm,repeat=repeat,
                    token_ids=list(range(256)) if stage=='fixed' else [1,2],rounds=[],
                    asd=dict(spent=0,relaxed_tokens=0) if arm=='JointTree-ASD' else None,
                    latency_seconds=.9 if arm=='JointTree-ASD' else 1.,ttft_seconds=.1))
    return rows,[reference],[task]


def test_frozen_seven_tasks_are_disjoint_from_prior_exposure_and_balanced():
    rows=load_records(ROOT)
    manifest=json.loads((ROOT/'data/research/asd-paper/manifest.json').read_text())
    excluded=set(json.loads((ROOT/'data/research/asd-paper/exclusions.json').read_text())['excluded_ids'])
    assert not {r['id'] for r in rows}&excluded
    assert len(rows)==241 and len(expanded_ids(rows))==249
    assert {b:sum(r['benchmark']==b for r in rows) for b in COUNTS}==COUNTS
    assert len({r['subject'] for r in rows if r['benchmark']=='mmlu'})==57
    assert len({r['category'] for r in rows if r['benchmark']=='mtbench'})==8
    assert manifest['sources']['alpaca.json']['commit']=='761dc5bfbdeeffa89b8bff5d038781a4055f796a'


def test_standard_code_tests_and_references_are_withheld_from_model_prompt():
    for row in load_records(ROOT):
        if row['benchmark']=='mbpp':
            prompt=request_for(row,128).messages[0]['content']
            assert row['code'] not in prompt and all(test not in prompt for test in row['test_list'])
            assert row['signature'] in prompt and row['signature']==mbpp_interface(row)
        elif row['benchmark']=='humaneval':
            prompt=request_for(row,128).messages[0]['content']
            assert row['test'] not in prompt and row['canonical_solution'] not in prompt


def test_second_turn_requires_actual_history_and_preserves_role_order():
    task=dict(id='mtbench:1',benchmark='mtbench',turns=['question1','question2'])
    with pytest.raises(ValueError):conversation_record(task,2)
    a=conversation_record(task,2,'actual prior response')
    req=request_for(a,128)
    assert [m['role'] for m in req.messages]==['user','assistant','user']
    assert req.messages[1]['content']=='actual prior response' and req.id=='mtbench:1:turn2'
    assert conversation_record(task,1)['messages']==[dict(role='user',content='question1')]


def test_mmlu_has_no_gold_answer_in_prompt_and_open_quality_is_not_fake_accuracy():
    task=dict(id='mmlu:1',benchmark='mmlu',question='Question?',choices=['a','b','c','d'],answer=2)
    assert score(task,'C\n')['correct'] and not score(task,'B')['correct']
    assert 'Expected' not in request_for(task,128).messages[0]['content']
    assert score(dict(benchmark='alpaca'),'anything')['correct'] is None
    assert stages(task)==('natural','prefix','fixed')


def test_mbpp_interface_only_exposes_required_api_and_not_helper_or_body():
    task=dict(code='def helper(x): return x*2\ndef result(x,y=3): return helper(x)+y',test_list=['assert result(1)==5'])
    interface=mbpp_interface(task)
    assert interface=='def result(x, y=3):' and 'helper' not in interface and 'return' not in interface


def test_complete_five_arm_matrix_separates_natural_and_fixed_work():
    result=aggregate(*fixture())
    assert result['timed_generations']==20
    item=result['families']['gsm8k']['JointTree-ASD']
    assert item['natural']['ASD']['ratio']==pytest.approx(.9)
    assert item['prefix']['ASD']['n']==0
    assert item['fixed']['ASD']['n']==1 and item['fixed']['tokens']==256


@pytest.mark.parametrize('change',['missing','duplicate','strict','input','time','budget','repeat','fixed'])
def test_invalid_paper_evidence_is_rejected(change):
    rows,refs,tasks=fixture()
    if change=='missing':rows.pop()
    if change=='duplicate':rows.append(deepcopy(rows[0]))
    if change=='strict':rows[0]['token_ids']=[99]
    if change=='input':rows[0]['prompt_sha256']='different'
    if change=='time':rows[0]['latency_seconds']=float('nan')
    if change=='budget':rows[-1]['asd']['spent']=9
    if change=='repeat':rows[1]['text']='changed'
    if change=='fixed':rows[-1]['token_ids']=[1]
    with pytest.raises(ValueError):aggregate(rows,refs,tasks)


def test_actual_runtime_optional_factory_and_ignore_eos_apply_to_all_five_arms(engine):
    from scripts.run_asd_joint_tree import execute
    from tests.test_budget_tree import costs
    from memorydraft.backends.budget_tree import BudgetTreeConfig
    record=dict(id='mmlu:fixture',benchmark='mmlu',question='which?',choices=['a','b','c','d'],answer=1)
    req=request_for(record,8)
    reference=engine.generate(req,'AR')
    engine.stops=[reference.token_ids[0]]
    for arm in ARMS:
        row=execute(engine,record,arm,8,BudgetTreeConfig(costs()),request_factory=request_for,fixed=True)
        assert len(row['token_ids'])==8 and row['arm']==arm
        assert engine.stops==[reference.token_ids[0]]


def test_blind_judge_prompt_does_not_expose_method_names():
    from scripts.judge_asd_paper import judge_prompt
    messages=judge_prompt('question','answer1','answer2')
    text=json.dumps(messages)
    assert all(name not in text for name in ARMS)
    assert '<answer_a>' in text and '<answer_b>' in text and 'Do not reward verbosity' in text


from test_runtime import engine  # noqa: E402,F401
