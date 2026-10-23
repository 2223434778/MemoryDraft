from copy import deepcopy
from pathlib import Path
import json

import pytest

from memorydraft.evaluation.asd_joint_tree import ARMS, LegacyCheck, aggregate, audit_allocation, frozen_config
from memorydraft.evaluation.asd_tree import ARMS as ORIGINAL_ARMS, REPEATS, aggregate as original_aggregate
from scripts.write_asd_tree_report import read_rows
from tests.test_asd_tree_report import fixture

ROOT = Path(__file__).resolve().parents[1]


def joint_fixture():
    rows,refs = fixture()
    for arm in ARMS[4:]:
        for stage,repeats in REPEATS.items():
            for repeat in range(repeats):
                rows.append(dict(refs[0],stage=stage,arm=arm,repeat=repeat,rounds=[],
                                 asd=dict(spent=0,relaxed_tokens=0),latency_seconds=.9))
    return rows,refs


def test_complete_six_arm_statistics_isolate_spine_control():
    rows,refs = joint_fixture()
    result = aggregate(rows,refs)
    assert result['timed_generations']==30
    joint=result['families']['gsm8k']['JointTree-ASD']
    assert joint['natural']['BudgetTree-ASD']['ratio']==pytest.approx(.9)
    assert joint['natural']['SpineTree-ASD']['ratio']==1
    assert joint['prefix']['SpineTree-ASD']['n']==0


def test_four_arms_are_insufficient_for_second_stage():
    with pytest.raises(ValueError):aggregate(*fixture())


def test_quality_and_length_cost_are_not_merged():
    rows,refs=joint_fixture()
    for row in rows:
        if row['arm']=='JointTree-ASD':row['quality']=dict(correct=False)
    result=aggregate(rows,refs)['families']['gsm8k']['JointTree-ASD']['natural']
    assert result['SpineTree-ASD']['lost']==['a'] and result['correct']==0


def allocation_row():
    return dict(arm='JointTree-ASD',asd=dict(spent=4),rounds=[dict(proposed=15,tree_nodes=32,
        relaxed_tokens=1,regret=4,tree_budget=dict(spine_mode='remaining',horizon=15,
            spine_requested=15,spine_reserved=15,max_depth=15,regret_before=8)),
        dict(proposed=15,tree_nodes=32,relaxed_tokens=0,regret=0,tree_budget=dict(
            spine_mode='remaining',horizon=15,spine_requested=9,spine_reserved=9,max_depth=9,regret_before=4))])


def test_allocation_is_driven_by_actual_previous_charge():
    assert audit_allocation([allocation_row()])==dict(checked_blocks=2,remaining_shortened_blocks=1)


@pytest.mark.parametrize('change',['snapshot','length','reservation','ledger','overspend'])
def test_invalid_allocation_is_rejected(change):
    row=allocation_row()
    if change=='snapshot':row['rounds'][1]['tree_budget']['regret_before']=8
    if change=='length':row['rounds'][1]['tree_budget']['spine_requested']=15
    if change=='reservation':row['rounds'][1]['tree_budget']['spine_reserved']=16
    if change=='ledger':row['asd']['spent']=1
    if change=='overspend':row['rounds'][1]['regret']=5
    with pytest.raises(ValueError):audit_allocation([row])


def test_frozen_profiles_and_all_first_stage_trajectories():
    for shard in (0,1):
        config,source=frozen_config(ROOT,shard)
        assert config.spine_mode=='none' and config.max_nodes==32
        assert len(source['sha256'])==64
    check=LegacyCheck(ROOT)
    old=read_rows(ROOT/'results/asd-budget-tree','rows')
    for row in old:check(row)
    assert check.checked==960
    changed=deepcopy(old[0]);changed['token_ids']=[-1]
    with pytest.raises(ValueError):check(changed)


def test_generic_refactor_preserves_first_stage_archived_statistics():
    directory=ROOT/'results/asd-budget-tree'
    rows,refs=read_rows(directory,'rows'),read_rows(directory,'references')
    result=original_aggregate(rows,refs,arms=ORIGINAL_ARMS)
    archived=json.loads((directory/'analysis.json').read_text())
    archived.pop('resources')
    assert json.loads(json.dumps(result))==archived
