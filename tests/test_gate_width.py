import math

import pytest
import torch

from memorydraft.backends.branch_drift import (BranchDriftVerifier, DriftConfig, FeedbackTreeConfig,
    GateTreeConfig, WidthBudget, build_paired_tree)
from memorydraft.backends.budget_tree import BudgetTreeConfig, build_budget_tree
from memorydraft.backends.gate_width import FeedbackWidthVerifier
from tests.test_branch_drift import example
from tests.test_budget_tree import costs
from tests.test_runtime import engine


def test_zero_penalty_recovers_original_cost_tree_on_nontied_scores():
    scores = torch.randn(1, 15, 7, generator=torch.Generator().manual_seed(20261030))
    block = torch.zeros(1, 16, dtype=torch.long)
    ordinary = build_budget_tree(block, scores, BudgetTreeConfig(costs()), 128)
    wide = build_paired_tree(block, scores, GateTreeConfig(costs(), gate_penalty=0.), 128)
    assert ordinary.tokens.tolist() == wide.tokens.tolist()
    assert ordinary.parents == wide.parents and ordinary.depths == wide.depths


def feedback_example(verifier, *, miss=True):
    tree, logits = example(far=True)
    tree.width_candidates = ((0, 1, 2, 3) if miss else (1, 0, 2, 3),)*15
    tree.budget_metadata = dict(gate_penalty=verifier.width_penalty)
    return tree, logits


def test_feedback_uses_only_past_coverage_and_recovers_after_four_quiet_rounds():
    verifier = FeedbackWidthVerifier(drift=DriftConfig(evaluation='lazy', max_token_regret=math.log(4)))
    used = []
    for miss in (True, True, False, False, False, False):
        tree, logits = feedback_example(verifier, miss=miss)
        used.append(verifier.construction_budget().gate_penalty)
        plain = BranchDriftVerifier(drift=verifier.drift)
        assert verifier.decide_tree(tree, logits, horizon=15) == plain.decide_tree(tree, logits, horizon=15)
        assert tree.budget_metadata['width_recoverable_miss'] == miss
    assert used == [1., .5, 0., 0., 0., .5]
    assert verifier.width_penalty == 1.
    assert FeedbackWidthVerifier().width_penalty == 1.  # Request isolation.


def test_feedback_does_not_widen_for_stop_or_tokens_outside_the_candidate_pool():
    for stops in ((1,), ()):
        verifier = FeedbackWidthVerifier(drift=DriftConfig(evaluation='lazy'))
        tree, logits = feedback_example(verifier)
        if not stops:
            tree.width_candidates = ((0, 2),)*15
        verifier.decide_tree(tree, logits, horizon=15, stop_token_ids=stops)
        assert verifier.width_penalty == 1.


def test_feedback_requires_metadata_before_mutating_budget():
    tree, logits = example()
    verifier = FeedbackWidthVerifier()
    with pytest.raises(ValueError, match='rank metadata'):
        verifier.decide_tree(tree, logits, horizon=15)
    assert verifier.spent == 0 and not verifier.width_history
    with pytest.raises(ValueError, match='snapshot'):
        build_paired_tree(torch.zeros(1, 16, dtype=torch.long), torch.zeros(1, 15, 4),
                          FeedbackTreeConfig(costs()), 128)


@pytest.mark.parametrize('penalty', [0., .5, 1.])
def test_feedback_snapshot_controls_real_tree_and_remains_immutable(penalty):
    config = FeedbackTreeConfig(costs())
    budget = WidthBudget(8., 8., .25, 2, penalty)
    tree = build_paired_tree(torch.zeros(1, 16, dtype=torch.long),
        torch.randn(1, 15, 4, generator=torch.Generator().manual_seed(3)), config, 128,
        regret_budget=budget)
    assert tree.budget_metadata['gate_penalty'] == penalty
    assert len(tree.parents) <= 32 and len(tree.width_candidates) == 15
    assert config.gate_penalty == 1.


@pytest.mark.parametrize('arm', ['WidthHalf-ASD', 'WidthZero-ASD', 'WidthFeedback-ASD'])
def test_runtime_width_dispatch_and_request_local_snapshot(engine, arm):
    from scripts.run_gate_width import execute
    row = execute(engine, dict(id='width-fixture', benchmark='gsm8k', question='w1 w2', answer='#### 2'),
                  arm, 25, BudgetTreeConfig(costs(), 2, 12, 4))
    assert row['arm'] == arm and len(row['token_ids']) == 25
    config = row['asd']['branch_drift']
    assert config['evaluation'] == 'lazy' and config['max_js'] == .1
    assert config['max_token_regret'] == math.log(4)
    penalties = [b['tree_budget']['gate_penalty'] for b in row['rounds']]
    if arm == 'WidthFeedback-ASD':
        assert penalties[0] == 1. and all(p in (0., .5, 1.) for p in penalties)
        assert row['asd']['width_feedback']['blocks'] == len(row['rounds'])
    else:
        assert set(penalties) == ({.5} if arm == 'WidthHalf-ASD' else {0.})
