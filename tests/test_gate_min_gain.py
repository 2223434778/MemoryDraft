import math

import pytest
import torch

from memorydraft.backends.asd import ASDConfig
from memorydraft.backends.branch_drift import DriftConfig
from memorydraft.backends.draft_tree import DraftTree
from memorydraft.backends.gate_width import FeedbackWidthVerifier, MinGainFeedbackVerifier
from memorydraft.backends.budget_tree import BudgetTreeConfig
from tests.test_branch_drift import example
from tests.test_budget_tree import costs
from tests.test_runtime import engine


def fixture(*, gain=1):
    tree, logits = example()
    if gain == 2:
        tree = DraftTree(torch.tensor([[0, 1, 2, 3, 0]]), (-1, 0, 0, 2, 3), (0, 1, 1, 2, 3), 1)
        logits = torch.cat([logits, torch.tensor([[0., 4., -8., -8.]])])
    tree.width_candidates = ((0, 1, 2, 3),)*15
    tree.budget_metadata = dict(gate_penalty=1.)
    return tree, logits


def test_one_token_gain_falls_back_before_budget_and_feedback_commit():
    tree, logits = fixture()
    verifier = MinGainFeedbackVerifier(record=True)
    path, recovery, decision = verifier.decide_tree(tree, logits, horizon=3)
    assert path == [0, 1] and recovery == 3
    assert decision.accepted == decision.strict_accepted == 1
    assert decision.relaxed == 0 and verifier.spent == verifier.relaxed == 0
    assert verifier.events[-1]['path'] == path and verifier.events[-1]['regret'] == 0
    assert tree.budget_metadata['min_gain_strict_fallback'] is True
    assert tree.budget_metadata['min_gain_candidate_extra'] == 1
    # The selected strict leaf predicts rank 4; the discarded leaf predicts rank 1.
    assert verifier.width_history[-1] is True
    assert verifier.summary()['min_gain']['foregone_tokens'] == 1


def test_two_token_gain_keeps_original_path_budget_and_feedback():
    tree, logits = fixture(gain=2)
    verifier = MinGainFeedbackVerifier(record=True)
    actual = verifier.decide_tree(tree, logits, horizon=3)
    baseline_tree, baseline_logits = fixture(gain=2)
    baseline = FeedbackWidthVerifier(record=True)
    assert actual == baseline.decide_tree(baseline_tree, baseline_logits, horizon=3)
    assert actual[0] == [0, 2, 3, 4] and actual[2].relaxed == 1
    assert verifier.spent == pytest.approx(.1)
    assert verifier.width_history == baseline.width_history


@pytest.mark.parametrize('budget,stops', [(0, ()), (8, (1,)), (8, (2,))])
def test_strict_and_stop_paths_are_preserved(budget, stops):
    tree, logits = fixture()
    verifier = MinGainFeedbackVerifier(ASDConfig(budget=budget))
    original = FeedbackWidthVerifier(ASDConfig(budget=budget))
    assert verifier.decide_tree(tree, logits, horizon=3, stop_token_ids=stops) == original.decide_tree(
        tree, logits, horizon=3, stop_token_ids=stops)
    assert not tree.budget_metadata['min_gain_strict_fallback']


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_minimum_gain(value):
    with pytest.raises(ValueError, match='positive integer'):
        MinGainFeedbackVerifier(min_extra_tokens=value)


def test_runtime_dispatch_and_metadata(engine):
    from scripts.run_gate_min_gain import execute
    row = execute(engine, dict(id='min-gain-fixture', benchmark='gsm8k', question='w1 w2', answer='#### 2'),
                  'MinGainFeedback-ASD', 25, BudgetTreeConfig(costs(), 2, 12, 4))
    assert len(row['token_ids']) == 25 and row['asd']['min_gain']['min_extra_tokens'] == 2
    assert row['asd']['branch_drift']['max_token_regret'] == math.log(4)
    assert all(b['tree_budget']['min_extra_tokens'] == 2 for b in row['rounds'])
    assert all(not b['relaxed_tokens'] or b['accepted']-b['strict_accepted'] >= 2 for b in row['rounds'])
