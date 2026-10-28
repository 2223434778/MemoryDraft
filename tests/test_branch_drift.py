from dataclasses import replace
import math

import pytest
import torch

from memorydraft.backends.asd import ASDConfig
from memorydraft.backends.branch_drift import (BranchDriftVerifier, DriftConfig,
    PairedTreeConfig, build_paired_tree, sibling_js)
from memorydraft.backends.budget_tree import BudgetTreeConfig, TreeASDVerifier
from memorydraft.backends.dflash_reuse import generate_reuse
from memorydraft.backends.draft_tree import DraftTree
from tests.test_budget_tree import costs
from tests.test_dflash_reuse import CheckedDraft, tiny


def example(*, far=False, reference=True):
    # Greedy sibling 1 is a leaf; approximate sibling 2 unlocks another strict token.
    tree = DraftTree(torch.tensor([[0, 1, 2, 3]]), (-1, 0, 0, 2), (0, 1, 1, 2), 1)
    logits = torch.tensor([[-8., 2., 1.9, -8.], [-8., -8., -8., 4.],
                           [-8., -8., -8., 4.], [4., -8., -8., -8.]])
    if far:
        logits[1] = torch.tensor([-8., 4., -8., -8.])
    if not reference:
        tree.tokens[0, 1] = 0
    return tree, logits


def test_full_vocabulary_js_is_symmetric_normalized_and_shift_invariant():
    logits = torch.tensor([[1000., -1000., -1000.], [-1000., 1000., -1000.],
                           [1007., -993., -993.]])
    actual = sibling_js(logits, [(0, 1), (1, 0), (0, 2)])
    assert actual == pytest.approx([1., 1., 0.], abs=1e-6)
    assert sibling_js(logits, []) == []


def test_counterfactual_gates_longer_path_and_charges_only_selected():
    tree, logits = example(far=True)
    original = TreeASDVerifier().decide_tree(tree, logits, horizon=2)
    assert original[0] == [0, 2, 3]
    guarded = BranchDriftVerifier(record=True)
    path, _, decision = guarded.decide_tree(tree, logits, horizon=2)
    assert path == [0, 1] and decision.regret == guarded.spent == 0
    assert guarded.summary()['branch_drift']['drift_blocked'] == 1
    assert guarded.events[-1]['drift_edges'][0]['reference'] == 1
    control = BranchDriftVerifier(drift=DriftConfig(max_js=1.))
    assert control.decide_tree(tree, logits, horizon=2)[0] == original[0]


def test_close_branches_keep_realized_prefix_and_original_budget():
    tree, logits = example()
    verifier = BranchDriftVerifier()
    path, _, decision = verifier.decide_tree(tree, logits, horizon=2)
    assert path == [0, 2, 3] and decision.relaxed == 1
    assert decision.regret == pytest.approx(.1)
    assert verifier.summary()['branch_drift']['selected_compared_edges'] == 1


def test_missing_actual_target_sibling_cannot_be_replaced_by_draft_top1():
    tree, logits = example(reference=False)
    verifier = BranchDriftVerifier(drift=DriftConfig(max_js=1.))
    assert verifier.decide_tree(tree, logits, horizon=2)[0] == [0]
    assert verifier.summary()['branch_drift']['missing_reference'] >= 1


@pytest.mark.parametrize('stop', [1, 2])
def test_stop_guard_blocks_both_avoiding_target_eos_and_premature_eos(stop):
    tree, logits = example()
    verifier = BranchDriftVerifier(drift=DriftConfig(max_js=1.))
    assert verifier.decide_tree(tree, logits, horizon=2, stop_token_ids=(stop,))[0] == [0, 1]
    assert verifier.summary()['branch_drift']['stop_blocked'] == 1


def test_zero_budget_has_no_approximation_or_counterfactual_work():
    tree, logits = example()
    verifier = BranchDriftVerifier(ASDConfig(budget=0))
    expected = tree.select(logits.argmax(-1).unsqueeze(0))
    assert verifier.decide_tree(tree, logits, horizon=2)[0] == expected[0]
    assert verifier.summary()['branch_drift'].get('compared_edges', 0) == 0


def test_invalid_distribution_does_not_mutate_budget():
    tree, logits = example()
    logits[1, 0] = float('-inf')
    verifier = BranchDriftVerifier()
    with pytest.raises(ValueError, match='counterfactual'):
        verifier.decide_tree(tree, logits, horizon=2)
    assert verifier.spent == 0


@pytest.mark.parametrize('count', [8, 12, 32])
def test_paired_groups_are_ancestor_closed_bounded_and_not_split(count):
    scores = torch.tensor([[[4., 3., 1., 0.]]*15])
    config = PairedTreeConfig(costs(), count, count, 4)
    tree = build_paired_tree(torch.zeros(1, 16, dtype=torch.long), scores, config, 20)
    assert len(tree.parents) <= count and max(tree.depths) <= 15
    children = {}
    for i, parent in enumerate(tree.parents[1:], 1):
        assert parent < i and tree.depths[i] == tree.depths[parent]+1
        children.setdefault(parent, set()).add(int(tree.tokens[0, i]))
    assert all((0 in ids) == (1 in ids) for ids in children.values())
    assert all(len(ids) == len(set(ids)) for ids in children.values())


def test_paired_tree_never_expands_stop_and_handles_empty_horizon():
    config = PairedTreeConfig(costs(), 2, 8, 2)
    tree = build_paired_tree(torch.tensor([[4, 0, 0]]), torch.tensor([[[4., 3., 1., 0.]]*2]), config, 10,
                             stop_token_ids=(0, 1))
    assert len(tree.parents) == 3
    empty = build_paired_tree(torch.tensor([[4]]), torch.zeros(1, 0, 4), config, 10)
    assert empty.parents == (-1,)


def test_real_tree_sibling_rows_match_their_own_sequential_context(tiny):
    from transformers import DynamicCache
    tree, _ = example()
    cache = DynamicCache()
    with torch.inference_mode():
        tiny.target(tiny.ids, past_key_values=cache, use_cache=True)
        actual = tiny.target(tree.tokens, position_ids=torch.tensor([tree.depths])+4,
            past_key_values=cache, attention_mask=tree.mask(4, torch.float32), use_cache=True)
        for path in ([0, 1], [0, 2, 3]):
            full = tiny.target(torch.cat([tiny.ids, tree.tokens[:, path]], 1))
            torch.testing.assert_close(actual.logits[:, path], full.logits[:, 4:], atol=2e-6, rtol=2e-5)
        assert math.isfinite(sibling_js(actual.logits[0], [(1, 2)])[0])


def test_generation_integrates_paired_builder_zero_budget_and_kv(tiny):
    config = PairedTreeConfig(costs(), 2, 12, 4)
    verifier = BranchDriftVerifier(ASDConfig(budget=0))
    published = []
    result = generate_reuse(tiny.target, CheckedDraft(tiny.features), tiny.ids, max_new_tokens=25,
        tree_config=config, verifier=verifier, on_tokens=published.extend)
    assert result.sequences.tolist() == tiny.reference[:, :29].tolist()
    assert published == result.sequences[0, 4:].tolist()
    assert all(r['tree_budget']['sibling_pairs'] for r in result.rounds)


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan'), True])
def test_invalid_thresholds(value):
    with pytest.raises(ValueError):
        DriftConfig(max_js=value)
