import itertools

import pytest
import torch

from memorydraft.backends.asd import ASDConfig
from memorydraft.backends.asd_cpu import CPUASDVerifier
from memorydraft.backends.budget_tree import BudgetTreeConfig, TreeASDVerifier, TreeCosts, build_budget_tree
from memorydraft.backends.dflash_reuse import generate_reuse
from memorydraft.backends.draft_tree import DraftTree, TreeConfig
from tests.test_dflash_reuse import CheckedDraft, tiny


def costs():
    return TreeCosts((1, 100), (2, 32), ((.01, .02), (.02, .03)), .01)


def test_best_first_covers_highest_mass_ancestor_closed_paths():
    probabilities = torch.tensor([[[.8, .2], [.6, .4], [.9, .1]]])
    tree = build_budget_tree(torch.tensor([[4, 0, 0, 0]]), probabilities.log(),
                             BudgetTreeConfig(costs(), 8, 8, 2), 10)
    paths = {(): 1.}
    for depth in range(1, 4):
        for path in itertools.product(range(2), repeat=depth):
            paths[path] = float(torch.tensor([probabilities[0, i, v] for i,v in enumerate(path)]).prod())
    actual = [()]
    for i, parent in enumerate(tree.parents[1:], 1):
        actual.append((*actual[parent], int(tree.tokens[0, i])))
    expected = sorted(paths.values(), reverse=True)[:8]
    assert sorted((paths[p] for p in actual), reverse=True) == expected
    assert max(tree.depths) <= 3 and len(tree.parents) == 8


def test_cost_model_changes_node_budget_without_target_information():
    logits = torch.zeros(1, 15, 4)
    block = torch.zeros(1, 16, dtype=torch.long)
    cheap = TreeCosts((1, 100), (2, 32), ((.01, .01), (.01, .01)), .1)
    costly = TreeCosts((1, 100), (2, 32), ((.001, 5.), (.001, 5.)), .001)
    a = build_budget_tree(block, logits, BudgetTreeConfig(cheap), 50)
    b = build_budget_tree(block, logits, BudgetTreeConfig(costly), 50)
    assert len(a.parents) == 32 and len(b.parents) == 8


def test_eos_terminates_the_path_mass_surrogate_and_tree_expansion():
    block = torch.tensor([[4,0,0,0]])
    logits = torch.tensor([[[5.,0.],[5.,0.],[5.,0.]]])
    tree = build_budget_tree(block,logits,BudgetTreeConfig(costs(),8,32,1),10,stop_token_ids=(0,))
    assert tree.tokens.tolist() == [[4,0]] and tree.parents == (-1,0)
    assert tree.budget_metadata["expected_advance"] < 2


def test_tree_asd_longest_then_least_cost_charges_only_selected_path():
    tree = DraftTree(torch.tensor([[0, 1, 2, 3, 2]]), (-1, 0, 1, 0, 3), (0, 1, 2, 1, 2), 1)
    logits = torch.full((5, 5), -9.)
    logits[:, 4] = 0
    logits[0, 1], logits[0, 3] = -.4, -.2
    logits[1, 2], logits[3, 2] = 1., 1.
    verifier = TreeASDVerifier(ASDConfig(budget=8), record=True)
    path, recovery, decision = verifier.decide_tree(tree, logits, horizon=2)
    assert path == [0, 3, 4] and recovery == 4
    assert decision.regret == pytest.approx(.2) and decision.relaxed == 1
    assert verifier.spent == decision.regret


def test_linear_tree_equals_original_asd_and_honors_eos_and_remaining_budget():
    for budget, eos in itertools.product((0, .25, 8), ((), (2,))):
        tree = DraftTree(torch.tensor([[0, 1, 2, 3]]), (-1, 0, 1, 2), (0, 1, 2, 3), 4)
        logits = torch.full((4, 5), -4.)
        logits[:, 4] = 0
        logits[0, 1], logits[1, 2], logits[2, 3] = -.2, .3, -.2
        config = ASDConfig(budget=budget)
        a, b = TreeASDVerifier(config), CPUASDVerifier(config)
        for _ in range(3):
            _, _, actual = a.decide_tree(tree, logits, horizon=3, stop_token_ids=eos)
            expected = b.decide(tree.tokens[0, 1:], logits[:-1], stop_token_ids=eos)
            assert actual == expected and a.spent == b.spent


@pytest.mark.parametrize("limit", [0, 1, 2, 9, 29])
@pytest.mark.parametrize("tree_config", [TreeConfig(2, 2), BudgetTreeConfig(costs(), 2, 12, 4)])
def test_zero_budget_real_generation_stream_features_and_eos(tiny, limit, tree_config):
    published = []
    result = generate_reuse(tiny.target, CheckedDraft(tiny.features), tiny.ids, max_new_tokens=limit,
        tree_config=tree_config, verifier=TreeASDVerifier(ASDConfig(budget=0)), on_tokens=published.extend)
    assert result.sequences.tolist() == tiny.reference[:, :4+limit].tolist()
    assert published == tiny.reference[0, 4:4+limit].tolist()


def test_nonzero_tree_cache_and_followup_logits_match_real_realized_history(tiny):
    from transformers import DynamicCache
    from memorydraft.backends.draft_tree import build_tree, commit_tree_cache
    block = tiny.reference[:, 4:8].clone()
    block[0, 1] = (block[0, 1]+1) % 32
    tree = build_tree(block, torch.zeros(1, 3, 32), TreeConfig(2, 2))
    cache = DynamicCache()
    with torch.inference_mode():
        tiny.target(tiny.ids, past_key_values=cache, use_cache=True)
        output = tiny.target(tree.tokens, past_key_values=cache, use_cache=True,
            position_ids=torch.tensor([tree.depths])+4, attention_mask=tree.mask(4, torch.float32))
        verifier = TreeASDVerifier(ASDConfig(budget=8, gate=100, cap=3))
        path, recovery, decision = verifier.decide_tree(tree, output.logits[0], horizon=3)
        assert decision.relaxed > 0
        chosen = tree.tokens[:, path]
        reference = tiny.target(torch.cat([tiny.ids, chosen], 1), use_cache=True)
        torch.testing.assert_close(output.logits[:, path], reference.logits[:, 4:], atol=2e-6, rtol=2e-5)
        commit_tree_cache(cache, 4, path)
        step = tiny.target(torch.tensor([[recovery]]), past_key_values=cache, use_cache=True)
        full = tiny.target(torch.cat([tiny.ids, chosen, torch.tensor([[recovery]])], 1))
        torch.testing.assert_close(step.logits[:, -1], full.logits[:, -1], atol=2e-6, rtol=2e-5)


def test_zero_budget_rejects_different_tied_argmax_and_eos_descendants():
    tree = DraftTree(torch.tensor([[4, 1, 2]]), (-1, 0, 1), (0, 1, 2), 3)
    logits = torch.zeros(3, 5)
    path, _, _ = TreeASDVerifier(ASDConfig(budget=0)).decide_tree(tree, logits, horizon=2)
    assert path == [0]
    path, _, decision = TreeASDVerifier().decide_tree(tree, logits, horizon=2, stop_token_ids=(1,))
    assert path == [0, 1] and decision.relaxed == 1


def test_nonzero_full_generation_catches_up_features_on_actual_output(tiny):
    published = []
    class RealizedDraft(CheckedDraft):
        def forward(self, **kwargs):
            with torch.inference_mode():
                full = torch.cat([tiny.ids,tiny.ids.new_tensor([published])],1)
                output = tiny.target(full,output_hidden_states=True)
                self.expected = torch.cat([output.hidden_states[1],output.hidden_states[2]],-1)
            return super().forward(**kwargs)
    verifier = TreeASDVerifier(ASDConfig(budget=8,gate=100,cap=3))
    result = generate_reuse(tiny.target,RealizedDraft(tiny.features),tiny.ids,max_new_tokens=29,
        tree_config=BudgetTreeConfig(costs(),2,12,4),verifier=verifier,on_tokens=published.extend)
    assert result.sequences[0,4:].tolist() == published and len(published) == 29
    assert 0 < verifier.spent <= 8 and verifier.relaxed > 0


@pytest.mark.parametrize("kwargs", [dict(min_nodes=0), dict(max_nodes=100), dict(top_k=0)])
def test_invalid_builder_config(kwargs):
    with pytest.raises(ValueError):
        BudgetTreeConfig(costs(), **kwargs)


def test_nonfinite_target_scores_rejected_without_spending():
    tree = DraftTree(torch.tensor([[0, 1]]), (-1, 0), (0, 1), 2)
    verifier = TreeASDVerifier()
    with pytest.raises(ValueError):
        verifier.decide_tree(tree, torch.full((2, 4), float("nan")), horizon=1)
    assert verifier.spent == 0
