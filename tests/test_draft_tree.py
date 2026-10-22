import pytest
import torch
from transformers import DynamicCache

from memorydraft.backends.dflash_reuse import generate_reuse
from memorydraft.backends.draft_tree import TreeConfig, build_tree, commit_tree_cache
from tests.test_dflash_reuse import CheckedDraft, tiny


def test_siblings_have_same_position_but_never_attend_to_each_other():
    block = torch.tensor([[5, 2, 3, 4]])
    tree = build_tree(block, torch.zeros(1, 3, 8), TreeConfig(2, 2))
    assert tree.parents == (-1, 0, 1, 2, 0, 4, 1, 6)
    assert tree.depths == (0, 1, 2, 3, 1, 2, 2, 3)
    mask = tree.mask(7, torch.float32)[0, 0]
    assert torch.all(mask[:, :7] == 0)
    assert mask[5, 7+4] == 0 and mask[5, 7] == 0
    assert mask[5, 7+1] < -1e30 and mask[5, 7+6] < -1e30


def test_true_tree_branch_logits_and_committed_kv_match_separate_real_paths(tiny):
    block = tiny.reference[:, 4:8].clone()
    gold = int(block[0, 1])
    block[0, 1] = (gold+1) % 32
    logits = torch.full((1, 3, 32), -10.)
    logits[0, 0, gold] = 2
    tree = build_tree(block, logits, TreeConfig(1, 3))
    assert tree.tokens[0, 4] == gold
    cache = DynamicCache()
    with torch.inference_mode():
        tiny.target(tiny.ids, past_key_values=cache, use_cache=True)
        output = tiny.target(tree.tokens, past_key_values=cache, use_cache=True,
                             position_ids=torch.tensor([tree.depths])+4,
                             attention_mask=tree.mask(4, torch.float32), output_hidden_states=True)
        path, recovery, native = tree.select(output.logits.argmax(-1))
        assert path == [0, 4, 5, 6] and native == 0
        chosen = tree.tokens[:, path]
        reference = tiny.target(torch.cat([tiny.ids, chosen], 1), use_cache=True, output_hidden_states=True)
        torch.testing.assert_close(output.logits[:, path], reference.logits[:, 4:], atol=2e-6, rtol=2e-5)
        commit_tree_cache(cache, 4, path)
        assert cache.get_seq_length() == 8
        for layer in range(2):
            for actual, expected in zip(cache[layer], reference.past_key_values[layer]):
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        step = tiny.target(torch.tensor([[recovery]]), past_key_values=cache, use_cache=True)
        full = tiny.target(torch.cat([tiny.ids, chosen, torch.tensor([[recovery]])], 1))
        torch.testing.assert_close(step.logits[:, -1], full.logits[:, -1], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("config", [TreeConfig(0), TreeConfig(2, 2), TreeConfig(4, 8)])
@pytest.mark.parametrize("budget", [0, 1, 2, 9, 29])
def test_full_tree_generation_features_stream_and_budget_are_strict(tiny, config, budget):
    published = []
    result = generate_reuse(tiny.target, CheckedDraft(tiny.features), tiny.ids,
                            max_new_tokens=budget, tree_config=config, on_tokens=published.extend)
    expected = tiny.reference[:, :4+budget].tolist()
    assert result.sequences.tolist() == expected
    assert published == expected[0][4:]
    assert all(r["tree_gain"] >= 0 and r["tree_nodes"] <= 28 for r in result.rounds)
    if config.forks == 0:
        assert all(r["tree_path"] == list(range(r["advance"])) for r in result.rounds)


def test_tree_eos_on_committed_path(tiny):
    eos = int(tiny.reference[0, 8])
    tokens = tiny.reference[0, 4:].tolist()
    expected = tokens[:tokens.index(eos)+1]
    result = generate_reuse(tiny.target, CheckedDraft(tiny.features), tiny.ids, max_new_tokens=29,
                            tree_config=TreeConfig(), stop_token_ids=[eos])
    assert result.sequences[0, 4:].tolist() == expected


def test_branch_switch_features_are_valid_for_next_draft_catchup(tiny, monkeypatch):
    from memorydraft.backends import draft_tree as module
    boundary = [0]
    original = module.build_tree
    def scripted(block, logits, config):
        # Test-only gold proposals force a branch on every eligible round.
        start = 4+boundary[0]-1
        gold = tiny.reference[:, start:start+block.shape[1]].clone()
        scores = logits.clone()
        if gold.shape[1] > 1:
            token = int(gold[0, 1])
            gold[0, 1] = (token+1) % 32
            scores[0, 0, token] = 10
        return original(gold, scores, TreeConfig(1, 3))
    monkeypatch.setattr(module, "build_tree", scripted)
    draft = CheckedDraft(tiny.features)
    result = generate_reuse(tiny.target, draft, tiny.ids, max_new_tokens=29, tree_config=TreeConfig(),
                            on_draft=lambda confirmed, *args: boundary.__setitem__(0, len(confirmed)))
    assert result.sequences.tolist() == tiny.reference.tolist()
    assert any(r["tree_path"] != list(range(r["advance"])) for r in result.rounds)
    assert len(draft.calls) > 1


def test_tied_top_two_removes_native_using_native_argmax_convention():
    tree = build_tree(torch.tensor([[5, 0, 0]]), torch.zeros(1, 2, 8), TreeConfig(1, 1))
    assert tree.tokens.tolist() == [[5, 0, 0, 1]]


@pytest.mark.parametrize("kwargs", [dict(matcher=object()), dict(verifier=object()),
                                     dict(round_action=lambda *a: "native"), dict(draft_block_size=2),
                                     dict(verify_length=lambda n: 1), dict(route_sequence=("D",))])
def test_tree_rejects_ambiguous_experimental_combinations(tiny, kwargs):
    with pytest.raises(ValueError):
        generate_reuse(tiny.target, CheckedDraft(tiny.features), tiny.ids, max_new_tokens=5,
                       tree_config=TreeConfig(), **kwargs)


@pytest.mark.parametrize("args", [(-1, 2), (1, 0), (True, 2), (1, 1.5)])
def test_invalid_tree_budget(args):
    with pytest.raises(ValueError):
        TreeConfig(*args)


def test_invalid_tree_commit_and_missing_distributions():
    with pytest.raises(ValueError):
        commit_tree_cache(DynamicCache(), 0, [1])
    with pytest.raises(ValueError):
        build_tree(torch.ones(1, 4, dtype=torch.long), torch.zeros(1, 2, 8), TreeConfig())
