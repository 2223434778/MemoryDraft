from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

from memorydraft.backends.dflash_reuse import ReuseConfig, generate_reuse
from memorydraft.memory.store import MemoryEvent, Snapshot
from memorydraft.retrieval.reuse import ReuseMatcher, ReuseProposal
from memorydraft.retrieval.suffix import SuffixArrayIndex


class IntegerTokenizer:
    all_special_ids = [31]

    def encode(self, text, add_special_tokens=False):
        return [int(token) for token in text.split()]


class CheckedDraft(torch.nn.Module):
    """Check every catch-up feature against a real tiny Target's correct prefix."""
    block_size = 4
    mask_token_id = 31
    target_layer_ids = [0, 1]

    def __init__(self, expected):
        super().__init__()
        self.expected, self.calls = expected, []

    def forward(self, *, target_hidden, noise_embedding, position_ids, past_key_values, **kwargs):
        cached = past_key_values.get_seq_length()
        end = cached + target_hidden.shape[1]
        torch.testing.assert_close(target_hidden, self.expected[:, cached:end], atol=2e-5, rtol=2e-5)
        assert position_ids.shape[1] == target_hidden.shape[1] + self.block_size
        assert position_ids[0, 0].item() == cached
        self.calls.append((cached, end))
        length = target_hidden.shape[1] + self.block_size
        values = noise_embedding.new_zeros((1, 1, length, 8))
        past_key_values.update(values, values, 0)
        return torch.zeros_like(noise_embedding)


class ScriptedMatcher:
    """Gold proposals are strictly test-only, including deliberate first/partial rejection."""
    def __init__(self, reference, reject_at=None):
        self.reference, self.reject_at = reference, reject_at

    def propose(self, confirmed, max_tokens):
        tokens = list(self.reference[len(confirmed):len(confirmed) + max_tokens])
        if not tokens:
            return None
        if self.reject_at is not None and len(tokens) > self.reject_at:
            tokens[self.reject_at] = (tokens[self.reject_at] + 1) % 31
        return ReuseProposal(tuple(tokens), "test-only", 8, len(confirmed), 1000, "test")

    def observe(self, proposal, accepted, confirmed):
        pass


@pytest.fixture(scope="module")
def tiny():
    torch.set_num_threads(1)
    torch.manual_seed(119)
    config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=512, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "eager"
    target = Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[3, 4, 5, 6]])
    with torch.inference_mode():
        reference = target.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=29,
                                    do_sample=False, use_model_defaults=False, eos_token_id=None)
        output = target(reference, output_hidden_states=True)
        features = torch.cat([output.hidden_states[1], output.hidden_states[2]], dim=-1)
    return SimpleNamespace(target=target, ids=ids, reference=reference, features=features)


@pytest.mark.parametrize("reject_at", [None, 0, 2])
def test_real_target_features_survive_reuse_and_draft_switches(tiny, reject_at):
    draft = CheckedDraft(tiny.features)
    matcher = ScriptedMatcher(tiny.reference[0, tiny.ids.shape[1]:].tolist(), reject_at)
    result = generate_reuse(tiny.target, draft, tiny.ids, max_new_tokens=29, matcher=matcher,
                            config=ReuseConfig(max_tokens=4, min_match=2, cooldown_rounds=0),
                            route_sequence=("R", "R", "D", "R", "D"))
    assert result.sequences.tolist() == tiny.reference.tolist()
    assert {row["route"] for row in result.rounds} == {"R", "D"}
    assert any(end - start > 1 for start, end in draft.calls)
    assert all(row["target_cache_after"] == tiny.ids.shape[1] + row["generated_before"] - 1 + row["advance"]
               for row in result.rounds)


@pytest.mark.parametrize("budget", [0, 1, 2, 9, 29])
def test_budget_and_eos_do_not_commit_rejected_suffix(tiny, budget):
    matcher = ScriptedMatcher(tiny.reference[0, tiny.ids.shape[1]:].tolist(), 2)
    result = generate_reuse(tiny.target, CheckedDraft(tiny.features), tiny.ids, max_new_tokens=budget,
                            matcher=matcher, config=ReuseConfig(max_tokens=8, cooldown_rounds=0))
    assert result.sequences.tolist() == tiny.reference[:, :tiny.ids.shape[1] + budget].tolist()
    if budget:
        eos = int(tiny.reference[0, tiny.ids.shape[1]])
        stopped = generate_reuse(tiny.target, CheckedDraft(tiny.features), tiny.ids,
                                 max_new_tokens=budget, stop_token_ids=[eos], matcher=matcher)
        assert stopped.sequences.shape[1] == tiny.ids.shape[1] + 1


def test_pending_feature_limit_forces_real_draft_catchup(tiny):
    matcher = ScriptedMatcher(tiny.reference[0, tiny.ids.shape[1]:].tolist())
    result = generate_reuse(tiny.target, CheckedDraft(tiny.features), tiny.ids, max_new_tokens=29,
                            matcher=matcher, config=ReuseConfig(max_tokens=4, max_pending_tokens=6))
    assert result.sequences.tolist() == tiny.reference.tolist()
    assert result.buffer_flushes > 0
    assert result.feature_peak_tokens <= 6 + 5


def test_cursor_realigns_after_one_changed_token_and_stays_in_snapshot():
    snapshot = Snapshot("u", 10, (MemoryEvent("u", "old", 1, "1 2 3 4 5 6 7 8 9 10 11 12"),
                                  MemoryEvent("u", "special", 2, "20 21 31 22 23")))
    index = SuffixArrayIndex(snapshot, IntegerTokenizer(), min_match=2, max_match=5)
    matcher = ReuseMatcher(index, local_window=20)
    first = matcher.propose([1, 2], 3)
    assert first.tokens == (3, 4, 5)
    matcher.observe(first, 3, [1, 2, 3, 4, 5])
    assert matcher.propose([1, 2, 3, 4, 5, 6], 3).lookup == "cursor"
    changed = matcher.propose([1, 2, 3, 4, 5, 6], 4)
    matcher.observe(changed, 0, [1, 2, 3, 4, 5, 6])
    recovered = matcher.propose([1, 2, 3, 4, 5, 6, 19, 8, 9], 9)
    assert recovered.lookup == "local"
    assert recovered.tokens == (10, 11, 12)
    assert index.locate([20, 21], 8) is None
    assert index.locate([11, 12], 8) is None
    assert index.locate([27, 28, 29], 8) is None
    assert index.locate([1, 2], 2, max_positions=1).source == "old"


def test_invalid_controls_are_explicit(tiny):
    with pytest.raises(ValueError):
        ReuseConfig(max_pending_tokens=0)
    with pytest.raises(ValueError, match="route"):
        generate_reuse(tiny.target, CheckedDraft(tiny.features), tiny.ids,
                       max_new_tokens=5, route_sequence=("H",))


@pytest.mark.skipif(transformers.__version__ != "4.57.3", reason="Audited DFlash environment only")
def test_actual_audited_dflash_forward_and_native_control(tiny):
    from pathlib import Path
    from transformers import AutoConfig, AutoModel

    directory = Path(__file__).resolve().parents[1] / "models/dflash-qwen3-8b"
    if not directory.exists():
        pytest.skip("Audited local DFlash assets absent")
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    used = set(tiny.reference[0].tolist())
    mask = next(token for token in range(32) if token not in used)
    config.hidden_size, config.intermediate_size = 32, 64
    config.num_hidden_layers, config.num_target_layers = 1, 2
    config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 8
    config.layer_types, config.block_size, config.vocab_size = ["full_attention"], 4, 32
    config.dflash_config = dict(target_layer_ids=[0, 1], mask_token_id=mask)
    config._attn_implementation = "eager"
    draft = AutoModel.from_config(config, trust_remote_code=True).float().eval()
    native = draft.spec_generate(target=tiny.target, input_ids=tiny.ids, max_new_tokens=29,
                                 stop_token_ids=[], temperature=0.)
    empty = generate_reuse(tiny.target, draft, tiny.ids, max_new_tokens=29)
    matcher = ScriptedMatcher(tiny.reference[0, tiny.ids.shape[1]:].tolist(), 2)
    mixed = generate_reuse(tiny.target, draft, tiny.ids, max_new_tokens=29, matcher=matcher,
                           config=ReuseConfig(max_tokens=4, cooldown_rounds=0),
                           route_sequence=("R", "R", "D"))
    assert native.tolist() == empty.sequences.tolist() == mixed.sequences.tolist() == tiny.reference.tolist()
