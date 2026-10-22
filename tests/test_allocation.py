import pytest
import torch

from memorydraft.backends.allocation import ACTIONS, ForkAction, action_spec, features
from memorydraft.backends.dflash_reuse import ReuseConfig, generate_reuse
from test_dflash_reuse import tiny


class VariableDraft(torch.nn.Module):
    block_size, mask_token_id, target_layer_ids = 4, 31, [0, 1]

    def __init__(self, expected):
        super().__init__()
        self.expected, self.calls = expected, []

    def forward(self, *, target_hidden, noise_embedding, position_ids, past_key_values, **kwargs):
        cached = past_key_values.get_seq_length()
        end = cached + target_hidden.shape[1]
        torch.testing.assert_close(target_hidden, self.expected[:, cached:end], atol=2e-5, rtol=2e-5)
        n = noise_embedding.shape[1]
        assert position_ids.shape[1] == target_hidden.shape[1] + n
        assert position_ids[0, 0] == cached
        self.calls.append((cached, end, n))
        values = noise_embedding.new_zeros((1, 1, target_hidden.shape[1]+n, 8))
        past_key_values.update(values, values, 0)
        return torch.zeros_like(noise_embedding)


@pytest.mark.parametrize("sequence", [ACTIONS, ("target", "target", "native"), ("half", "native"), ("verify-half",), ("target",)])
def test_live_kv_actions_stream_and_recover_exact_greedy(tiny, sequence):
    draft = VariableDraft(tiny.features)
    commits = []
    result = generate_reuse(tiny.target, draft, tiny.ids, max_new_tokens=29,
                            round_action=lambda i, g, h: sequence[i % len(sequence)], on_tokens=commits.extend,
                            config=ReuseConfig(max_pending_tokens=8))
    assert result.sequences.tolist() == tiny.reference.tolist()
    assert commits == tiny.reference[0, tiny.ids.shape[1]:].tolist()
    assert len(draft.calls) == sum(r["draft_forward"] for r in result.rounds)
    for row in result.rounds:
        if row["allocation"] == "target":
            assert row["route"] == "A" and row["advance"] == 1 and not row["draft_forward"] and row["proposed"] == 0
        else:
            assert row["pending_after"] == row["advance"]


@pytest.mark.parametrize("budget", [1, 2, 3, 11])
def test_allocation_respects_output_limit_and_eos(tiny, budget):
    reference = tiny.reference[0, tiny.ids.shape[1]:].tolist()
    result = generate_reuse(tiny.target, VariableDraft(tiny.features), tiny.ids, max_new_tokens=budget,
                            round_action=lambda *args: "half", stop_token_ids=[reference[1]])
    assert result.sequences[0, tiny.ids.shape[1]:].tolist() == reference[:min(2, budget)]


def test_default_has_identical_trace_to_explicit_native(tiny):
    a = generate_reuse(tiny.target, VariableDraft(tiny.features), tiny.ids, max_new_tokens=29)
    b = generate_reuse(tiny.target, VariableDraft(tiny.features), tiny.ids, max_new_tokens=29, round_action=lambda *args: "native")
    assert a.sequences.tolist() == b.sequences.tolist()
    assert a.rounds == [{k: v for k, v in r.items() if k != "allocation"} for r in b.rounds]


def test_invalid_action_and_incompatible_options_fail(tiny):
    with pytest.raises(ValueError, match="Unknown allocation"):
        action_spec("future-answer", 16)
    with pytest.raises(ValueError, match="exclusive"):
        generate_reuse(tiny.target, VariableDraft(tiny.features), tiny.ids, max_new_tokens=4,
                       round_action=lambda *args: "native", verify_length=lambda n: n)


def test_fork_checks_history_and_starts_before_action():
    source = [dict(generated_before=1, accepted=0, advance=1, proposed=3, pending_before=4, draft_cache_before=0),
              dict(generated_before=2)]
    ticks = []
    fork = ForkAction(source, 1, "target", lambda: ticks.append("sync"), lambda: 42)
    assert fork(0, 1, ()) == "native" and fork.started is None
    assert fork(1, 2, (source[0],)) == "target" and ticks == ["sync"] and fork.started == 42
    with pytest.raises(ValueError, match="frontier"):
        fork(1, 3, (source[0],))
    assert features(2, (source[0],), 12)["last_advance"] == 1
