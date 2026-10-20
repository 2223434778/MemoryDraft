import pytest
import torch

from memorydraft.backends.asd import ASDConfig, ASDVerifier, suffix_values


def scores(regrets, tokens=None):
    tokens = torch.ones(len(regrets), dtype=torch.long) if tokens is None else torch.tensor(tokens)
    logits = torch.tensor([[max(r, 0.), 0. if r >= 0 else -r, -100.] for r in regrets])
    return tokens, logits


@pytest.mark.parametrize("config", [ASDConfig(budget=0), ASDConfig(gate=0), ASDConfig(cap=0)])
def test_disabled_mode_rejects_distinct_tied_tokens(config):
    verifier = ASDVerifier(config)
    decision = verifier.decide(*scores([-1, 0, -1]))
    assert decision.accepted == decision.strict_accepted == 1
    assert verifier.spent == verifier.relaxed == 0


def test_budget_persists_across_blocks_and_unaccepted_cost_is_not_charged():
    verifier = ASDVerifier(ASDConfig(budget=.75, gate=1, cap=2))
    first = verifier.decide(*scores([.5, -1, -1]))
    second = verifier.decide(*scores([.5, -1, -1]))
    assert (first.accepted, first.relaxed, second.accepted) == (3, 1, 0)
    assert verifier.spent == .5


def test_block_cap_cannot_skip_an_unaffordable_token():
    verifier = ASDVerifier(ASDConfig(budget=8, gate=1, cap=1))
    decision = verifier.decide(*scores([.25, -1, .25, -1]))
    assert (decision.accepted, decision.relaxed, decision.regret) == (2, 1, .25)


def test_actual_run_distinguishes_long_good_tail_from_merely_long_block():
    values = suffix_values(torch.tensor([True, True, False, False, True, False]), "exact-run")
    assert values.tolist() == [1, 3, 2, 1, 2, 1]
    tokens, logits = scores([.5, .5, -1, -1])
    assert ASDVerifier().decide(tokens, logits).accepted == 4
    assert ASDVerifier(ASDConfig(value="exact-run")).decide(tokens, logits).accepted == 0
    assert ASDVerifier(ASDConfig(value="exact-run")).decide(*scores([.5, -1, -1, -1])).accepted == 4


def test_eos_caps_emitted_prefix_and_charges_no_discarded_exceptions():
    verifier = ASDVerifier(ASDConfig(budget=2, gate=1, cap=2))
    decision = verifier.decide(*scores([.25, -1, .5]), stop_token_ids=[1])
    assert (decision.accepted, decision.relaxed, verifier.spent) == (1, 1, .25)


@pytest.mark.parametrize("kwargs", [{"budget": float("nan")}, {"gate": -1}, {"cap": True}, {"value": "unknown"}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ASDConfig(**kwargs)


def test_nonfinite_target_scores_fail_without_spending():
    verifier = ASDVerifier()
    tokens, logits = scores([.1])
    logits[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        verifier.decide(tokens, logits)
    assert verifier.spent == 0


@pytest.mark.parametrize("selection", ["longest", "utility", "confidence"])
def test_approximate_real_target_features_follow_realized_prefix_and_stream(selection):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from memorydraft.backends.dflash_reuse import ReuseConfig, generate_reuse
    from memorydraft.retrieval.reuse import ReuseProposal

    torch.set_num_threads(1)
    torch.manual_seed(119)
    config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=128, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "eager"
    target = Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[3, 4, 5]])
    committed = []

    class Draft(torch.nn.Module):
        block_size, mask_token_id, target_layer_ids = 4, 31, [0, 1]

        def forward(self, *, target_hidden, noise_embedding, position_ids, past_key_values, **kwargs):
            cached = past_key_values.get_seq_length()
            end = cached + target_hidden.shape[1]
            realized = torch.cat((ids, torch.tensor([committed], dtype=torch.long)), 1)
            ref = target(realized, output_hidden_states=True)
            expected = torch.cat([ref.hidden_states[1], ref.hidden_states[2]], -1)[:, cached:end]
            torch.testing.assert_close(target_hidden, expected, atol=2e-5, rtol=2e-5)
            values = noise_embedding.new_zeros((1, 1, target_hidden.shape[1] + self.block_size, 8))
            past_key_values.update(values, values, 0)
            return torch.zeros_like(noise_embedding)

    class Matcher:
        def propose(self, confirmed, max_tokens):
            return ReuseProposal(tuple([1] * min(5, max_tokens)), "test", 8, 0, 100, "test")

        def observe(self, *args):
            pass

    from memorydraft.backends.asd_confidence import ConfidenceUtilityVerifier
    policy_type = ConfidenceUtilityVerifier if selection == "confidence" else ASDVerifier
    policy = policy_type(ASDConfig(budget=100, gate=100, cap=8,
                                  selection="utility" if selection == "confidence" else selection))
    with torch.inference_mode():
        result = generate_reuse(target, Draft(), ids, max_new_tokens=25, matcher=Matcher(), verifier=policy,
                                route_sequence=("R", "D"), config=ReuseConfig(max_tokens=5, cooldown_rounds=0),
                                on_tokens=committed.extend)
        greedy = target.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=25,
                                 do_sample=False, use_model_defaults=False)
    assert committed == result.sequences[0, ids.shape[1]:].tolist()
    assert result.sequences.tolist() != greedy.tolist() and policy.relaxed > 0
    assert {r["route"] for r in result.rounds} == {"R", "D"}
