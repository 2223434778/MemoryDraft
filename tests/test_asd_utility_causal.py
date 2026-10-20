from copy import deepcopy

import pytest
import torch

from memorydraft.backends.asd import ASDVerifier
from scripts.run_asd_utility_causal_probe import (
    CONFIG, RepairUtilityVerifier, exception_states, selected_states, sources, validate_repair,
)


def test_frozen_states_include_later_block_exceptions_and_preserve_ledger():
    rows = sources()
    gsm = selected_states(rows["gsm8k:21"])
    assert [s["position"] for s in gsm] == [9, 14, 17, 94, 107, 110, 156, 180, 211]
    assert gsm[1]["spent_before_block"] == gsm[0]["spent_before_block"]
    assert gsm[1]["spent_before_token"] == pytest.approx(gsm[0]["spent_before_token"] + gsm[0]["regret"])
    assert [s["position"] for s in selected_states(rows["math500:7"])] == [10, 230, 756]
    broken = deepcopy(rows["gsm8k:21"])
    broken["events"][0]["spent"] += .01
    with pytest.raises(ValueError, match="ledger"):
        exception_states(broken)


def test_single_repair_keeps_earlier_charge_and_refunds_discarded_tail():
    tokens = torch.ones(6, dtype=torch.long)
    logits = torch.tensor([[.25, 0.], [-1., 0.], [.25, 0.], [-1., 0.], [-1., 0.], [-1., 0.]])
    source = ASDVerifier(CONFIG, record=True)
    source.decide(tokens, logits)
    event = {**source.events[0], "proposal_ids": tokens.tolist(), "target_ids": logits.argmax(-1).tolist()}
    state = dict(block=0, index=2, spent_before_block=0.)
    repair = RepairUtilityVerifier(state, event)
    decision = repair.decide(tokens, logits)
    assert decision.accepted == 2 and decision.relaxed == 1
    assert decision.regret == repair.spent == .25
    assert repair.relaxed == 1 and repair.blocks == 1 and repair.shortened == 1
    assert repair.foregone == 4 and repair.events[0]["spent"] == .25
    # The request ledger remains spent on the next ordinary block.
    repair.decide(tokens, logits)
    assert repair.spent == .75 and repair.relaxed == 3


def test_repair_rejects_replay_drift_instead_of_forcing_wrong_state():
    tokens = torch.ones(3, dtype=torch.long)
    logits = torch.tensor([[.1, 0.], [-1., 0.], [-1., 0.]])
    source = ASDVerifier(CONFIG, record=True)
    source.decide(tokens, logits)
    event = {**source.events[0], "proposal_ids": [2, 1, 1], "target_ids": logits.argmax(-1).tolist()}
    with pytest.raises(ValueError, match="exact source"):
        RepairUtilityVerifier(dict(block=0, index=0, spent_before_block=0.), event).decide(tokens, logits)


def test_repair_output_check_and_source_eos_exclusion():
    validate_repair([1, 2, 9, 8], [1, 2, 3, 4], dict(position=2, recovery=9))
    with pytest.raises(ValueError, match="prefix"):
        validate_repair([1, 0, 9], [1, 2, 3], dict(position=2, recovery=9))
    row = deepcopy(sources()["gsm8k:21"])
    state = selected_states(row)[0]
    row["events"][state["block"]]["proposal_eos"].append(state["index"] + 1)
    assert state["position"] not in [s["position"] for s in exception_states(row)]


def test_in_place_repair_through_real_target_kv_and_dflash_loop():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from memorydraft.backends.dflash_reuse import generate_reuse
    from scripts.run_asd_confidence_probe import ObservedVerifier

    torch.manual_seed(119)
    torch.set_num_threads(1)
    config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=128, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "eager"
    target = Qwen3ForCausalLM(config).eval()
    inputs = torch.tensor([[3, 4, 5]])

    class Draft(torch.nn.Module):
        block_size, mask_token_id, target_layer_ids = 8, 31, [0, 1]

        def forward(self, *, target_hidden, noise_embedding, past_key_values, **kwargs):
            values = noise_embedding.new_zeros((1, 1, target_hidden.shape[1] + self.block_size, 8))
            past_key_values.update(values, values, 0)
            return torch.zeros_like(noise_embedding)

    source = ASDVerifier(CONFIG, record=True)
    observed = ObservedVerifier(source)
    before = generate_reuse(target, Draft(), inputs, max_new_tokens=25, verifier=observed,
                            on_draft=observed.on_draft)
    for event, rd in zip(source.events, before.rounds):
        event["generated_before"] = rd["generated_before"]
    row = dict(token_ids=before.sequences[0, inputs.shape[1]:].tolist(), events=source.events)
    states = exception_states(row)
    assert states
    state = states[0]
    repair = RepairUtilityVerifier(state, source.events[state["block"]])
    streamed = []
    after = generate_reuse(target, Draft(), inputs, max_new_tokens=25, verifier=repair, on_tokens=streamed.extend)
    validate_repair(after.sequences[0, inputs.shape[1]:].tolist(), row["token_ids"], state)
    assert streamed == after.sequences[0, inputs.shape[1]:].tolist()
    assert repair.spent <= CONFIG.budget
