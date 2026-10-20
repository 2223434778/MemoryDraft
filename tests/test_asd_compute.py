from copy import deepcopy

import pytest
import torch

from memorydraft.backends.asd import ASDVerifier
from scripts.run_asd_compute_probe import CONFIG, ComputeFork, selected_states, sources
from scripts.write_asd_compute_report import predict, summarize


def test_frozen_selection_is_four_distinct_blocks_per_task_without_future_labels():
    for q, row in sources().items():
        states = selected_states(row)
        assert len(states) == len({s["block"] for s in states}) == 4
        assert all(s["position"] < 512 and s["position"] + 128 < len(row["token_ids"]) for s in states)
        assert all(s["local_gain"] > 0 and s["regret_delta"] > 0 for s in states)
        altered = deepcopy(row)
        altered["quality"] = {"correct": not row["quality"]["correct"]}
        assert selected_states(altered) == states


def source_action():
    tokens = torch.ones(6, dtype=torch.long)
    logits = torch.tensor([[.25, 0.], [-1., 0.], [.25, 0.], [-1., 0.], [-1., 0.], [-1., 0.]])
    verifier = ASDVerifier(CONFIG, record=True)
    verifier.decide(tokens, logits)
    e = {**verifier.events[0], "proposal_ids": tokens.tolist(), "target_ids": logits.argmax(-1).tolist()}
    return tokens, logits, dict(events=[e]), dict(block=0, index=0, spent_before_block=0.)


@pytest.mark.parametrize("arm,accepted,charge", [("accept", 6, .5), ("reject", 0, 0.)])
def test_fork_charges_chosen_action_then_is_strict_without_resetting_old_charge(arm, accepted, charge):
    tokens, logits, source, state = source_action()
    syncs = []
    fork = ComputeFork(source, state, arm, lambda: syncs.append(1), lambda: 0.)
    assert fork.decide(tokens, logits).accepted == accepted
    assert fork.started == 0. and fork.prefix.spent == charge and fork.action["charge"] == charge
    assert fork.decide(tokens, logits).accepted == 0
    assert fork.prefix.spent == charge and fork.strict.spent == 0 and len(syncs) == 1


def test_fork_refuses_source_argmax_drift():
    tokens, logits, source, state = source_action()
    source["events"][0]["target_ids"][0] = 1
    with pytest.raises(ValueError, match="exact source"):
        ComputeFork(source, state, "accept").decide(tokens, logits)


def entries():
    return [dict(request_id=q, position=i, features=[float(i), .2, 3., 5.], regret=.2, run=3,
                 complete=True, correct=[True, True], finish=["eos", "eos"],
                 rounds=[10, 5 if i == 0 else 15], seconds=[1., .5 if i == 0 else 1.5])
            for q in ("q0", "q1", "q2", "q3") for i in (0, 1)]


def test_ridge_has_no_same_task_label_leakage():
    original = entries()
    before = predict(original)
    altered = deepcopy(original)
    for e in altered:
        if e["request_id"] == "q0":
            e["rounds"] = [1000, 1]
    after = predict(altered)
    assert all(before["q0", i] == after["q0", i] for i in (0, 1))


def test_oracle_compares_best_fixed_and_quality_filter_not_shorter_eos():
    rows = entries()
    report = summarize(rows)
    assert report["eligible"] == 8 and report["oracle_pass"]
    assert report["totals"]["Oracle"]["rounds"] == 60
    assert report["oracle_gain_over_best_fixed"] == .25
    rows[0]["complete"] = False
    rows[1]["correct"] = [False, True]
    report = summarize(rows)
    assert report["eligible"] == 6 and not report["enough"] and not report["oracle_pass"]
    assert report["totals"]["Utility"]["natural_correct"] == 7


def test_ridge_falls_back_when_other_tasks_have_too_few_labels():
    rows = entries()[:2]
    assert all(p["fallback"] and p["action"] == 0 for p in predict(rows).values())


def test_live_kv_fork_matches_realized_prefix_then_strict_target():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from memorydraft.backends.dflash_reuse import generate_reuse
    from scripts.run_asd_confidence_probe import ObservedVerifier
    from scripts.run_asd_utility_causal_probe import exception_states

    torch.set_num_threads(1)
    torch.manual_seed(119)
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

    policy = ASDVerifier(CONFIG, record=True)
    observer = ObservedVerifier(policy)
    ref = generate_reuse(target, Draft(), inputs, max_new_tokens=25, verifier=observer, on_draft=observer.on_draft)
    for e, r in zip(policy.events, ref.rounds):
        e["generated_before"] = r["generated_before"]
    source = dict(events=policy.events, token_ids=ref.sequences[0, 3:].tolist())
    state = exception_states(source)[0]
    b = state["block"]
    for arm in ("accept", "reject"):
        fork = ComputeFork(source, state, arm)
        result = generate_reuse(target, Draft(), inputs, max_new_tokens=25, verifier=fork)
        boundary = source["events"][b]["generated_before"]
        a = fork.action["accepted"]
        committed = result.sequences[:, :3 + boundary + a]
        strict = target.generate(committed, attention_mask=torch.ones_like(committed),
                                 max_new_tokens=25 - boundary - a, do_sample=False, use_model_defaults=False)
        assert result.sequences.tolist() == strict.tolist()
        assert all(r["regret"] == 0 for r in result.rounds[b + 1:])
        assert fork.started is not None
