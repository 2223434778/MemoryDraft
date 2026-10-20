from dataclasses import asdict
import math
import random

import pytest
import torch

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.backends.asd_confidence import ConfidenceUtilityVerifier, token_confidence
from scripts.run_asd_confidence_probe import TASK_IDS, branch_prefix, previous_references, select_states


def test_published_reference_locations_are_read_before_model_load():
    refs = previous_references()
    assert all((q, method) in refs for q in TASK_IDS for method in ("DFlash", "ASD", "ASD-Run", "ASD-Utility"))
    assert all(refs[q, "ASD"]["finish_reason"] == "eos" for q in TASK_IDS[:3])


def test_probability_normalization_shift_invariance_and_matching_mass():
    tokens = torch.tensor([1, 0])
    logits = torch.tensor([[1., 0., -1.], [3., 2., 1.]])
    observed = token_confidence(tokens, logits)
    expected = logits.softmax(-1)
    torch.testing.assert_close(observed["proposal_probability"], expected.gather(1, tokens[:, None]).squeeze(1))
    assert observed["pair_mass"][1] == observed["top_probability"][1]
    for name, values in token_confidence(tokens, logits + 100).items():
        torch.testing.assert_close(values, observed[name], rtol=1e-5, atol=1e-6)


def test_equal_regret_different_concentration_and_unchanged_raw_budget():
    tokens = torch.ones(2, dtype=torch.long)
    concentrated = torch.full((2, 20), -100.)
    concentrated[:, :2] = torch.tensor([[1.1, 0.], [-1., 0.]])
    diffuse = concentrated.clone()
    diffuse[0, 2:] = 1.
    config = ASDConfig(budget=8, gate=1, selection="utility", regret_price=2)
    assert ASDVerifier(config).decide(tokens, concentrated).accepted == 0
    high = ConfidenceUtilityVerifier(config)
    low = ConfidenceUtilityVerifier(config)
    assert high.decide(tokens, concentrated).accepted == 2
    assert low.decide(tokens, diffuse).accepted == 0
    assert math.isclose(high.spent, 1.1, abs_tol=1e-6)
    assert low.spent == 0
    # A cheap selection price never buys an exception outside the raw budget.
    assert ConfidenceUtilityVerifier(ASDConfig(budget=1, gate=1, selection="utility")).decide(
        tokens, concentrated).accepted == 0


def test_alpha_zero_matches_original_with_persistent_budget_and_eos():
    torch.manual_seed(119)
    rng = random.Random(20261020)
    for _ in range(30):
        config = ASDConfig(budget=rng.choice([0, 1, 8]), gate=1, cap=3,
                           selection="utility", regret_price=rng.choice([0, 1, 2, 3]))
        old, new = ASDVerifier(config, record=True), ConfidenceUtilityVerifier(config, alpha=0, record=True)
        for _ in range(5):
            logits = torch.randn(5, 8)
            tokens = torch.randint(0, 8, (5,))
            stops = [1] if rng.random() < .5 else []
            assert asdict(old.decide(tokens, logits, stop_token_ids=stops)) == asdict(
                new.decide(tokens, logits, stop_token_ids=stops))
            assert old.spent == new.spent and old.events == new.events


def test_confidence_utility_never_crosses_original_feasible_or_eos():
    torch.manual_seed(220)
    for _ in range(40):
        logits, tokens = torch.randn(7, 12), torch.randint(0, 12, (7,))
        longest = ASDVerifier(ASDConfig(budget=8), record=True)
        policy = ConfidenceUtilityVerifier(record=True)
        ceiling = longest.decide(tokens, logits, stop_token_ids=[2]).accepted
        decision = policy.decide(tokens, logits, stop_token_ids=[2])
        assert decision.accepted <= ceiling
        assert decision.regret == sum(policy.events[0]["regrets"][i] for i in range(decision.accepted)
                                      if policy.events[0]["mismatch"][i])


@pytest.mark.parametrize("alpha", [-1, 1, float("nan"), True])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        ConfidenceUtilityVerifier(alpha=alpha)


def test_intervention_prefix_uses_recorded_ids_and_changes_one_token_only():
    prompt, source = [7, 8], [2, 3, 4, 5]
    state = dict(position=2, candidate=4)
    assert branch_prefix(prompt, source, state, 4) == [7, 8, 2, 3, 4]
    assert branch_prefix(prompt, source, state, 6) == [7, 8, 2, 3, 6]
    with pytest.raises(ValueError):
        branch_prefix(prompt, [2, 3, 9], state, 4)


def test_state_selection_matches_regret_and_run_without_reading_outcomes():
    def event(position, mass, gap):
        return dict(generated_before=position, mismatch=[True, False], accepted=2,
                    proposal_eos=[], pair_mass=[mass, 1.], regrets=[gap, 0.],
                    proposal_ids=[1, 1], target_ids=[0, 1])
    row = dict(events=[event(1, .1, .5), event(20, .9, .6), event(40, .99, 2.)])
    chosen = select_states(row)
    assert chosen["matched"] and [s["position"] for s in chosen["states"]] == [1, 20]
    row["events"][1]["proposal_eos"] = [1]
    chosen = select_states(row)
    assert not chosen["matched"] and len(chosen["states"]) == 2
    row["events"][0]["accepted"] = 0
    row["events"][2]["generated_before"] = 512
    assert select_states(row)["states"] == []
