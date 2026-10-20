import random

import pytest
import torch

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.evaluation.asd_budget import boxed_answer, load_tasks, normalize_answer, request_for, score
from test_asd import scores


def test_utility_saves_budget_for_later_long_tail():
    config = ASDConfig(budget=4, gate=2, selection="utility", regret_price=2)
    utility, longest = ASDVerifier(config, record=True), ASDVerifier(ASDConfig(budget=4, gate=2))
    short = scores([3.4, -1, 30, -1, -1])
    assert utility.decide(*short).accepted == 0
    assert longest.decide(*short).accepted == 2
    tail = scores([1, -1, -1, -1, -1, -1])
    assert utility.decide(*tail).accepted == 6
    assert longest.decide(*tail).accepted == 0
    assert utility.spent == 1 and utility.summary()["shortened_blocks"] == 1
    assert utility.events[0]["feasible_accepted"] == 2


def test_choose_partial_prefix_instead_of_all_affordable_exceptions():
    policy = ASDVerifier(ASDConfig(budget=8, gate=2, selection="utility"))
    result = policy.decide(*scores([.1, -1, -1, 2, -1]))
    assert (result.accepted, result.relaxed) == (3, 1)
    assert result.regret == pytest.approx(.1)


def test_eos_is_not_a_reason_to_buy_post_eos_tokens():
    policy = ASDVerifier(ASDConfig(budget=8, gate=2, selection="utility"))
    # All proposals are token 1; the first is EOS, so the apparent long tail has no value.
    result = policy.decide(*scores([1, -1, -1, -1, -1]), stop_token_ids=[1])
    assert result.accepted == 0 and policy.spent == 0


@pytest.mark.parametrize("kwargs", [{"selection": "other"}, {"regret_price": -1}, {"regret_price": float("nan")}])
def test_bad_utility_config(kwargs):
    with pytest.raises(ValueError):
        ASDConfig(**kwargs)


def test_utility_matches_scalar_prefix_enumeration():
    rng = random.Random(187)
    for _ in range(100):
        values = [rng.choice([-1, .1, .5, 1., 3., 10.]) for _ in range(rng.randrange(2, 10))]
        config = ASDConfig(budget=2, gate=.5, cap=2, selection="utility")
        tokens, logits = scores(values)
        candidates, cost, exceptions, strict = [], 0., 0, next((i for i,v in enumerate(values) if v >= 0), len(values))
        for length in range(len(values) + 1):
            if length:
                r = values[length - 1]
                if r >= 0:
                    cost += float(logits[length - 1, 0])
                    exceptions += 1
                    if r / (len(values) - length + 1) > config.gate:
                        break
            if cost > config.budget or exceptions > config.cap:
                break
            if length >= strict:
                candidates.append((length - strict - 2 * cost, -cost, length))
        expected = max(candidates)[2]
        result = ASDVerifier(config).decide(tokens, logits)
        assert result.accepted == expected
        assert result.regret <= 2 and result.relaxed <= 2


def test_frozen_tasks_references_and_prompt_isolation():
    records = load_tasks("data/research/asd-budget/records.json")
    assert len(records) == 24
    for record in records:
        text = record.get("solution", record.get("answer"))
        assert score(record, text)["correct"]
        assert request_for(record).messages[0]["content"].endswith(record.get("question", record.get("problem")))
    assert boxed_answer(r"x=\boxed{\frac{1}{2}} then \boxed{\frac{14}{3}}") == r"\frac{14}{3}"
    assert normalize_answer(r"\left( 3, \dfrac{\pi}{2}\right)") == r"(3,\frac{\pi}{2})"
    assert not score(records[-1], r"\boxed{91}")["correct"]


def test_runner_resets_budget_per_request_and_strict_timing_uses_original_verifier():
    from types import SimpleNamespace
    from scripts.run_asd_budget_probe import BudgetHook

    seen = []

    def generate(**kwargs):
        policy = kwargs.get("verifier")
        seen.append(policy)
        if policy:
            policy.decide(*scores([.1, -1, -1]))
        return SimpleNamespace(rounds=[dict(generated_before=1)])

    hook = BudgetHook(generate, "ASD-Utility", True)
    hook()
    hook()
    assert seen[0] is not seen[1]
    assert seen[0].spent == seen[1].spent == pytest.approx(.1)
    assert hook.policy.events[0]["generated_before"] == 1
    BudgetHook(generate, "DFlash", False)()
    assert seen[-1] is None
