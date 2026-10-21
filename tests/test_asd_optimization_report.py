from copy import deepcopy

import pytest

from scripts.write_asd_optimization_report import draft_summary, summarize


def pairs():
    return [dict(request_id=q, position=pos, arm=arm, common_sha256="same",
                 first_round=dict(accepted=accepted), new_tokens=128, rounds=rounds)
            for q in ("q1", "q2", "q3") for pos in range(3)
            for arm, accepted, rounds in (("proposal", 3, 15), ("recovery", 6, 10))]


def test_training_gate_requires_both_degradation_measures_and_enough_common_work():
    data = pairs()
    assert draft_summary(data)["training_gate"]
    less = deepcopy(data)
    for row in less:
        if row["arm"] == "proposal":
            row["rounds"] = 9
    assert not draft_summary(less)["training_gate"]
    for row in data[:4]:
        row["new_tokens"] = 64
    result = draft_summary(data)
    assert result["equal128_pairs"] == 7 and not result["training_gate"]


def test_diagnostic_rejects_duplicate_and_changed_prefix_pairs():
    data = pairs()
    with pytest.raises(ValueError, match="Duplicate"):
        draft_summary(data + data[:1])
    data[0]["common_sha256"] = "changed"
    with pytest.raises(ValueError, match="Unpaired"):
        draft_summary(data)


def test_summary_retains_wrong_answers_and_early_eos_instead_of_inventing_equal_work():
    rows = [dict(experiment_profile="p", arm="U-fast", max_new_tokens=2048, latency_seconds=2.,
                 token_ids=[1] * 150, quality=dict(correct=False), finish_reason="eos"),
            dict(experiment_profile="p", arm="U-fast", max_new_tokens=128, latency_seconds=1.,
                 token_ids=[1] * 90, rounds=[{}] * 10, selected_lengths=[7] * 10)]
    result = summarize(rows)["p/U-fast"]
    assert result["correct"] == 0 and not result["prefix_complete"]
    assert result["prefix_tokens"] == 90 and result["prefix_rounds"] == 10


def test_bf16_extension_keeps_matched_baseline_and_quality_gate():
    from scripts.run_asd_optimization_extension import choose_lengths
    arms = ("D-fast", "ASD-fast", "U-fast", "D-verify7", "U-verify7", "D-draft8", "U-draft8", "U-recent", "U-budget")
    rows = []
    for q in range(4):
        for arm in arms:
            for limit in (2048, 128, 128):
                rows.append(dict(experiment_profile="bf16-sdpa", arm=arm, request_id=str(q),
                                 max_new_tokens=limit, latency_seconds=.8 if arm.endswith("verify7") else 1.,
                                 token_ids=[1] * 128, rounds=[{}] * 10, selected_lengths=[],
                                 quality=dict(correct=arm != "U-verify7"), finish_reason="eos"))
    choice = choose_lengths(rows)
    assert "D-verify7" in choice["arms"] and "U-verify7" not in choice["arms"]
