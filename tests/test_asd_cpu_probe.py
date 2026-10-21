from copy import deepcopy

import pytest
import torch

from memorydraft.backends.asd import ASDVerifier
from scripts.run_asd_budget_probe import POLICIES
from scripts.run_asd_cpu_probe import ARMS, METHODS, AuditVerifier, timing_order
from scripts.run_asd_execution_probe import selected_records
from scripts.write_asd_cpu_report import audit, summarize


def test_frozen_schedule_keeps_policy_pairs_adjacent_and_all_arms():
    for r in selected_records():
        for repeat in range(5):
            order = timing_order(r["id"], repeat)
            assert set(order) == set(ARMS) and len(order) == 7
            assert order == timing_order(r["id"], repeat)
            for method in METHODS:
                indices = [i for i, arm in enumerate(order) if arm[0] == method]
                assert indices[1]-indices[0] == 1


def test_shadow_checks_every_actual_block_and_budget():
    policy = AuditVerifier(POLICIES["ASD"])
    tokens = torch.ones(2, dtype=torch.long)
    logits = torch.tensor([[.1, 0.], [0., 1.]])
    for _ in range(3):
        policy.decide(tokens, logits)
    assert policy.summary() == policy.shadow.summary()
    assert len(policy.events) == policy.shadow.fast_calls == 3
    assert all(e["proposal_token_ids"] == [1, 1] for e in policy.events)
    assert [s["round"] for s in policy.states] == [0]


def test_shadow_stops_on_decision_change():
    policy = AuditVerifier(POLICIES["ASD"])
    policy.shadow = ASDVerifier(POLICIES["DFlash"])
    with pytest.raises(ValueError, match="shadow changed"):
        policy.decide(torch.ones(2, dtype=torch.long), torch.tensor([[.1, 0.], [0., 1.]]))


@pytest.fixture
def matrix_fixture():
    """Synthetic matrix/timings, exclusively to test the reporting contract."""
    rows, mechanism, previous = [], [], {}
    tokens = torch.tensor([1, 2])
    logits = torch.tensor([[0., 1., 0.], [0., 0., 1.]])
    for record in selected_records():
        for method in ("DFlash", *METHODS):
            verifier = ASDVerifier(POLICIES[method], record=True)
            decision = verifier.decide(tokens, logits)
            event = dict(verifier.events[0], proposal_token_ids=[1, 2], target_top_token_ids=[1, 2])
            rd = dict(generated_before=1, proposed=2, accepted=2, advance=3,
                      strict_accepted=2, relaxed_tokens=0, regret=0.)
            old = dict(request_id=record["id"], method=method, token_ids=[0, 1, 2, 3], text="fixture",
                       prompt_sha256="fixture", prompt_tokens=1, snapshot_id="fixture", asd=verifier.summary(),
                       rounds=[rd], finish_reason="eos", benchmark=record["benchmark"], max_new_tokens=128)
            previous[record["id"], method] = old
            for impl in (("native",) if method == "DFlash" else ("tensor", "cpu")):
                for repeat in range(5):
                    latency = .99 if impl == "cpu" else 1.
                    row = dict(deepcopy(old), implementation=impl, repeat=repeat, stage="timed", latency_seconds=latency,
                               engine_latency_seconds=latency, policy_setup_seconds=0.,
                               slot=timing_order(record["id"], repeat).index((method, impl)))
                    if impl == "cpu":
                        row["cpu_execution"] = dict(fast_calls=1, fallback_calls=0)
                    rows.append(row)
            if method != "DFlash":
                capture = dict(deepcopy(old), stage="capture", instrumented_latency_seconds=1., events=[event],
                               cpu_execution=dict(fast_calls=1, fallback_calls=0), resident_replays=[dict(
                                   round=0, spent_before=0., proposals=2, expected=vars(decision),
                                   median_seconds=dict(tensor=.001, cpu=.0002),
                                   samples_seconds=dict(tensor=[.001]*30, cpu=[.0002]*30))])
                mechanism.append(capture)
    return rows, mechanism, previous


def test_report_audits_complete_matrix_and_question_level_ratios(matrix_fixture):
    rows, mechanism, previous = matrix_fixture
    a = audit(rows, mechanism, previous, [3])
    assert a == dict(timed=420, capture=36, shadow_blocks=36, resident_states=36, old_trajectory_checks=456)
    result = summarize(rows, mechanism)
    for value in result["overall"].values():
        assert value["n"] == 12 and value["ratio"] == pytest.approx(.99)
        assert value["ci95"] == pytest.approx([.99, .99])


@pytest.mark.parametrize("change", ["drop", "duplicate", "timing-leak", "spent", "fallback", "resident", "order", "setup"])
def test_report_rejects_invalid_measurement_or_ledger(matrix_fixture, change):
    rows, mechanism, previous = matrix_fixture
    if change == "drop":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = deepcopy(rows[-2])
    elif change == "timing-leak":
        mechanism[0]["latency_seconds"] = 1.
    elif change == "spent":
        mechanism[0]["resident_replays"][0]["spent_before"] = .1
    elif change == "fallback":
        mechanism[0]["cpu_execution"]["fallback_calls"] = 1
    elif change == "resident":
        mechanism[0]["resident_replays"][0]["samples_seconds"]["cpu"].pop()
    elif change == "setup":
        rows[0]["policy_setup_seconds"] = .01
    else:
        rows[0]["slot"] = (rows[0]["slot"]+1) % 7
    with pytest.raises(ValueError):
        audit(rows, mechanism, previous, [3])
