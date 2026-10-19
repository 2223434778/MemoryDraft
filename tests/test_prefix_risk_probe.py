from pathlib import Path

from memorydraft.evaluation.prefix_risk_probe import inputs, timing_gate
from scripts.prepare_prefix_risk import cases

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_new_projects_disjoint_from_old_calibration_and_legal_history():
    tasks, events, calibration = inputs(ROOT)
    assert len(tasks) == 12 and len({t.project_id for t in tasks}) == 6
    assert not {t.project_id for t in tasks} & set(calibration["training_projects"])
    assert calibration["training_states"] == 56
    assert all(e.order < min(t.request.order for t in tasks if t.request.user_id == e.user_id) for e in events)
    import json
    assert cases() == json.loads((ROOT / "data/research/prefix-risk/cases.json").read_text())


def test_no_candidate_change_stops_timing_without_cherry_picking_requests():
    rows = [{"method": "Risk-Splice", "counters": {"risk_switches": 0}},
            {"method": "Risk-NoConfidence", "counters": {"risk_switches": 4}}]
    assert not timing_gate(rows)["passed"]
    rows[0]["counters"]["risk_switches"] = 1
    assert timing_gate(rows) == dict(passed=True, switches=1)


def test_decision_audit_handles_live_tuple_proposals_and_rejects_changed_selection():
    from copy import deepcopy
    from types import SimpleNamespace
    import pytest
    from memorydraft.evaluation.prefix_risk_probe import audit_decisions
    from memorydraft.retrieval.prefix_risk import PrefixRiskPolicy

    policy = PrefixRiskPolicy([4., -8., 0., 0., 0., 0.], confidence=False)
    draft, reference = list(range(15)), [*range(7), *range(100, 125)]
    pool = [dict(tokens=list(range(32)), prefix_length=14, cheap=.95, position=1, accepted=7),
            dict(tokens=reference, prefix_length=7, cheap=.8, position=2, accepted=32)]
    values = [[0., 0., 0.]] * 15
    baseline, selected, curve = policy.select(pool, values)
    row = dict(method="Risk-NoConfidence", request_id="q", decisions=[dict(generated=0, candidates=pool,
        draft_tokens=draft, draft_accepted=7, confidence=values, baseline=baseline, selected=selected, survival=curve)],
        calls=[dict(generated=0, proposal=dict(tokens=tuple(reference), draft_prefix=7))],
        counters=dict(risk_evaluations=1, risk_switches=1, risk_single_boundary=0, risk_unsafe_unchanged=0))
    tasks = [SimpleNamespace(request=SimpleNamespace(id="q"), reference=dict(token_ids=reference))]
    calibration = dict(models={"Risk-NoConfidence": dict(weights=policy.weights, confidence=False)})
    audit_decisions([row], tasks, calibration)
    changed = deepcopy(row)
    changed["decisions"][0]["selected"] = 0
    with pytest.raises(ValueError, match="handoff"):
        audit_decisions([changed], tasks, calibration)
