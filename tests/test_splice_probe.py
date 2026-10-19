from copy import deepcopy
from pathlib import Path

import pytest

from memorydraft.evaluation.splice_probe import COUNTERS, METHODS, audit, input_digest, inputs, mechanism_gate


ROOT = Path(__file__).resolve().parents[1]


def observations():
    tasks, events = inputs(ROOT)
    rows = []
    for task in tasks:
        for method in METHODS:
            counters = dict.fromkeys(COUNTERS, 0)
            counters.update(rounds=3, draft_rounds=2, reuse_rounds=1, draft_forward_rounds=2)
            rows.append(dict(request_id=task.request.id, project_id=task.project_id,
                             group=getattr(task, "group", "development"), method=method, repeat=0,
                             token_ids=task.reference["token_ids"], prompt_sha256=task.reference["prompt_sha256"],
                             snapshot_id=task.snapshot_id, correct=True, counters=counters,
                             instrumented_latency_seconds=1., calls=[]))
    return tasks, events, rows


def test_fixed_inputs_and_instrumented_audit_do_not_publish_timing(tmp_path):
    from memorydraft.memory.store import MemoryStore

    tasks, events, rows = observations()
    assert len(tasks) == 16 and len(rows) == 96
    with MemoryStore(tmp_path / "history.db") as store:
        store.append(events)
        assert all(store.snapshot(t.request.user_id, t.request.order).snapshot_id == t.snapshot_id for t in tasks)
    assert len(input_digest(tasks, events)) == 64
    summary = audit(rows, tasks, 1, instrumented=True)
    assert "seconds" not in summary["all"]["Splice-OT"]
    assert not mechanism_gate(rows)["passed"]


@pytest.mark.parametrize("change", ["missing", "tokens", "snapshot", "attribution", "forwards", "nan"])
def test_audit_rejects_incomplete_or_false_results(change):
    tasks, _, rows = observations()
    row = next(r for r in rows if r["method"] == "Splice-OT")
    if change == "missing":
        rows.pop()
    elif change == "tokens":
        row["token_ids"] = [999]
    elif change == "snapshot":
        row["snapshot_id"] = "changed"
    elif change == "attribution":
        row["counters"]["splice_history_accepted_tokens"] = 3
    elif change == "forwards":
        row["counters"]["draft_forward_rounds"] = 0
    else:
        row["instrumented_latency_seconds"] = float("nan")
    with pytest.raises(ValueError):
        audit(rows, tasks, 1, instrumented=True)


def test_gate_requires_two_requests_and_internal_continuous_acceptance_gain():
    tasks, _, rows = observations()
    for task in tasks[:2]:
        row = next(r for r in rows if r["method"] == "Splice-OT" and r["request_id"] == task.request.id)
        row["counters"].update(rounds=2, draft_rounds=1)
        row["calls"] = [dict(proposal=dict(draft_prefix=6), oracle_accepted=20, draft_oracle_accepted=15)]
    assert mechanism_gate(rows)["passed"]
    changed = deepcopy(rows)
    next(r for r in changed if r["method"] == "Splice-OT" and r["calls"])["calls"][0]["oracle_accepted"] = 15
    assert not mechanism_gate(changed)["passed"]
