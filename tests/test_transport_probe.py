from copy import deepcopy
from pathlib import Path

import pytest

from memorydraft.evaluation.transport_probe import METHODS, REPEATS, audit, probe_inputs


ROOT = Path(__file__).resolve().parents[1]


def rows_for(tasks):
    return [{"request_id": t.request.id, "method": method, "repeat": repeat, "correct": True,
             "project_id": t.project_id, "budget": t.request.max_new_tokens,
             "token_ids": t.reference["token_ids"], "prompt_sha256": t.reference["prompt_sha256"],
             "snapshot_id": t.snapshot_id, "latency_seconds": 1.,
             "counters": {"rounds": 1, "draft_rounds": 1, "reuse_rounds": 0, "reuse_accepted_tokens": 0}}
            for t in tasks for method in METHODS for repeat in range(REPEATS)]


def test_probe_restores_both_old_workloads_and_requires_all_pairs():
    tasks, events = probe_inputs(ROOT)
    assert len(tasks) == 10 and len(events) == 9
    assert {t.request.id for t in tasks[-2:]} == {"copy", "explain"}
    assert events[-1].text.endswith(".\n")
    assert audit(rows_for(tasks), tasks)["all"]["OT-R"]["observations"] == 20


@pytest.mark.parametrize("change", ["tokens", "snapshot", "prompt", "budget", "missing", "duplicate", "nan"])
def test_probe_audit_rejects_changed_answers_inputs_and_accounting(change):
    tasks, _ = probe_inputs(ROOT)
    rows = deepcopy(rows_for(tasks))
    if change == "tokens":
        rows[0]["token_ids"] = [999]
    elif change == "snapshot":
        rows[0]["snapshot_id"] = "future"
    elif change == "prompt":
        rows[0]["prompt_sha256"] = "changed"
    elif change == "budget":
        rows[0]["counters"]["alignment_calls"] = 5
    elif change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["latency_seconds"] = float("nan")
    with pytest.raises(ValueError):
        audit(rows, tasks)
