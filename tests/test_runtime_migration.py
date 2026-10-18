from copy import deepcopy
from pathlib import Path

import pytest

from memorydraft.evaluation.runtime_migration import METHODS, audit_rows, frozen_inputs
from memorydraft.memory.store import MemoryStore


ROOT = Path(__file__).resolve().parents[1]


def fake_rows(tasks):
    return [dict(request_id=t.request.id, method=m, repeat=r, correct=True,
                 project_id=t.project_id, turn=t.turn, budget=t.request.max_new_tokens,
                 token_ids=t.reference["token_ids"], prompt_sha256=t.reference["prompt_sha256"],
                 snapshot_id=t.snapshot_id, latency_seconds=1.,
                 counters={"rounds": 1, "draft_rounds": 1, "reuse_rounds": 0, "reuse_accepted_tokens": 0})
            for t in tasks for m in METHODS for r in range(3)]


def test_original_legal_snapshots_are_restored_without_current_answer(tmp_path):
    tasks, events = frozen_inputs(ROOT / "results/reuse-confirmation")
    assert len(tasks) == 8 and len(events) == 8
    with MemoryStore(tmp_path / "history.db") as store:
        store.append(events)
        for task in tasks:
            snapshot = store.snapshot(task.request.user_id, task.request.order)
            assert snapshot.snapshot_id == task.snapshot_id
            assert len(snapshot.records) == 1
            assert all(r.order < task.request.order and r.user_id == task.request.user_id for r in snapshot.records)
            if task.turn == 1:
                assert snapshot.records[0].text == tasks[tasks.index(task) - 1].reference["text"]
            assert snapshot.records[0].text != task.reference["text"]
    summary = audit_rows(fake_rows(tasks), tasks)
    assert summary["rows"] == 120
    assert summary["projects"] == 4


@pytest.mark.parametrize("change", ["tokens", "snapshot", "prompt", "duplicate", "missing", "counters"])
def test_migration_audit_rejects_corrupted_or_unpaired_evidence(change):
    tasks, _ = frozen_inputs(ROOT / "results/reuse-confirmation")
    rows = deepcopy(fake_rows(tasks))
    if change == "tokens":
        rows[0]["token_ids"] = [999]
    elif change == "snapshot":
        rows[0]["snapshot_id"] = "future"
    elif change == "prompt":
        rows[0]["prompt_sha256"] = "changed-target-input"
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "missing":
        rows.pop()
    else:
        rows[0]["counters"]["rounds"] = 2
    with pytest.raises(ValueError):
        audit_rows(rows, tasks)
