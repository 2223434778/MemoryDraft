from copy import deepcopy
import json
from pathlib import Path

import pytest

from memorydraft.evaluation.runtime_migration import METHODS, audit_rows, frozen_inputs, input_hashes
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


@pytest.mark.parametrize("change", ["fresh_reference", "resources", "summary"])
def test_report_checks_reference_resources_and_summary_before_claiming_success(tmp_path, change):
    from scripts.write_runtime_migration_report import rebuild

    original = ROOT / "results/reuse-confirmation"
    tasks, _ = frozen_inputs(original)
    rows = fake_rows(tasks)

    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")

    write("provenance.json", {"status": "complete", "fresh_ar_correct": 8, "original_inputs_unchanged": True,
                              "original_input_sha256": input_hashes(original), "original_runtime_source": "unit-test",
                              "source": {"sha256": "unit-test-only"}, "runner_sha256": "unit-test"})
    write("summary.json", audit_rows(rows, tasks))
    with (tmp_path / "rows.jsonl").open("w") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in rows)
    fresh = [{**task.reference, "request_id": task.request.id} for task in tasks]
    with (tmp_path / "fresh-ar.jsonl").open("w") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in fresh)
    resources = [{"request_id": task.request.id, "legacy": "same", "aligned": "same"} for task in tasks]
    write("resources.json", resources)
    # These rows are simulated audit inputs, never published performance evidence.
    assert rebuild(tmp_path, original, tmp_path / "unit-report.md")["d_budget_passed"]
    if change == "fresh_reference":
        with (tmp_path / "fresh-ar.jsonl").open("a") as stream:
            stream.write(json.dumps(fresh[0]) + "\n")
    elif change == "resources":
        resources[0]["aligned"] = "different"
        write("resources.json", resources)
    else:
        write("summary.json", {})
    with pytest.raises(ValueError):
        rebuild(tmp_path, original, tmp_path / "invalid-report.md")
