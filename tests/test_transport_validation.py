from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from memorydraft.evaluation.runtime_migration import file_hash, input_hashes
from memorydraft.evaluation.transport_probe import METHODS, REPEATS
from memorydraft.evaluation.transport_validation import (GROUPS, attach_references, audit_rows, comparisons,
                                                       contract_check, load_inputs)
from memorydraft.memory.store import MemoryStore
from scripts.make_transport_validation import authored_projects


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data/research/ot-validation/cases.json"


def simulated_evidence():
    """Unit-only rows test rejection logic; never used as model/performance evidence."""
    tasks, _ = load_inputs(CASES)
    references = [{"request_id": t.request.id, "method": "AR", "token_ids": [2, 3],
                   "text": t.expected_text, "prompt_sha256": "a" * 64, "prompt_tokens": 50,
                   "snapshot_id": t.snapshot_id, "latency_seconds": 2., "stop_reason": "eos",
                   "contract": contract_check(t.expected_text, t.expected_text)} for t in tasks]
    tasks = attach_references(tasks, references)
    rows = [{"request_id": t.request.id, "method": m, "repeat": repeat, "correct": True,
             "project_id": t.project_id, "group": t.group, "budget": t.request.max_new_tokens,
             "token_ids": t.reference["token_ids"], "prompt_sha256": t.reference["prompt_sha256"],
             "snapshot_id": t.snapshot_id, "latency_seconds": 1., "stop_reason": "eos",
             "contract": t.reference["contract"],
             "counters": {"rounds": 1, "draft_rounds": 1, "reuse_rounds": 0, "reuse_accepted_tokens": 0}}
            for t in tasks for m in METHODS for repeat in range(REPEATS)]
    return tasks, references, rows


def test_new_cases_are_frozen_distinct_and_match_real_legal_snapshots(tmp_path):
    assert json.loads(CASES.read_text(encoding="utf-8")) == json.loads(json.dumps(authored_projects()))
    tasks, events = load_inputs(CASES)
    assert len(tasks) == 24 and len(events) == 13
    assert {t.group for t in tasks} == set(GROUPS)
    assert len({t.request.user_id for t in tasks}) == 12
    with MemoryStore(tmp_path / "history.db") as store:
        store.append(events)
        for task in tasks:
            snapshot = store.snapshot(task.request.user_id, task.request.order)
            assert snapshot.snapshot_id == task.snapshot_id
            assert all(e.order < task.request.order and e.user_id == task.request.user_id for e in snapshot.records)
            assert all(e.text != task.expected_text for e in snapshot.records)
    assert contract_check("x\n", "x") == {"artifact_correct": True, "exact_text": False}
    assert not contract_check("x \n", "x")["artifact_correct"]


@pytest.mark.parametrize("change", ["future", "cross_user", "changed_contract", "duplicate_request"])
def test_case_loader_rejects_changed_legal_history_and_contract(tmp_path, change):
    value = deepcopy(authored_projects())
    if change == "future":
        value["projects"][0]["history"][0]["order"] = 12
    elif change == "cross_user":
        value["projects"][0]["history"][0]["user_id"] = "other"
    elif change == "changed_contract":
        value["projects"][0]["requests"][0]["expected_text"] = "hidden answer"
    else:
        value["projects"][0]["requests"][1]["request"]["id"] = value["projects"][0]["requests"][0]["request"]["id"]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_inputs(path)


@pytest.mark.parametrize("change", ["tokens", "prompt", "snapshot", "group", "contract", "missing", "duplicate", "nan", "calls", "pool"])
def test_validation_audit_rejects_wrong_results_or_unpaired_costs(change):
    tasks, _, rows = simulated_evidence()
    rows = deepcopy(rows)
    if change in {"tokens", "prompt", "snapshot", "group", "contract", "nan"}:
        key, value = {"tokens": ("token_ids", [4]), "prompt": ("prompt_sha256", "b" * 64),
                      "snapshot": ("snapshot_id", "other"), "group": ("group", "other"),
                      "contract": ("contract", {}), "nan": ("latency_seconds", float("nan"))}[change]
        rows[0][key] = value
    elif change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    else:
        rows[3]["counters"].update(alignment_calls=5 if change == "calls" else 1, alignment_candidates=9)
    with pytest.raises(ValueError):
        audit_rows(rows, tasks)


def test_project_comparison_gate_requires_spread_and_task_success():
    tasks, _, rows = simulated_evidence()
    for row in rows:
        if row["group"] == "local-edit" and row["method"] == "OT-R":
            row["latency_seconds"] = .9
    summary = audit_rows(rows, tasks)
    comparison = comparisons(summary, tasks)
    assert comparison["local-edit"]["engineering_gate"]
    assert comparison["local-edit"]["Suffix-R"]["projects_faster"] == 4
    assert not comparison["control"]["engineering_gate"]
    failed = [replace(t, reference={**t.reference, "contract": {"artifact_correct": False, "exact_text": False}})
              if t.request.id == tasks[0].request.id else t for t in tasks]
    for row in rows:
        if row["request_id"] == tasks[0].request.id:
            row["contract"] = failed[0].reference["contract"]
    assert not comparisons(audit_rows(rows, failed), failed)["local-edit"]["engineering_gate"]


@pytest.mark.parametrize("change", ["fresh_ar", "summary", "cases", "identity", "original"])
def test_report_reaudits_references_inputs_identities_and_summary(tmp_path, change):
    from scripts.write_transport_validation_report import rebuild

    tasks, references, rows = simulated_evidence()
    summary = audit_rows(rows, tasks)
    info = dict(status="complete", original_unchanged=True, rows=192, fresh_ar_correct=24,
                fresh_ar_artifact_correct=24, cases_sha256=file_hash(CASES),
                original_sha256={n: input_hashes(ROOT / "results" / n)
                                 for n in ("ot-r", "reuse-confirmation", "infra-v0.3", "runtime-migration")},
                identities={r["request_id"]: {k: r[k] for k in ("prompt_sha256", "snapshot_id", "prompt_tokens")}
                            for r in references}, load_and_identity_seconds=1., source={"sha256": "unit-only"},
                protocol_sha256="unit-only", runner_sha256="unit-only")

    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")

    write("provenance.json", info)
    write("summary.json", summary)
    write("comparisons.json", comparisons(summary, tasks))
    for name, values in (("fresh-ar.jsonl", references), ("rows.jsonl", rows)):
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in values) + "\n", encoding="utf-8")
    rebuild(tmp_path, tmp_path / "unit-report.md", cases=CASES)
    assert b"\r" not in (tmp_path / "unit-report.md").read_bytes()
    assert b"\r" not in (tmp_path / "diagnostic.json").read_bytes()
    if change == "fresh_ar":
        with (tmp_path / "fresh-ar.jsonl").open("a") as stream:
            stream.write(json.dumps(references[0]) + "\n")
    elif change == "summary":
        write("summary.json", {})
    elif change == "cases":
        info["cases_sha256"] = "wrong"
    elif change == "identity":
        info["identities"][tasks[0].request.id]["prompt_sha256"] = "b" * 64
    else:
        info["original_sha256"] = {}
    write("provenance.json", info)
    with pytest.raises(ValueError):
        rebuild(tmp_path, tmp_path / "bad-report.md", cases=CASES)
