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


@pytest.mark.parametrize("change", ["fresh_ar", "summary", "original"])
def test_report_checks_fresh_reference_summary_and_original_hashes(tmp_path, change):
    import json

    from memorydraft.evaluation.runtime_migration import input_hashes
    from scripts.write_transport_report import rebuild

    tasks, _ = probe_inputs(ROOT)
    rows = rows_for(tasks)
    originals = {name: input_hashes(ROOT / "results" / name) for name in ("reuse-confirmation", "infra-v0.3")}
    info = dict(status="complete", original_unchanged=True, fresh_ar_correct=10, rows=80,
                original_sha256=originals, source={"sha256": "unit-test-only"},
                runner_sha256="unit-test", protocol_sha256="unit-test", load_seconds=1.)
    fresh = [{"request_id": t.request.id, "token_ids": t.reference["token_ids"],
              "prompt_sha256": t.reference["prompt_sha256"], "snapshot_id": t.snapshot_id, "latency_seconds": 1.}
             for t in tasks]
    offline = {"original_sha256": originals, "per_request": {
        t.request.id: {"misses": 0, "raw_ge8": 0, "pool_ge8": 0,
                       "ot": {"recovered_ge8": 0, "zero_accept": 0},
                       "edit": {"recovered_ge8": 0, "zero_accept": 0}} for t in tasks}}

    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")

    write("provenance.json", info)
    write("offline-local-v1.json", offline)
    write("summary.json", audit(rows, tasks))
    for name, values in (("rows.jsonl", rows), ("fresh-ar.jsonl", fresh)):
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in values) + "\n", encoding="utf-8")
    # Simulated evidence tests only the report audit and is never published.
    rebuild(tmp_path, tmp_path / "unit-report.md")
    assert b"\r" not in (tmp_path / "diagnostic.json").read_bytes()
    if change == "fresh_ar":
        with (tmp_path / "fresh-ar.jsonl").open("a") as stream:
            stream.write(json.dumps(fresh[0]) + "\n")
    elif change == "summary":
        write("summary.json", {})
    else:
        info["original_sha256"] = {}
        write("provenance.json", info)
    with pytest.raises(ValueError):
        rebuild(tmp_path, tmp_path / "invalid.md")
