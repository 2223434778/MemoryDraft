from pathlib import Path
import json
import shutil

import pytest

from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from scripts.run_prefix_risk_controls import IDS, inputs
from scripts.write_prefix_risk_continuation_report import ROOT, build

SOURCE = ROOT / "results/prefix-risk/continuation"


def test_all_requests_complete_despite_failed_contract_and_no_risk_switches():
    before = {p: file_hash(p) for p in SOURCE.rglob("*") if p.is_file()}
    report, d = build(SOURCE)
    assert report == (ROOT / "docs/PREFIX_RISK_CONTINUATION_RESULTS.md").read_text(encoding="utf-8")
    assert d["reference_source"]["references"] == 12 and d["reference_source"]["contract_correct"] == 11
    assert d["mechanism"]["all"]["Risk-Splice"]["counters"]["risk_switches"] == 0
    assert len(read_rows(SOURCE / "timing/rows.jsonl")) == 120
    assert before == {p: file_hash(p) for p in before}


def test_task_failure_cannot_be_relabelled_as_correct(tmp_path):
    source = tmp_path / "evidence"
    shutil.copytree(SOURCE, source)
    rows = read_rows(source / "timing/rows.jsonl")
    row = next(r for r in rows if r["request_id"] == "vector-ingest:0")
    row["contract"]["artifact_correct"] = True
    (source / "timing/rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="Task quality"):
        build(source)


def test_coverage_controls_are_fixed_known_training_projects():
    tasks, _, calibration = inputs(ROOT)
    assert tuple(t.request.id for t in tasks) == IDS
    assert {t.project_id for t in tasks} <= set(calibration["training_projects"])
