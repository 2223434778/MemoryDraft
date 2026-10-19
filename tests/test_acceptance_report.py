from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.evaluation.splice_probe import inputs
from scripts.evaluate_acceptance_probe import audit_states, evaluate
from scripts.write_acceptance_report import ROOT, build


SOURCE = ROOT / "results/acceptance"


def test_report_rebuild_checks_frozen_results_and_preserves_evidence():
    before = {p: file_hash(p) for p in SOURCE.rglob("*") if p.is_file()}
    report, saved = build(SOURCE)
    assert report == (ROOT / "docs/ACCEPTANCE_RESULTS.md").read_text(encoding="utf-8")
    assert saved["summary"]["states"] == 56 and saved["summary"]["candidates"] == 69
    assert not saved["summary"]["gate"]["passed"]
    assert saved["summary"]["paired"]["confidence"]["error"] < saved["summary"]["paired"]["old"]["error"]
    assert before == {p: file_hash(p) for p in before}


def test_audit_rejects_duplicate_states_and_changed_candidate_labels():
    states = read_rows(SOURCE / "collection/states.jsonl")
    tasks, _ = inputs(ROOT)
    with pytest.raises(ValueError, match="Duplicate"):
        audit_states(states + [states[0]], tasks)
    changed = deepcopy(states)
    candidate = next(s["candidates"][0] for s in changed if s["candidates"])
    candidate["accepted"] = (candidate["accepted"] + 1) % len(candidate["tokens"])
    with pytest.raises(ValueError, match="candidate"):
        audit_states(changed, tasks)


def test_saved_calibration_cannot_bypass_project_exclusion():
    from memorydraft.retrieval.acceptance import AcceptancePredictor

    models = json.loads((SOURCE / "evaluation/models.json").read_text())
    project = next(iter(models))
    models[project]["confidence"]["projects"].append(project)
    fitted = {p: {name: AcceptancePredictor(**m) for name, m in values.items()} for p, values in models.items()}
    with pytest.raises(ValueError, match="leaked"):
        evaluate(read_rows(SOURCE / "collection/states.jsonl"), calibrated=fitted)


def test_report_rejects_changed_saved_predictions(tmp_path):
    source = tmp_path / "evidence"
    shutil.copytree(SOURCE, source)
    path = source / "evaluation/paired.json"
    value = json.loads(path.read_text())
    value[0]["confidence"]["predicted"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="prediction evidence"):
        build(source)
