from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from memorydraft.evaluation.benefit_probe import inputs
from memorydraft.evaluation.runtime_migration import read_rows
from scripts.write_benefit_report import audit_costs, audit_decisions, build, check_provenance, equivalent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/ot-benefit"


def test_report_rebuilds_actual_costs_project_exclusion_and_strict_paths():
    report, diagnostic = build(SOURCE)
    assert report == (ROOT / "docs/OT_BENEFIT_RESULTS.md").read_text(encoding="utf-8")
    assert diagnostic == json.loads((SOURCE / "diagnostic.json").read_text())
    assert diagnostic["paired_paths"] == 112
    assert diagnostic["offline"]["states"] == 14


def test_changed_decision_or_same_project_neighbour_is_rejected():
    costs = audit_costs(SOURCE)
    rows = deepcopy(read_rows(SOURCE / "mechanism/rows.jsonl"))
    tasks, _ = inputs(ROOT, subset=True)
    row = next(r for r in rows if r["decisions"])
    row["decisions"][0]["projects"][0] = row["project_id"]
    with pytest.raises(ValueError, match="pre-OT benefit decision"):
        audit_decisions(rows, costs, tasks)


def test_cost_kernel_measurements_must_match_policy_cost(tmp_path):
    directory = tmp_path / "cost"
    directory.mkdir()
    for name in ("costs.json", "measurements.json"):
        shutil.copyfile(SOURCE / "cost" / name, directory / name)
    value = json.loads((directory / "costs.json").read_text())
    value["profiles"][0]["verify"]["32"] *= 2
    (directory / "costs.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Target cost"):
        audit_costs(tmp_path)


def test_execution_runner_and_package_are_bound_to_original_manifest():
    manifest = json.loads((SOURCE / "execution-source.json").read_text())
    info = json.loads((SOURCE / "mechanism/provenance.json").read_text())
    info["runner_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="execution provenance"):
        check_provenance(info, manifest, "scripts/run_benefit_probe.py")


def test_float_audit_tolerance_does_not_relax_routes_lengths_or_tokens():
    assert equivalent({"advance": 25.0}, {"advance": 25.000000000000004})
    assert not equivalent([32], [31])
    assert not equivalent({"route": "ot"}, {"route": "edit"})
    assert not equivalent(25., 25.00001)
