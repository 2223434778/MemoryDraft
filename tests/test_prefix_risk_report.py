from pathlib import Path
import shutil

import pytest

from memorydraft.evaluation.runtime_migration import file_hash
from scripts.write_prefix_risk_report import ROOT, build

SOURCE = ROOT / "results/prefix-risk"


def test_contract_stop_is_reported_without_inventing_policy_measurements():
    before = {p: file_hash(p) for p in SOURCE.rglob("*") if p.is_file()}
    report, diagnostic = build(SOURCE)
    assert report == (ROOT / "docs/PREFIX_RISK_RESULTS.md").read_text(encoding="utf-8")
    assert diagnostic["status"] == "stopped_target_contract" and diagnostic["references"] == 12
    assert not diagnostic["contract_passed"]
    assert "mechanism" not in diagnostic and "timing" not in diagnostic
    assert before == {p: file_hash(p) for p in before}


def test_report_refuses_timing_after_contract_stop(tmp_path):
    source = tmp_path / "evidence"
    shutil.copytree(SOURCE, source)
    (source / "timing").mkdir()
    with pytest.raises(ValueError, match="continued"):
        build(source)
