import shutil

import pytest

from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from scripts.write_prefix_risk_controls_report import ROOT, build

SOURCE = ROOT / "results/prefix-risk/continuation/controls"


def test_known_case_switch_is_reported_separately_without_confidence_attribution():
    before = {p: file_hash(p) for p in SOURCE.rglob("*") if p.is_file()}
    report, d = build(SOURCE)
    assert report == (ROOT / "docs/PREFIX_RISK_CONTROLS_RESULTS.md").read_text(encoding="utf-8")
    payment = d["requests"]["payment-runbook:1"]
    assert payment["Risk-Splice"]["rounds"] == payment["Risk-NoConfidence"]["rounds"] == 7
    assert payment["Splice-Edit"]["rounds"] == 8
    assert len(read_rows(SOURCE / "timing/rows.jsonl")) == 20
    assert "不并入12请求" in report and "尚无置信度的独立增量证据" in report
    assert before == {p: file_hash(p) for p in before}


def test_controls_cannot_change_recorded_primary_evidence(tmp_path):
    from unittest.mock import patch

    source = tmp_path / "evidence"
    shutil.copytree(SOURCE, source)
    with patch("scripts.write_prefix_risk_controls_report.primary_hashes", return_value={}):
        with pytest.raises(ValueError, match="primary evidence"):
            build(source)
