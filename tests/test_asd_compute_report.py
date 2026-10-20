from copy import deepcopy

import pytest

from scripts.run_asd_compute_probe import ROOT
from scripts.write_asd_compute_report import analyze, render


READY = (ROOT / "results/asd-compute/execution-verification.json").exists()


@pytest.mark.skipif(not READY, reason="Frozen GPU compute evidence not published yet")
def test_published_compute_report_rebuild():
    result = analyze(ROOT / "results/asd-compute")
    assert result["generations"] == 100 and len(result["entries"]) == 16
    assert render(result) == (ROOT / "docs/ASD_COMPUTE_RESULTS.md").read_text(encoding="utf-8")


@pytest.mark.skipif(not READY, reason="Frozen GPU compute evidence not published yet")
@pytest.mark.parametrize("mutation,expected", [("reset", "Preserved action ledger"),
                                              ("duplicate", "Duplicate/missing branches"),
                                              ("future_relaxed", "Strict future")])
def test_published_compute_audit_rejects_invalid_branch_data(monkeypatch, mutation, expected):
    import scripts.write_asd_compute_report as module
    original = module.read_rows

    def altered(path):
        rows = deepcopy(original(path))
        if path.parent.name == "math-a" and path.name == "branches.jsonl":
            if mutation == "reset":
                rows[0]["action"]["spent"] = 999.
            elif mutation == "duplicate":
                rows[1] = deepcopy(rows[0])
            else:
                row = rows[0]
                row["rounds"][row["state"]["block"] + 1]["regret"] = .1
        return rows

    monkeypatch.setattr(module, "read_rows", altered)
    with pytest.raises(ValueError, match=expected):
        module.analyze(ROOT / "results/asd-compute")
