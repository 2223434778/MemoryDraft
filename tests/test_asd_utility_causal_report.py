from copy import deepcopy

import pytest

from scripts.run_asd_utility_causal_probe import ROOT, sources
from scripts.write_asd_utility_causal_report import analyze, audit_policy, render


def test_scalar_audit_reconstructs_published_utility_source_trajectories():
    for row in sources().values():
        assert audit_policy(row, None, row, [151645, 151643]) == len(row["events"])


@pytest.mark.parametrize("field", ["spent", "accepted", "proposal_eos"])
def test_scalar_audit_rejects_ledger_selection_and_eos_tampering(field):
    source = sources()["gsm8k:21"]
    row = deepcopy(source)
    row["events"][1][field] = [] if field == "proposal_eos" else 999
    if field == "proposal_eos":
        row["events"][1][field] = [1]
    with pytest.raises(ValueError):
        audit_policy(row, None, source, [151645, 151643])


@pytest.mark.skipif(not (ROOT / "results/asd-utility-causal/execution-verification.json").exists(),
                    reason="GPU diagnostic evidence not published yet")
def test_published_causal_evidence_rebuild():
    result = analyze(ROOT / "results/asd-utility-causal")
    assert result["policy_rows"] == 14 and result["branch_rows"] == 48
    assert render(result) == (ROOT / "docs/ASD_UTILITY_CAUSAL_RESULTS.md").read_text(encoding="utf-8")


@pytest.mark.skipif(not (ROOT / "results/asd-utility-causal/execution-verification.json").exists(),
                    reason="GPU diagnostic evidence not published yet")
@pytest.mark.parametrize("mutation", ["budget_reset", "duplicate_branch"])
def test_published_audit_rejects_reset_budget_and_duplicate_branches(monkeypatch, mutation):
    import scripts.write_asd_utility_causal_report as module
    original = module.read_rows

    def altered(path):
        rows = deepcopy(original(path))
        if path.parent.name == "gsm":
            if mutation == "budget_reset" and path.name == "policies.jsonl":
                rows[2]["intervention"]["spent"] = 0.
            elif mutation == "duplicate_branch" and path.name == "branches.jsonl":
                rows[1] = deepcopy(rows[0])
        return rows

    monkeypatch.setattr(module, "read_rows", altered)
    expected = "Repair budget reset" if mutation == "budget_reset" else "Duplicate/missing branch"
    with pytest.raises(ValueError, match=expected):
        module.analyze(ROOT / "results/asd-utility-causal")
