from pathlib import Path

import pytest

from memorydraft.evaluation.prefix_risk_continuation import check_reference_profile, protected, reference_source
from memorydraft.evaluation.prefix_risk_probe import inputs

ROOT = Path(__file__).resolve().parents[1]


def test_continuation_includes_failed_contract_and_all_original_references():
    tasks, _, _ = inputs(ROOT)
    attached, metadata, profile = reference_source(ROOT, tasks)
    assert len(attached) == metadata["references"] == 12
    assert metadata["contract_correct"] == 11
    assert [t.request.id for t in attached if not t.reference["contract"]["artifact_correct"]] == ["vector-ingest:0"]
    assert metadata["historical_ar_seconds"] > 0 and profile["dtype"] == "float32"
    assert all(a.request == b.request and a.snapshot_id == b.snapshot_id for a, b in zip(tasks, attached))


def test_reference_inference_profile_must_remain_fixed():
    tasks, _, _ = inputs(ROOT)
    _, _, profile = reference_source(ROOT, tasks)
    check_reference_profile(profile, {**profile, "db": "new-isolated-history.db"})
    with pytest.raises(ValueError, match="profile"):
        check_reference_profile(profile, {**profile, "template_options": {"enable_thinking": True}})


def test_original_stop_evidence_is_protected_but_new_output_is_separate(tmp_path):
    old = tmp_path / "results/prefix-risk/mechanism/provenance.json"
    new = tmp_path / "results/prefix-risk/continuation/mechanism/provenance.json"
    for p in (old, new):
        p.parent.mkdir(parents=True)
        p.write_text("{}")
    actual = protected(tmp_path)
    assert old.relative_to(tmp_path).as_posix() in actual
    assert new.relative_to(tmp_path).as_posix() not in actual
