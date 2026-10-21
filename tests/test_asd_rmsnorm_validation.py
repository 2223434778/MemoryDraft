from copy import deepcopy
import gzip
import hashlib
import json

import pytest

from memorydraft.evaluation.asd_suite import BENCHMARKS, load_suite
from memorydraft.evaluation.runtime_migration import read_rows
from scripts import run_asd_rmsnorm_suite_probe as runner
from scripts import run_asd_rmsnorm_validation_probe as probe
from scripts import write_asd_rmsnorm_validation_report as report
from test_asd_rmsnorm_suite import matrix, synthetic


def test_validation_loader_uses_only_fixed_48_and_is_disjoint():
    records, positions = probe.records_and_positions()
    assert len(records) == 48 and {r["split"] for r in records} == {"validation"}
    ids = {r["id"] for r in records}
    for split in ("development", "reserve"):
        assert not ids & {r["id"] for r in load_suite(probe.ROOT/"data/research/asd-suite", split)}
    for shard in (0, 1):
        selected = [r for r in records if positions[r["id"]]%2 == shard]
        assert len(selected) == 24
        assert all(sum(r["benchmark"] == b for r in selected) == 4 for b in BENCHMARKS)


@pytest.mark.parametrize("compressed", [False, True])
def test_isolation_finds_nested_previous_generation(tmp_path, compressed):
    dest = tmp_path/"results/previous"
    dest.mkdir(parents=True)
    data = json.dumps(dict(request_id="held-out", result=dict(generated_token_ids=[10])))
    (dest/("rows.jsonl.gz" if compressed else "rows.json")).write_bytes(gzip.compress(data.encode()) if compressed else data.encode())
    with pytest.raises(ValueError, match="already present"):
        probe.exposure_check(tmp_path, {"held-out"})


@pytest.mark.parametrize("folder,allowed", [("asd-rmsnorm-validation", True), ("old-job", False)])
def test_isolation_excludes_only_current_raw_directory(tmp_path, folder, allowed):
    dest = tmp_path/"results/raw"/folder
    dest.mkdir(parents=True)
    (dest/"rows.jsonl").write_text(json.dumps(dict(request_id="held-out", token_ids=[1])))
    if allowed:
        assert probe.exposure_check(tmp_path, {"held-out"})["protected_generation_matches"] == 0
    else:
        with pytest.raises(ValueError, match="already present"):
            probe.exposure_check(tmp_path, {"held-out"})


def test_candidate_checksum_rejects_changed_code_with_line_ending_tolerance(tmp_path):
    (tmp_path/"fixture.py").write_bytes(b"x = 1\r\n")
    expected = {"fixture.py": hashlib.sha256(b"x = 1\n").hexdigest()}
    assert probe.candidate_check(tmp_path, expected)["files"] == expected
    (tmp_path/"fixture.py").write_bytes(b"x = 2\n")
    with pytest.raises(ValueError, match="candidate changed"):
        probe.candidate_check(tmp_path, expected)


@pytest.mark.parametrize("input_drift", [False, True])
def test_new_question_has_no_old_answer_dependency_and_checks_inputs(monkeypatch, tmp_path, input_drift):
    # Reuse a public development trajectory as a SYNTHETIC output fixture only.
    old = runner.previous_rows()
    q = runner.records_and_positions()[0][0]
    record = dict(q, id="new-validation-fixture", split="validation")
    def execute(engine, compiler, r, method, profile, limit):
        row = synthetic(old[q["id"], method], method, profile)
        row["request_id"] = r["id"]
        if input_drift and profile != "eager":
            row["prompt_sha256"] = "changed"
        return row
    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(runner, "score", lambda *args: dict(correct=False))
    if input_drift:
        with pytest.raises(ValueError, match="input"):
            runner.run_question(None, None, record, None, tmp_path)
    else:
        assert runner.run_question(None, None, record, None, tmp_path)["timed"] == 24
        assert all(not r["quality"]["correct"] for r in read_rows(tmp_path/"rows.jsonl"))


@pytest.fixture
def gate_fixture(matrix):
    """Synthetic timings over development fixtures, not validation evidence."""
    rows, refs, records, _, stops = matrix
    audit = report.common.audit(rows, refs, records, None, stops)
    summary = report.common.summarize(rows, refs, records, seed="rmsnorm-validation", development_subsets=False)
    info = [dict(status="complete", split="validation", original_unchanged=True, cache_bytes=64,
                 compiler=dict(graph_count=4, backend_seconds=1., float32_matmul_precision="highest", allow_tf32=False),
                 initial_isolation=dict(protected_generation_matches=0, candidate=dict(files=deepcopy(probe.CANDIDATE_FILES)))) for _ in (0, 1)]
    execution = dict(status="complete", files_verified_before=100, files_verified_after=100)
    return summary, audit, rows, refs, info, execution


def test_validation_audit_has_no_old_output_claim_and_no_development_subsets(gate_fixture):
    summary, audit, *rest = gate_fixture
    assert audit["eager_old_trajectory_checks"] == 0 and audit["quality_rows"] == 480
    assert all(set(v) == {"same_tokens_descriptive"} for v in summary["subsets"].values())
    assert report.admission(summary, audit, *rest)["eligible_for_explicit_profile"]


@pytest.mark.parametrize("change", ["ci", "family", "strict", "quality", "text", "cache", "isolation", "candidate"])
def test_preregistered_gate_does_not_hide_failed_conditions(gate_fixture, change):
    summary, audit, rows, refs, info, execution = gate_fixture
    if change == "ci":
        summary["overall"]["DFlash"]["ci95"][1] = 1.001
    elif change == "family":
        summary["families"]["json"]["DFlash"]["ratio"] = 1.001
    elif change == "strict":
        audit["strict_matches_AR"]["rmsnorm-inductor"] = 47
    elif change in {"quality", "text"}:
        row = next(r for r in rows if r["profile"] != "eager" and r["repeat"] == 0 and r["quality"]["correct"])
        if change == "quality":
            row["quality"]["correct"] = False
        else:
            row["text"] += "changed"
    elif change == "cache":
        info[0]["cache_bytes"] = 4*2**30+1
    elif change == "isolation":
        info[0]["initial_isolation"]["protected_generation_matches"] = 1
    else:
        info[0]["initial_isolation"]["candidate"]["files"] = {}
    result = report.admission(summary, audit, rows, refs, info, execution)
    assert not result["eligible_for_explicit_profile"] and not all(result["checks"].values())
