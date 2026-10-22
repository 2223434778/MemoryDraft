from copy import deepcopy

import pytest

from scripts.run_asd_candidate_probe import METHODS
from scripts.write_asd_candidate_report import analyze


def rows():
    return [dict(request_id=f"q{q}", benchmark="mbpp", candidate=m, repeat=r,
                 latency_seconds=1. if m in {"native", "packed-linear"} else .9,
                 token_ids=[1, 2], text="a", token_match=True, text_match=True, finish_match=True,
                 prompt_sha256="fixed", snapshot_id="empty", finish_reason="eos", quality={"passed": True},
                 rounds=[dict(proposed=3, accepted=1, draft_forward=True)])
            for q in range(2) for m in METHODS for r in range(3)]


def test_natural_request_gate_uses_real_times_and_keeps_all_methods():
    result = analyze(rows())
    assert set(result["scopes"]["overall"]) == set(METHODS)
    assert set(result["expand_candidates"]) == {"tree32", "tree48", "repair4", "repair8"}
    assert result["scopes"]["overall"]["tree32"]["latency_ratio"] == pytest.approx(.9)
    assert result["scopes"]["overall"]["tree32"]["verify_nodes"] == 8


@pytest.mark.parametrize("bad", ["missing", "duplicate", "nan", "zero"])
def test_missing_or_invalid_candidate_measurements_rejected(bad):
    data = rows()
    if bad == "missing":
        data.pop()
    elif bad == "duplicate":
        data.append(deepcopy(data[0]))
    else:
        data[0]["latency_seconds"] = float("nan") if bad == "nan" else 0
    with pytest.raises(ValueError):
        analyze(data)


@pytest.mark.parametrize("bad", ["tokens", "input", "repeat", "reference", "native"])
def test_numerical_or_trajectory_divergence_is_retained_and_blocks_gate(bad):
    data = rows()
    chosen = next(r for r in data if r["candidate"] == ("native" if bad == "native" else "tree32"))
    if bad in {"tokens", "native"}:
        chosen["token_ids"] = [9]
    elif bad == "input":
        chosen["prompt_sha256"] = "changed"
    elif bad == "repeat":
        chosen["rounds"] = []
    else:
        chosen["token_match"] = False
    result = analyze(data)
    assert not result["scopes"]["overall"]["tree32"]["safe"]
    assert "tree32" not in result["expand_candidates"]


def test_one_lucky_repeat_cannot_pass_even_with_good_aggregate():
    data = rows()
    for row in data:
        if row["candidate"] == "tree32":
            row["latency_seconds"] = 1.1 if row["repeat"] == 0 else .8
    assert "tree32" not in analyze(data)["expand_candidates"]


def test_failure_in_another_family_blocks_even_a_fast_safe_family():
    data = rows()
    extra = deepcopy([r for r in data if r["request_id"] == "q0"])
    for row in extra:
        row.update(request_id="q2", benchmark="gsm8k")
        if row["candidate"] == "tree32":
            row["token_match"] = False
    result = analyze([*data, *extra])
    assert result["scopes"]["mbpp"]["tree32"]["safe"]
    assert not result["globally_safe"]["tree32"]
    assert "tree32" not in result["expand_candidates"]


def test_faster_mechanism_but_slower_request_has_no_admission():
    data = rows()
    for row in data:
        if row["candidate"] not in {"native", "packed-linear"}:
            row.update(latency_seconds=1.01, rounds=[])
    assert not analyze(data)["expand_candidates"]


def test_fresh_pipeline_creates_artifacts_before_cpu_preflight(tmp_path, monkeypatch):
    import json
    import sys
    from scripts import run_asd_candidate_pipeline as pipeline
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["pipeline"])
    (tmp_path/"ASD_CANDIDATE_SOURCE_MANIFEST.json").write_text(json.dumps(dict(commit="test", files={})))
    def stop(*args, **kwargs):
        assert (tmp_path/"artifacts").is_dir()
        raise RuntimeError("fixture before GPU")
    monkeypatch.setattr(pipeline, "run_owned", stop)
    with pytest.raises(RuntimeError, match="fixture before GPU"):
        pipeline.main()
    assert (tmp_path/"results/raw/asd-candidate-pilot/execution-verification.json").is_file()
