from copy import deepcopy

import pytest

from scripts.run_asd_adaptation_eval import POLICIES_USED, VARIANTS
from scripts.write_asd_adaptation_report import analyze


def rows():
    return [dict(request_id="q", benchmark="gsm8k", stage=s, variant=v, policy=p, repeat=i,
                latency_seconds=1. if v == "base" else .9, token_ids=[1]*128, rounds=[{}]*10,
                asd=dict(spent=1., budget=8.) if p == "ASD" else None,
                same_input=True, strict_token_match=True, base_reference_match=True,
                quality=dict(correct=True), finish_reason="eos")
            for s, repeats in (("natural", 3), ("prefix", 2)) for v in VARIANTS for p in POLICIES_USED for i in range(repeats)]


def test_matched_budget_natural_and_equal_prefix_costs_are_separate():
    a = analyze(rows())
    assert a["all_safe"]
    assert a["scopes"]["gsm8k"]["natural/ordinary/DFlash"]["latency_ratio"] == pytest.approx(.9)
    assert a["scopes"]["gsm8k"]["prefix/asd-prefix/ASD"]["common_full_128_vs_ordinary"] == 1


@pytest.mark.parametrize("bad", ["missing", "duplicate", "nan", "zero"])
def test_unusable_measurements_rejected(bad):
    data = rows()
    if bad == "missing":
        data.pop()
    elif bad == "duplicate":
        data.append(deepcopy(data[0]))
    else:
        data[0]["latency_seconds"] = float("nan") if bad == "nan" else 0
    with pytest.raises(ValueError):
        analyze(data)


@pytest.mark.parametrize("bad", ["strict", "input", "budget", "base", "unstable"])
def test_quality_and_numerical_failures_are_retained(bad):
    data = rows()
    row = next(r for r in data if r["stage"] == "natural" and r["policy"] == ("ASD" if bad == "budget" else "DFlash"))
    if bad == "strict":
        row["strict_token_match"] = False
    elif bad == "input":
        row["same_input"] = False
    elif bad == "budget":
        row["asd"]["spent"] = 9
    elif bad == "base":
        row["base_reference_match"] = False
    else:
        row["rounds"] = []
    assert not analyze(data)["all_safe"]


def test_shortened_asd_output_reports_quality_loss_and_cannot_enter_full128():
    data = rows()
    for r in data:
        if r["variant"] == "asd-prefix" and r["policy"] == "ASD":
            r["token_ids"] = [2]*60
            r["quality"] = dict(correct=False)
    a = analyze(data)["scopes"]["gsm8k"]
    assert a["natural/asd-prefix/ASD"]["lost_correct"] == 1
    assert a["prefix/asd-prefix/ASD"]["common_full_128_requests"] == 0
    assert a["prefix/asd-prefix/ASD"]["common_full_128_ratio"] is None


def test_fresh_pipeline_artifacts_parent_created_before_any_gpu(tmp_path, monkeypatch):
    import json
    from scripts import run_asd_adaptation_pipeline as pipeline
    monkeypatch.chdir(tmp_path)
    (tmp_path/"ASD_ADAPTATION_SOURCE_MANIFEST.json").write_text(json.dumps(dict(commit="test", files={})))
    def stop(*args, **kwargs):
        assert (tmp_path/"artifacts").is_dir()
        raise RuntimeError("fixture stop")
    monkeypatch.setattr(pipeline, "run_owned", stop)
    with pytest.raises(RuntimeError, match="fixture stop"):
        pipeline.main()
    assert (tmp_path/"results/raw/asd-adaptation/execution-verification.json").is_file()
