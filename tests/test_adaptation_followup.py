from copy import deepcopy
import json
from pathlib import Path

import pytest

from memorydraft.evaluation.adaptation import prefix_digest
from memorydraft.evaluation.adaptation_followup import aligned_controls, previous_rows, records, VARIANTS, POLICIES
from scripts.prepare_asd_adaptation_followup import normalized, old_questions
from scripts.write_asd_adaptation_followup_report import analyze

ROOT = Path(__file__).resolve().parents[1]


def source_fixture():
    point = dict(request_id="q",source_policy="asd-prefix",source_round=4,split="train",generated_before=3,
                 prefix_ids=[1,2,8,9,10],prefix_sha256=prefix_digest([1,2,8,9,10]),teacher_tokens=[99])
    sources = [dict(request_id="q",source_policy="ordinary",token_ids=[3,4,5,6],finish_reason="length"),
               dict(request_id="q",source_policy="asd-prefix",token_ids=[8,9,10,11],finish_reason="length")]
    return point,sources


def test_position_control_keeps_prompt_and_anchor_position_but_uses_strict_content():
    point,sources = source_fixture()
    result = aligned_controls([point],sources)[0]
    assert result["prefix_ids"] == [1,2,3,4,5]
    assert result["generated_before"] == point["generated_before"]
    assert len(result["prefix_ids"]) == len(point["prefix_ids"])
    assert "teacher_tokens" not in result and "feature_path" not in result
    assert result["matched_asd_prefix_sha256"] == point["prefix_sha256"]
    assert result["prefix_sha256"] == prefix_digest(result["prefix_ids"])


@pytest.mark.parametrize("bad",["ended","eos_anchor","prefix_hash","source_prefix","duplicate"])
def test_unavailable_or_misaligned_prefix_is_not_silently_replaced(bad):
    point,sources = source_fixture()
    if bad == "ended":
        sources[0]["token_ids"] = [3,4]
    elif bad == "eos_anchor":
        sources[0].update(token_ids=[3,4,5],finish_reason="eos")
    elif bad == "prefix_hash":
        point["prefix_sha256"] = "wrong"
    elif bad == "source_prefix":
        sources[1]["token_ids"][0] = 77
    else:
        sources.append(deepcopy(sources[0]))
    with pytest.raises(ValueError):
        aligned_controls([point],sources)


def test_real_previous_pairs_are_all_position_matchable_without_dropping_states():
    points = previous_rows(ROOT,"examples")
    result = aligned_controls(points,previous_rows(ROOT,"sources"))
    assert len(result) == 80
    asd = {(r["request_id"],r["source_round"]):r for r in points if r["source_policy"] == "asd-prefix"}
    for row in result:
        previous = asd[row["request_id"],row["source_round"]]
        assert row["generated_before"] == previous["generated_before"]
        assert len(row["prefix_ids"]) == len(previous["prefix_ids"])
    assert sum(r["split"] == "train" for r in result) == 64


def test_independent_questions_do_not_overlap_existing_benchmark_or_training_selections():
    selected = records(ROOT)
    independent = [r for r in selected if r["study_split"] == "independent"]
    assert len(independent) == 16
    for path in (ROOT/"data/research").rglob("*.json"):
        if "asd-adaptation-followup" in path.parts:
            continue
        for old in old_questions(json.loads(path.read_text(encoding="utf-8"))):
            assert all(old["id"] != r["id"] and normalized(old["question"]) != normalized(r["question"]) for r in independent)


def evaluation_rows():
    return [dict(request_id="q",study_split="independent",stage=s,variant=v,policy=p,repeat=i,
                 latency_seconds={"base":1.,"ordinary":.99,"ordinary-position":.98,"asd-prefix":.96}[v],
                 token_ids=[1]*128,rounds=[{}]*10,asd=dict(budget=8,spent=1,relaxed_tokens=1) if p == "ASD" else None,
                 finish_reason="eos",adapter_sha256=v,same_input=True,strict_token_match=True,
                 base_reference_match=None,quality=dict(correct=True))
            for s,n in (("natural",3),("prefix",2)) for v in VARIANTS for p in POLICIES for i in range(n)]


def test_independent_position_contrast_is_paired_and_all_four_models_share_prefix_set():
    result = analyze(evaluation_rows())
    item = result["scopes"]["independent"]["natural/asd-prefix/ASD"]
    assert result["all_safe"]
    assert item["position_ratio"] == pytest.approx(.96/.98)
    assert item["bootstrap"]["position_ratio"]["high"] == pytest.approx(.96/.98)
    data = evaluation_rows()
    for row in data:
        if row["stage"] == "prefix" and row["variant"] == "ordinary-position":
            row["token_ids"] = [1]*10
    summary = analyze(data)["scopes"]["independent"]
    assert all(v["common_full_128_requests"] == 0 and v["bootstrap"] is None for k,v in summary.items() if k.startswith("prefix"))


@pytest.mark.parametrize("bad",["missing","duplicate","nan","split","strict","budget","old_reference"])
def test_matrix_and_integrity_failures_cannot_be_called_safe(bad):
    data = evaluation_rows()
    if bad == "missing":
        data.pop()
    elif bad == "duplicate":
        data.append(deepcopy(data[0]))
    elif bad == "nan":
        data[0]["latency_seconds"] = float("nan")
    elif bad == "split":
        data[0]["study_split"] = "observed"
    elif bad == "strict":
        data[0]["strict_token_match"] = False
    elif bad == "budget":
        next(r for r in data if r["policy"] == "ASD")["asd"]["spent"] = 9
    else:
        for row in data:
            row["study_split"] = "observed"
            row["base_reference_match"] = False
    if bad in {"missing","duplicate","nan","split"}:
        with pytest.raises(ValueError):
            analyze(data)
    else:
        assert not analyze(data)["all_safe"]


def test_new_quality_regression_is_retained_even_with_faster_latency():
    data = evaluation_rows()
    for row in data:
        if row["stage"] == "natural" and row["variant"] == "asd-prefix" and row["policy"] == "ASD":
            row["quality"] = dict(correct=False)
            row["token_ids"] = [1]*50
    item = analyze(data)["scopes"]["independent"]["natural/asd-prefix/ASD"]
    assert item["lost_correct"] == item["lost_vs_position"] == 1
    assert item["output_tokens"] == 50
