from copy import deepcopy
from pathlib import Path

import pytest

from memorydraft.evaluation.long_context import ARMS, LEVELS, check_input, load_data, request_for, timing
from scripts.prepare_asd_long_context import build
from scripts.validate_asd_runtime import ADAPTER_SHA
from scripts.write_asd_long_context_report import analyze

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_inputs_are_complete_nested_examples_without_test_answer():
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):return list(range(len(messages[0]["content"].split())))
    task = dict(id="test",question="A final question?",answer="DO_NOT_INCLUDE_TEST_ANSWER")
    demos = [dict(id=str(i),question="An example?",answer="work "*100+"#### 7") for i in range(48)]
    result = build([task],demos,Tokenizer())
    assert [r["level"] for r in result] == list(LEVELS)
    for row in result:
        text = row["request"]["messages"][0]["content"]
        assert "DO_NOT_INCLUDE_TEST_ANSWER" not in text
        assert text.endswith("Final question:\nA final question?\nSolution:")
        assert row["demonstration_ids"] == [str(i) for i in range(row["examples"])]
        assert .75*row["level"] <= row["prompt_tokens"] <= 1.25*row["level"]
        req = request_for(row,128)
        assert req.max_new_tokens == 128 and req.messages == request_for(row).messages


def test_real_frozen_data_matrix_and_demo_exclusions():
    import json
    from scripts.prepare_asd_adaptation import normalized
    tasks,items,info = load_data(ROOT)
    demos = json.loads((ROOT/"data/workloads/asd-long-context/demonstrations.json").read_text(encoding="utf-8"))
    training = json.loads((ROOT/"data/research/asd-adaptation/questions.json").read_text(encoding="utf-8"))
    assert len(tasks) == 8 and len(items) == 24
    assert not {normalized(r["question"]) for r in demos} & {normalized(r["question"]) for r in [*tasks,*training]}
    assert len(info["files"]) == 4


def fixture():
    item = dict(task_id="q",level=512,prompt_tokens=500,prompt_sha256="p",examples=2,request=dict(id="r"))
    task = dict(id="q",benchmark="gsm8k",answer="#### 7")
    quality = dict(correct=True,parsed_answer="7",expected_answer="7",error=None)
    base = dict(request_id="r",prompt_tokens=500,prompt_sha256="p",execution_profile="rmsnorm-inductor",
                latency_seconds=3.,ttft_seconds=1.,token_ids=[7]*128,text="#### 7",finish_reason="eos",asd=None)
    rows = []
    for arm in ARMS:
        for stage,repeats in (("natural",2),("prefix",1)):
            for repeat in range(repeats):
                rows.append(dict(base,method="DFlash" if arm == "DFlash" else "ASD",arm=arm,stage=stage,repeat=repeat,
                    approximate=arm != "DFlash",draft_adapter_sha256=ADAPTER_SHA if arm == "position-ASD" else None,
                    asd=None if arm == "DFlash" else dict(budget=8,spent=1),quality=quality if stage == "natural" else None,
                    counters=dict(rounds=10,compile_graphs_before=4,compile_graphs_after=4)))
    return rows,[base],[item],[task]


def test_analysis_separates_eos_prefix_work_and_first_token_cost():
    rows,refs,items,tasks = fixture()
    assert timing(rows[0])["post_first_seconds"] == 2
    result = analyze(rows,refs,items,tasks)
    assert result["all_strict_match_ar"] and result["formal_new_graphs"] == 0
    assert result["summary"]["512"]["ASD"]["common_full_128"] == 1
    for row in rows:
        if row["arm"] == "position-ASD" and row["stage"] == "prefix":row["token_ids"] = [7]*100
    result = analyze(rows,refs,items,tasks)
    assert result["summary"]["512"]["ASD"]["prefix_ratio_to_asd"] is None


@pytest.mark.parametrize("bad",["missing","duplicate","input","adapter","budget","unstable","score"])
def test_incomplete_or_invalid_results_are_not_reported_as_success(bad):
    rows,refs,items,tasks = deepcopy(fixture())
    if bad == "missing":rows.pop()
    elif bad == "duplicate":rows[-1] = rows[0]
    elif bad == "input":rows[0]["prompt_sha256"] = "changed"
    elif bad == "adapter":rows[-1]["draft_adapter_sha256"] = "changed"
    elif bad == "budget":rows[-1]["asd"]["spent"] = 8.000001
    elif bad == "unstable":rows[1]["token_ids"] = [6]
    elif bad == "score":rows[0]["quality"] = dict(correct=False)
    with pytest.raises(ValueError):analyze(rows,refs,items,tasks)


def test_strict_ar_difference_is_kept_as_a_result_not_hidden():
    rows,refs,items,tasks = fixture()
    for row in rows:
        if row["arm"] == "DFlash":row["token_ids"] = [8]*128
    result = analyze(rows,refs,items,tasks)
    assert not result["all_strict_match_ar"] and len(result["strict_mismatches"]) == 3


@pytest.mark.parametrize("first,tokens",[(None,[1]),(4,[1]),(-1,[1]),(1,[])])
def test_invalid_timing_rejected(first,tokens):
    with pytest.raises(ValueError):timing(dict(latency_seconds=3,ttft_seconds=first,token_ids=tokens))
