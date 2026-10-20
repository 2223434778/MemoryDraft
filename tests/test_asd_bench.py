from pathlib import Path

import pytest

from memorydraft.evaluation.asd_bench import load_tasks, numeric_answer, request_for, score

SOURCE = Path(__file__).resolve().parents[1] / "data/research/asd-bench/records.json"


def test_frozen_slices_separate_labels_from_model_inputs():
    records = load_tasks(SOURCE)
    assert len(records) == 16
    assert len({request_for(r).user_id for r in records}) == 16
    for record in records:
        prompt = request_for(record).messages[0]["content"]
        hidden = record["answer"] if record["benchmark"] == "gsm8k" else record["code"]
        assert hidden not in prompt


@pytest.mark.parametrize("record", load_tasks(SOURCE), ids=lambda r: r["id"])
def test_published_reference_answers_pass_the_small_quality_checker(record):
    text = record["answer"] if record["benchmark"] == "gsm8k" else record["code"]
    assert score(record, text)["correct"]


def test_numeric_normalization_and_missing_answer():
    assert numeric_answer("reason 12\n#### $1,200.50") == numeric_answer(r"\boxed{1200.5}")
    assert numeric_answer("unfinished reasoning 1200") is None
    assert numeric_answer("#### <answer>18</answer>") == numeric_answer("#### 18")
    assert numeric_answer("#### <answer>\n20") == numeric_answer("#### 20")
    assert numeric_answer(r"\boxed{\$70,000}") == numeric_answer("#### 70000")


def test_mbpp_rejects_wrong_code_and_unrelated_process_access():
    record = next(r for r in load_tasks(SOURCE) if r["id"] == "mbpp:17")
    assert not score(record, "def square_perimeter(n):\n return n*3")["correct"]
    assert not score(record, "import os\ndef square_perimeter(n):\n return n*4")["correct"]
    assert score(record, "```python\ndef square_perimeter(n):\n return n*4\n```")["correct"]
