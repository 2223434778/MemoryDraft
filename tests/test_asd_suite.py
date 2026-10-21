from collections import Counter
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

import pytest

from memorydraft.evaluation.asd_suite import (BENCHMARKS, SPLITS, canonical, exact_json, json_score, load_suite,
                                              qa_score, request_for, scalar_answer, score, validate_code)
from scripts.prepare_asd_suite import fixtures

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/research/asd-suite"


def test_frozen_dataset_splits_and_squad_articles_are_disjoint():
    splits = {s:load_suite(DATA,s) for s in SPLITS}
    assert [len(splits[s]) for s in SPLITS] == [48,48,96]
    ids = [r["id"] for rows in splits.values() for r in rows]
    assert len(set(ids)) == len(ids)
    assert all(Counter(r["benchmark"] for r in rows)=={b:len(rows)//6 for b in BENCHMARKS} for rows in splits.values())
    assert len({r["group_id"] for rows in splits.values() for r in rows if r["benchmark"]=="squad"}) == 32
    assert len({r["subject"] for r in splits["development"] if r["benchmark"]=="math500"}) == 7


def test_generation_never_reads_hidden_reference_fields():
    for row in load_suite(DATA):
        before = request_for(row)
        altered = deepcopy(row)
        for key in ("answer","solution","canonical_solution","code","test","expected","answers"):
            if key in altered:
                altered[key] = "HIDDEN_REFERENCE_MUST_NOT_REACH_TARGET"
        assert request_for(altered) == before
        assert "HIDDEN_REFERENCE" not in str(before.messages)


def test_dataset_hash_tampering_is_rejected(tmp_path):
    import shutil
    shutil.copytree(DATA,tmp_path/"data")
    p=tmp_path/"data/gsm8k.json"; p.write_bytes(p.read_bytes()+b" ")
    with pytest.raises(ValueError,match="Changed frozen dataset"):
        load_suite(tmp_path/"data")


@pytest.mark.parametrize("split", SPLITS)
def test_all_canonical_references_pass_before_model_evaluation(split):
    for row in load_suite(DATA,split):
        result=score(row,canonical(row))
        assert result["correct"], (row["id"],result)


def test_question_answering_matches_reference_em_f1_without_dropping_explanation():
    assert qa_score("The Eiffel Tower.",["Eiffel Tower"])["correct"]
    partial=qa_score("Eiffel",["Eiffel Tower"])
    assert not partial["correct"] and partial["f1"]==pytest.approx(2/3)
    assert not qa_score("The answer is Eiffel Tower",["Eiffel Tower"])["correct"]


def test_json_scoring_rejects_duplicate_keys_extra_fields_and_bool_integer_confusion():
    assert json_score('{"v":1}',dict(v=1))["correct"]
    assert not json_score('{"v":true}',dict(v=1))["correct"]
    assert not json_score('{"v":1,"v":1}',dict(v=1))["valid_json"]
    assert not json_score('{"v":NaN}',dict(v=1))["valid_json"]
    assert not exact_json(dict(v=1,extra=2),dict(v=1))
    assert json_score('```json\n{"v":1}\n```',dict(v=1))["correct"]


def test_scalar_secondary_is_limited_and_does_not_execute_expressions():
    assert scalar_answer(r"-\frac{-1}{2}")==Fraction(1,2)
    assert scalar_answer("x=0.5")==Fraction(1,2)
    assert scalar_answer(r"\sqrt{2}") is None
    assert scalar_answer("1/0") is None


def test_code_scoring_obeys_tests_and_restricts_process_access():
    record=dict(benchmark="mbpp",test_list=["assert f(2)==3"])
    assert score(record,"def f(x): return x+1")["correct"]
    assert not score(record,"def f(x): return x")["correct"]
    with pytest.raises(ValueError):
        validate_code("import os\nos.system('echo forbidden')")


def test_synthetic_fixtures_are_deterministic_and_not_historical_answers():
    assert fixtures()==fixtures()
    assert len({r["group_id"] for r in fixtures()})==32
    assert all(r["source"].startswith("Project-authored synthetic") for r in fixtures())
