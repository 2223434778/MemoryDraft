from types import SimpleNamespace

import pytest

from memorydraft.evaluation.asd_budget import load_tasks, request_for
from scripts.run_asd_budget_probe import compare_previous, protected, select_records


def test_subset_keeps_source_order_and_rejects_wrong_or_duplicate_ids():
    records = load_tasks("data/research/asd-budget/records.json")
    result = select_records(records, "math500", ["math500:7", "math500:1"])
    assert [r["id"] for r in result] == ["math500:1", "math500:7"]
    for ids in (["gsm8k:8"], ["math500:1", "math500:1"], []):
        with pytest.raises(ValueError):
            select_records(records, "math500", ids)
    from dataclasses import replace
    before = request_for(result[0])
    after = replace(before, max_new_tokens=2048)
    assert before.messages == after.messages and before.max_new_tokens == 512


def test_old_limit_affected_block_is_reported_but_earlier_changes_fail():
    previous = dict(prompt_sha256="p", snapshot_id="s", token_ids=list(range(8)), finish_reason="length",
                    rounds=[dict(proposed=4, generated_before=1), dict(proposed=2, generated_before=6)])
    result = SimpleNamespace(prompt_sha256="p", snapshot_id="s", token_ids=[0,1,2,3,4,5,99,8,9])
    comparison = compare_previous(result, previous)
    assert comparison["common_prefix_tokens"] == comparison["required_prefix_tokens"] == 6
    assert not comparison["exact_previous_prefix"]
    result.token_ids[2] = 99
    with pytest.raises(ValueError, match="before"):
        compare_previous(result, previous)


def test_completed_previous_reference_requires_full_prefix_and_same_input():
    previous = dict(prompt_sha256="p", snapshot_id="s", token_ids=[1,2], finish_reason="eos",
                    rounds=[dict(proposed=4, generated_before=1)])
    result = SimpleNamespace(prompt_sha256="p", snapshot_id="s", token_ids=[1,2])
    assert compare_previous(result, previous)["exact_previous_prefix"]
    result.prompt_sha256 = "other"
    with pytest.raises(ValueError, match="prompt"):
        compare_previous(result, previous)


def test_previous_budget_publication_is_protected(tmp_path):
    old = tmp_path / "results/asd-budget/provenance.json"
    old.parent.mkdir(parents=True)
    old.write_text("frozen")
    ignored = tmp_path / "results/raw/new.json"
    ignored.parent.mkdir()
    ignored.write_text("new")
    assert set(protected(tmp_path)) == {"results/asd-budget/provenance.json"}
