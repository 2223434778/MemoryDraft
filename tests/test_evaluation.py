import pytest

from memorydraft.evaluation.replay import summarize


def row(method, request_id, seconds):
    return {"method": method, "request_id": request_id, "repeat": 0,
            "latency_seconds": seconds, "correct": True, "token_ids": [1, 2]}


def test_paired_total_time_ratio():
    result = summarize([row("AR", "a", 10), row("D", "a", 5), row("AR", "b", 1), row("D", "b", 2)])
    assert result["D"]["speedup_vs_ar"] == 11 / 7
    assert result["D"]["all_correct"]


def test_unpaired_data_cannot_claim_speedup():
    with pytest.raises(ValueError, match="Unpaired"):
        summarize([row("D", "a", 1)])
