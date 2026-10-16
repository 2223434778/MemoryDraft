import copy

import pytest

from scripts.write_research_report import speed_interval


def rows():
    result = []
    for request, seconds in [("a", 1), ("b", 4)]:
        for repeat in range(3):
            for method, scale in [("AR", 1), ("D", 0.5)]:
                result.append(dict(request_id=request, repeat=repeat, method=method, correct=True,
                                   token_ids=[1, 2], target_prompt_sha256="fixed", snapshot_id="same",
                                   latency_seconds=seconds * scale))
    return result


def test_paired_bootstrap_keeps_repeats_with_their_request():
    assert speed_interval(rows(), "D") == (2.0, 2.0)


def test_report_rejects_duplicate_pair_in_place_of_missing_pair():
    data = rows()
    data[-1] = copy.deepcopy(data[1])
    with pytest.raises(ValueError, match="paired"):
        speed_interval(data, "D")


def test_report_rejects_changed_target_even_with_equal_tokens():
    data = rows()
    data[1]["target_prompt_sha256"] = "changed"
    with pytest.raises(ValueError, match="Target"):
        speed_interval(data, "D")
