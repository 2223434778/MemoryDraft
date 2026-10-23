from copy import deepcopy

import pytest

from scripts.run_asd_comparison_engineering import analyze,candidate


def rows():
    return [dict(request_id="fixture",arm=arm,cache_state=s,repeat=i,same_input=True,strict_token_match=True,
        token_ids=[1,2],text="fixture",finish_reason="eos",asd=None,quality=dict(correct=True),
        counters=dict(prefix_cache_hit=int(s=="hit")),latency_seconds=1. if s != "hit" else .5,ttft_seconds=.2)
        for arm in ("base/DFlash","base/ASD","base/ASD-Utility") for s in ("off","cold","hit") for i in range(2)]


def test_engineering_is_matched_and_does_not_multiply_old_percentages():
    result = analyze(rows(),("base/DFlash","base/ASD","base/ASD-Utility"))
    assert result["summary"]["base/ASD-Utility"]["hit_off_ratio"] == .5
    assert result["summary"]["base/ASD-Utility"]["vs_asd"]["hit"] == 1


@pytest.mark.parametrize("change",["drop","duplicate","input","strict","hit","budget","time"])
def test_engineering_rejects_incomplete_or_invalid_evidence(change):
    data = rows()
    if change == "drop":data.pop()
    elif change == "duplicate":data.append(deepcopy(data[0]))
    elif change == "input":data[0]["same_input"] = False
    elif change == "strict":data[0]["strict_token_match"] = False
    elif change == "hit":data[-1]["counters"]["prefix_cache_hit"] = 0
    elif change == "budget":data[0]["asd"] = dict(spent=9)
    else:data[0]["latency_seconds"] = float("nan")
    with pytest.raises(ValueError):analyze(data,("base/DFlash","base/ASD","base/ASD-Utility"))


def test_cache_output_change_is_reported_without_hiding_the_request():
    data = rows()
    data[-1]["token_ids"] = [9]
    result = analyze(data,("base/DFlash","base/ASD","base/ASD-Utility"))
    assert not result["summary"]["base/ASD-Utility"]["all_outputs_equal"]
    assert result["requests"] == 1


def test_engineering_candidate_uses_confirmation_and_quality_first():
    analysis = dict(confirm=dict(families=dict(gsm8k={
        "base/DFlash":{},"base/ASD":{},
        "base/ASD-Utility":dict(natural={"base/ASD":dict(lost=1,new_truncations=0,ratio=.5)}),
        "ordinary-position/ASD":dict(natural={"base/ASD":dict(lost=0,new_truncations=0,ratio=.99)})})))
    assert candidate(analysis) == "ordinary-position/ASD"
