from copy import deepcopy

import pytest

from scripts.write_asd_shared_prefix_report import analyze_http


def fixture():
    items = [dict(task_id=str(i),level=4096,prompt_tokens=4000,prompt_sha256=str(i)) for i in range(4)]
    groups = {}
    for shard in (0,1):
        groups[shard] = {}
        for variant in ("whole","shared"):
            states = ("cache",) if variant == "whole" else ("off","cache")
            meta = dict(status="complete",json_and_sse_match=True,disconnect_cancelled=True,recovered=True)
            rows = []
            for state in states:
                hits = 19 if state == "cache" and (variant == "shared" or shard == 0) else 0
                meta[state] = dict(requests=20,tokens=20,hits=hits,wall_seconds=10. if hits else 20.)
                for i in range(20):
                    item = items[shard::2][i%2]
                    c = dict(rounds=1,compile_graphs_before=3,compile_graphs_after=3)
                    if state == "cache":c.update(prefix_cache_bytes=1000000000,prefix_cache_prefix_tokens=3934 if variant == "shared" else 3937,
                        prefix_cache_hit=int(i > 0 and hits > 0),prefix_cache_evictions=int(i > 0 and hits == 0),
                        prefix_cache_filled_bytes=1000000000 if i == 0 or hits == 0 else 0)
                    result = dict(request_id="r",prompt_tokens=4000,prompt_sha256=item["prompt_sha256"],
                        snapshot_id=item["task_id"],method="ASD",approximate=True,execution_profile="rmsnorm-inductor",
                        draft_adapter_sha256=None,asd=dict(budget=8,spent=1,gate=.25,cap=2),
                        token_ids=[7],text="7",finish_reason="eos",queue_seconds=1.,server_seconds=3.,peak_gpu_bytes=1,counters=c)
                    rows.append(dict(state=state,index=i,task_id=item["task_id"],result=result))
            groups[shard][variant] = meta,rows
    return groups,items


def test_same_stream_comparison_keeps_hit_rate_evictions_and_cold_cost():
    result = analyze_http(*fixture())
    assert result["0"]["shared_to_whole_wall_ratio"] == 1
    assert result["1"]["shared_to_whole_wall_ratio"] == .5
    assert result["1"]["whole/cache"]["evictions"] == 19
    assert result["1"]["shared/cache"]["filled_bytes"] == 1000000000
    assert result["1"]["shared/cache"]["hits"] == 19


def test_different_output_is_retained_without_a_paired_speed_claim():
    groups,items = fixture()
    groups[0]["whole"][1][0]["result"]["token_ids"] = [8]
    result = analyze_http(groups,items)
    assert not result["0"]["outputs_and_aggregate_counters_equal"]
    assert result["0"]["shared_to_whole_wall_ratio"] is None


@pytest.mark.parametrize("bad",["missing","budget","sequence","summary","compile"])
def test_invalid_http_records_cannot_be_published(bad):
    groups,items = deepcopy(fixture())
    meta,rows = groups[0]["shared"]
    if bad == "missing":rows.pop()
    elif bad == "budget":rows[-1]["result"]["asd"]["spent"] = 8.0001
    elif bad == "sequence":rows[-1]["task_id"] = "wrong"
    elif bad == "summary":meta["cache"]["hits"] = 0
    elif bad == "compile":rows[-1]["result"]["counters"]["compile_graphs_after"] = 4
    with pytest.raises(ValueError):analyze_http(groups,items)
