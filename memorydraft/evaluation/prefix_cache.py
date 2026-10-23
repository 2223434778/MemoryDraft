"""Compare public-prefix reuse without treating numerical drift as equality."""
import math
from statistics import mean

from memorydraft.evaluation.long_context import check_input, timing

STUDY = "asd-prefix-cache"
STATES = ("off", "cold", "hit")


def compare(reference, row):
    fields = ("token_ids","text","finish_reason","prompt_sha256","prompt_tokens","snapshot_id", "execution_profile")
    changed = [k for k in fields if reference[k] != row[k]]
    if reference["counters"]["rounds"] != row["counters"]["rounds"]:changed.append("rounds")
    a,b = reference.get("asd"),row.get("asd")
    delta = None
    if a is None or b is None:
        if a != b:changed.append("asd")
    else:
        if {k:v for k,v in a.items() if k != "spent"} != {k:v for k,v in b.items() if k != "spent"}:
            changed.append("asd_decisions")
        delta = b["spent"]-a["spent"]
    return dict(changed=changed,spent_delta=delta,spent_bitwise_equal=delta in (None,0.))


def analyze(rows, items):
    lookup = {(r["task_id"],r["level"]):r for r in items}
    expected = {(task,level,method,stage,state) for task,level in lookup
        for method in ("DFlash","ASD") for stage in ("natural","prefix") for state in STATES}
    actual = [(r["task_id"],r["level"],r["method"],r["stage"],r["cache_state"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:raise ValueError("Incomplete prefix cache matrix")
    cases = []
    for key,item in lookup.items():
        for method in ("DFlash","ASD"):
            for stage in ("natural","prefix"):
                group = {r["cache_state"]:r for r in rows if (r["task_id"],r["level"],r["method"],r["stage"]) == (*key,method,stage)}
                for state,row in group.items():
                    check_input(row,item)
                    timing(row)
                    c = row["counters"]
                    if row["approximate"] != (method == "ASD") or row["draft_adapter_sha256"] is not None:
                        raise ValueError("Changed cache experiment method/adapter")
                    if c["compile_graphs_after"] != c["compile_graphs_before"]:
                        raise ValueError("New compilation in formal sample")
                    if state != "off" and (c["prefix_cache_hit"] != (state == "hit")
                            or not 0 < c["prefix_cache_prefix_tokens"] < row["prompt_tokens"]
                            or c["prefix_cache_bytes"] > 2*2**30):
                        raise ValueError("Invalid prefix cache accounting")
                off = group["off"]
                pairs = {state:compare(off,group[state]) for state in ("cold","hit")}
                cases.append(dict(task_id=key[0],level=key[1],method=method,stage=stage,
                    timing={s:timing(r) for s,r in group.items()},comparisons=pairs,
                    prefix_tokens=group["hit"]["counters"]["prefix_cache_prefix_tokens"],
                    cache_bytes=group["hit"]["counters"]["prefix_cache_bytes"]))
    summary = {}
    for level in sorted({r["level"] for r in items}):
        for method in ("DFlash","ASD"):
            for stage in ("natural","prefix"):
                subset = [r for r in cases if (r["level"],r["method"],r["stage"]) == (level,method,stage)]
                eligible = [r for r in subset if not any(p["changed"] for p in r["comparisons"].values())]
                record = dict(requests=len(subset),same_output_and_decisions=len(eligible))
                if eligible:
                    record["seconds"] = {s:mean(r["timing"][s]["seconds"] for r in eligible) for s in STATES}
                    record["ttft_seconds"] = {s:mean(r["timing"][s]["ttft_seconds"] for r in eligible) for s in STATES}
                    record["hit_ratio"] = mean(r["timing"]["hit"]["seconds"]/r["timing"]["off"]["seconds"] for r in eligible)
                    record["cold_ratio"] = mean(r["timing"]["cold"]["seconds"]/r["timing"]["off"]["seconds"] for r in eligible)
                    record["cold_plus_one_hit_ratio"] = sum(r["timing"][s]["seconds"] for r in eligible for s in ("cold","hit"))/(2*sum(r["timing"]["off"]["seconds"] for r in eligible))
                    record["all_full_128"] = stage == "prefix" and all(r["timing"][s]["tokens"] == 128 for r in eligible for s in STATES)
                    full = [r for r in eligible if stage == "prefix" and all(r["timing"][s]["tokens"] == 128 for s in STATES)]
                    record["common_full_128"] = len(full)
                    record["common_full_128_hit_ratio"] = mean(r["timing"]["hit"]["seconds"]/r["timing"]["off"]["seconds"] for r in full) if full else None
                summary[f"{level}/{method}/{stage}"] = record
    differences = [dict(task_id=r["task_id"],level=r["level"],method=r["method"],stage=r["stage"],state=s,**p)
                   for r in cases for s,p in r["comparisons"].items() if p["changed"]]
    deltas = [abs(p["spent_delta"]) for r in cases for p in r["comparisons"].values() if p["spent_delta"] is not None]
    if not all(math.isfinite(d) for d in deltas):raise ValueError("Nonfinite ASD spending")
    return dict(rows=len(rows),cases=cases,summary=summary,differences=differences,
                all_outputs_and_decisions_equal=not differences,max_spent_abs_delta=max(deltas,default=0.),
                spent_bitwise_differences=sum(not p["spent_bitwise_equal"] for r in cases for p in r["comparisons"].values()),
                resource_peak_gpu_bytes=max(r["peak_gpu_bytes"] for r in rows))
