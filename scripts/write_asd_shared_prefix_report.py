"""Matched shared/whole-prefix HTTP streams; keep cold costs and EOS separate."""
import argparse
import json
from pathlib import Path

from memorydraft.evaluation.long_context import check_input, load_data
from memorydraft.evaluation.prefix_cache import analyze, compare
from memorydraft.evaluation.runtime_migration import read_rows

ROOT = Path(__file__).resolve().parents[1]
STUDY = "asd-shared-prefix"


def analyze_http(groups,items):
    results = {}
    for shard in (0,1):
        subset = items[shard::2]
        # items retain the original task order, filtered to 4K before this call.
        policies = groups[shard]
        by = {}
        summary = {}
        for variant in ("whole","shared"):
            meta,rows = policies[variant]
            expected_states = ("cache",) if variant == "whole" else ("off","cache")
            expected = {(s,i) for s in expected_states for i in range(20)}
            actual = [(r["state"],r["index"]) for r in rows]
            if len(actual) != len(expected) or set(actual) != expected:raise ValueError("Incomplete HTTP comparison")
            if meta["status"] != "complete" or not all(meta[k] for k in ("json_and_sse_match","disconnect_cancelled","recovered")):
                raise ValueError("HTTP acceptance failed")
            for r in rows:
                row = r["result"]
                item = subset[r["index"]%len(subset)]
                if r["task_id"] != item["task_id"]:raise ValueError("HTTP task sequence changed")
                check_input(row,item)
                if (row["method"] != "ASD" or not row["approximate"] or row["draft_adapter_sha256"] is not None
                        or row["asd"]["gate"] != .25 or row["asd"]["cap"] != 2
                        or row["peak_gpu_bytes"] > 46*2**30
                        or row["counters"]["compile_graphs_before"] != row["counters"]["compile_graphs_after"]
                        or not 0 <= row["queue_seconds"] <= row["server_seconds"]):
                    raise ValueError("HTTP profile/queue accounting changed")
                c = row["counters"]
                if c.get("prefix_cache_bytes",0) > 2*2**30:raise ValueError("HTTP cache capacity exceeded")
                if r["state"] == "cache" and variant == "shared" and c["prefix_cache_prefix_tokens"] != 3934:
                    raise ValueError("Unexpected shared prefix")
                if r["state"] == "cache" and variant == "whole" and c["prefix_cache_prefix_tokens"] not in (3937,4073):
                    raise ValueError("Whole-prefix control changed registry")
                by[variant,r["state"],r["index"]] = row
            for state in expected_states:
                selected = [r["result"] for r in rows if r["state"] == state]
                count = meta[state]
                if (count["requests"] != 20 or count["tokens"] != sum(len(r["token_ids"]) for r in selected)
                        or count["hits"] != sum(r["counters"].get("prefix_cache_hit",0) for r in selected)):
                    raise ValueError("HTTP aggregate differs from raw records")
                summary[f"{variant}/{state}"] = dict(count,
                    evictions=sum(r["counters"].get("prefix_cache_evictions",0) for r in selected),
                    filled_bytes=sum(r["counters"].get("prefix_cache_filled_bytes",0) for r in selected),
                    peak_cache_bytes=max(r["counters"].get("prefix_cache_bytes",0) for r in selected))
        pairs = [dict(index=i,variant=v,**compare(by["shared","off",i],by[v,"cache",i]))
                 for i in range(20) for v in ("whole","shared")]
        same = not any(r["changed"] for r in pairs)
        summary.update(outputs_and_aggregate_counters_equal=same,comparisons=pairs,
            shared_to_whole_wall_ratio=summary["shared/cache"]["wall_seconds"]/summary["whole/cache"]["wall_seconds"] if same else None,
            shared_to_off_wall_ratio=summary["shared/cache"]["wall_seconds"]/summary["shared/off"]["wall_seconds"] if same else None)
        results[str(shard)] = summary
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/f"results/{STUDY}")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    directory = args.input
    _,items,_ = load_data(ROOT)
    items = [r for r in items if r["level"] == 4096]
    rows = [r for i in (0,1) for r in read_rows(directory/f"eval-shared-{i}/rows.jsonl")]
    result = analyze(rows,items)
    groups = {i:{v:(json.loads((directory/f"eval-{v}-{i}/http-summary.json").read_text(encoding="utf-8")),
                    list(read_rows(directory/f"eval-{v}-{i}/http-load.jsonl"))) for v in ("whole","shared")} for i in (0,1)}
    for i in (0,1):
        for v in ("whole","shared"):
            meta = json.loads((directory/f"eval-{v}-{i}/provenance.json").read_text(encoding="utf-8"))
            config = meta["runtime"]["engine"]["config"]
            expected = json.loads((ROOT/("configs/runtime-dflash-asd-shared-prefix.json" if v == "shared"
                                         else "configs/runtime-dflash-asd-prefix-cache.json")).read_text(encoding="utf-8"))
            if (meta["status"] != "complete" or config["prefix_cache"]["max_bytes"] != 2*2**30
                    or config["prefix_cache"]["max_entries"] != 3
                    or config["prefix_cache"]["sha256"] != expected["prefix_cache"]["sha256"]
                    or config["asd"] != dict(budget=8,gate=.25,cap=2)):
                raise ValueError("Worker changed registry, cache budget or ASD configuration")
    result["http"] = analyze_http(groups,items)
    execution = json.loads((directory/"execution-verification.json").read_text(encoding="utf-8"))
    result.update(source_commit=execution["commit"],execution_status=execution["status"],
        all_strict_match_previous_ar=all(r["strict_token_match_ar"] for r in rows if r["method"] == "DFlash"),
        comparison_to_previous_differences=[dict(request_id=r["request_id"],method=r["method"],stage=r["stage"],
            cache_state=r["cache_state"],**r["comparison_to_previous"]) for r in rows if r["comparison_to_previous"]["changed"]],
        quality={f"{method}/{state}":sum(r["quality"]["correct"] for r in rows if
            (r["method"],r["cache_state"],r["stage"]) == (method,state,"natural"))
            for method in ("DFlash","ASD") for state in ("off","cold","hit")},
        counter_scope="GenerationResult totals, not a per-round trace",formal_new_graphs=0)
    output = args.output or directory
    output.mkdir(parents=True,exist_ok=True)
    (output/"analysis.json").write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+"\n").encode())
    lines = ["# 共享公共段缓存结果", "",f"执行源 `{execution['commit']}`；96正式生成及120个HTTP负载请求。",
        "一致性指输出、结束、输入/快照、验证总轮数及ASD汇总量；未导出逐轮trace。", "",
        "| 方法 | 阶段 | off / cold / hit秒 | hit/off题均比 | 首token off / hit秒 | 共同128题数/比 |",
        "|---|---|---|---|---|---|"]
    for key,r in result["summary"].items():
        _,method,stage = key.split("/")
        if "seconds" not in r:continue
        seconds = " / ".join(f"{r['seconds'][s]:.4f}" for s in ("off","cold","hit"))
        full = f"{r['common_full_128']}/"+(f"{r['common_full_128_hit_ratio']:.4f}" if r["common_full_128_hit_ratio"] is not None else "—")
        lines.append(f"|{method}|{stage}|{seconds}|{r['hit_ratio']:.4f}|{r['ttft_seconds']['off']:.4f} / {r['ttft_seconds']['hit']:.4f}|{full}|")
    lines += ["", "| GPU/同卡请求流 | off / whole / shared总wall秒 | whole / shared命中 | whole / shared淘汰 | shared/whole | shared/off |",
        "|---|---|---|---|---|---|"]
    for i,r in result["http"].items():
        a,b,c = r["shared/off"],r["whole/cache"],r["shared/cache"]
        ratios = (f"{r['shared_to_whole_wall_ratio']:.4f}|{r['shared_to_off_wall_ratio']:.4f}"
                  if r["outputs_and_aggregate_counters_equal"] else "不同输出不比较|不同输出不比较")
        lines.append(f"|{i}|{a['wall_seconds']:.4f} / {b['wall_seconds']:.4f} / {c['wall_seconds']:.4f}|{b['hits']} / {c['hits']}|{b['evictions']} / {c['evictions']}|{ratios}|")
    lines += ["",f"ASD spent最大配对绝对差 {result['max_spent_abs_delta']:.12g}，实际预算保持8。",
        "4K原8个已观察任务，单格一次测量；自然遵守EOS，128子集单列。各HTTP窗口20次、并发2、四题轮换、从空缓存起步；whole/shared执行先后在两卡交错，shared内部仍先off后cache。不是生产SLA/continuous batching。",
        "只将3934-token公共段统一登记，Target/Draft/接受规则不变；未实现通用块级或分页KV存储。", ""]
    (output/"results.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(dict(rows=len(rows),same_outputs=result["all_outputs_and_decisions_equal"],http_shards=len(result["http"]))))


if __name__ == "__main__":
    main()
