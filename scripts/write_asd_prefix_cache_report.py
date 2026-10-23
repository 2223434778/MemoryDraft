"""Report paired cache states and separate end-to-end queued HTTP observations."""
import argparse
import json
from pathlib import Path

from memorydraft.evaluation.long_context import load_data
from memorydraft.evaluation.prefix_cache import STUDY, analyze
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.runtime.benchmark import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/f"results/{STUDY}")
    parser.add_argument("--output",type=Path,default=None)
    args = parser.parse_args()
    directory = args.input
    _,items,_ = load_data(ROOT)
    rows = [r for i in (0,1) for r in read_rows(directory/f"eval-{i}/rows.jsonl")]
    result = analyze(rows,items)
    execution = json.loads((directory/"execution-verification.json").read_text())
    result["source_commit"] = execution["commit"]
    result["execution_status"] = execution["status"]
    result["comparison_to_previous_differences"] = [dict(request_id=r["request_id"],method=r["method"],
        stage=r["stage"],cache_state=r["cache_state"],**r["comparison_to_previous"])
        for r in rows if r["comparison_to_previous"]["changed"]]
    result["all_strict_match_previous_ar"] = all(r["strict_token_match_ar"] for r in rows if r["method"] == "DFlash")
    result["quality"] = {f"{level}/{method}/{state}":sum(r["quality"]["correct"] for r in rows if
        (r["level"],r["method"],r["cache_state"],r["stage"]) == (level,method,state,"natural"))
        for level in (512,2048,4096) for method in ("DFlash","ASD") for state in ("off","cold","hit")}
    result["http"] = {str(i):json.loads((directory/f"eval-{i}/http-summary.json").read_text())
                       for i in (0,1) if (directory/f"eval-{i}/http-summary.json").exists()}
    result["cross_question"] = {str(i):[dict(request_id=r["request_id"],**r["counters"]) for r in json.loads(
        (directory/f"eval-{i}/cross-question.json").read_text())] for i in (0,1) if (directory/f"eval-{i}/cross-question.json").exists()}
    dest = args.output or directory
    dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/"analysis.json",result)
    lines = ["# 公共前缀 KV / DFlash 特征缓存结果", "",
        f"GPU执行源 `{result['source_commit']}`；{len(rows)}正式生成，状态 {execution['status']}。",
        f"输出/决策一致：{result['all_outputs_and_decisions_equal']}；strict匹配旧AR：{result['all_strict_match_previous_ar']}。",
        f"ASD spent非逐bit相同 {result['spent_bitwise_differences']} 对，最大绝对差 {result['max_spent_abs_delta']:.12g}；线上预算未改。", "",
        "| 输入 | 方法 | 阶段 | 请求/同输出决策 | off / cold / hit秒 | hit/off | cold/off | (cold+hit)/(2off) | off / hit TTFT秒 | 共同128题数/比 |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for key,r in result["summary"].items():
        level,method,stage = key.split("/")
        if "seconds" not in r:
            lines.append(f"|{level}|{method}|{stage}|{r['requests']}/{r['same_output_and_decisions']}|不比较不同输出||||||")
            continue
        seconds = " / ".join(f"{r['seconds'][s]:.4f}" for s in ("off","cold","hit"))
        full = f"{r['common_full_128']}/"+(f"{r['common_full_128_hit_ratio']:.4f}" if r["common_full_128_hit_ratio"] is not None else "—")
        lines.append(f"|{level}|{method}|{stage}|{r['requests']}/{r['same_output_and_decisions']}|{seconds}|{r['hit_ratio']:.4f}|{r['cold_ratio']:.4f}|{r['cold_plus_one_hit_ratio']:.4f}|{r['ttft_seconds']['off']:.4f} / {r['ttft_seconds']['hit']:.4f}|{full}|")
    lines += ["", "自然为完整 EOS；prefix 为最多128且遵守EOS。每格仅一次测量；已观察工作负载，不是独立盲测。",
        "cold包含缓存写入，hit包含复制和查找。冷+一次命中是配对成本估算；不是另测的到达轨迹。", "",
        "| HTTP GPU | off / cache请求数 | off / cache总wall秒 | 启用cache命中数 | off / cache平均客户端秒 | off / cache平均排队秒 |",
        "|---|---|---|---|---|---|"]
    for i,h in result["http"].items():
        if h["status"] != "complete":continue
        a,b = h["off"],h["cache"]
        lines.append(f"|{i}|{a['requests']} / {b['requests']}|{a['wall_seconds']:.4f} / {b['wall_seconds']:.4f}|{b['hits']}|{a['mean_client_seconds']:.4f} / {b['mean_client_seconds']:.4f}|{a['mean_queue_seconds']:.4f} / {b['mean_queue_seconds']:.4f}|")
    lines += ["", "HTTP: 各卡20次关闭后20次启用、并发2、4道4K请求轮换、空缓存起步；包含实际冷 miss。顺序固定，保留次序混杂，不声称生产SLA/continuous batching。",
        "跨问题复用、JSON/SSE、断连取消及恢复的记录见原始JSON。完整数值/质量/旧轮差异在 analysis.json；任何差异均保留。", ""]
    (dest/"results.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(dict(rows=len(rows),outputs_equal=result["all_outputs_and_decisions_equal"],http_shards=len(result["http"]))))


if __name__ == "__main__":
    main()
