"""Complete request matrix; natural paths and equal-work prefixes kept separate."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_suite import score
from memorydraft.evaluation.long_context import ARMS, STUDY, check_input, load_data, timing
from scripts.validate_asd_runtime import ADAPTER_SHA

ROOT = Path(__file__).resolve().parents[1]


def analyze(rows,references,items,tasks):
    by_id = {r["request"]["id"]:r for r in items}
    task_map = {r["id"]:r for r in tasks}
    expected = {(rid,arm,stage,repeat) for rid in by_id for arm in ARMS for stage,repeats in (("natural",2),("prefix",1)) for repeat in range(repeats)}
    keys = [(r["request_id"],r["arm"],r["stage"],r["repeat"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:raise ValueError("Incomplete or duplicated timing matrix")
    refs = {r["request_id"]:r for r in references}
    if len(references) != len(by_id) or set(refs) != set(by_id):raise ValueError("Incomplete AR reference matrix")
    grouped = defaultdict(list)
    strict_mismatches,new_graphs = [],0
    for row in [*rows,*references]:check_input(row,by_id[row["request_id"]])
    for row in rows:
        item = by_id[row["request_id"]]
        expected_method = "DFlash" if row["arm"] == "DFlash" else "ASD"
        if row["method"] != expected_method or row["approximate"] != (expected_method == "ASD"):
            raise ValueError("Method metadata changed")
        if row["draft_adapter_sha256"] != (ADAPTER_SHA if row["arm"] == "position-ASD" else None):
            raise ValueError("Adapter identity changed")
        if row["stage"] == "natural" and row["quality"] != score(task_map[item["task_id"]],row["text"]):
            raise ValueError("Quality score differs from frozen task")
        if row["arm"] == "DFlash" and row["token_ids"] != refs[row["request_id"]]["token_ids"][:(4096 if row["stage"] == "natural" else 128)]:
            strict_mismatches.append([row["request_id"],row["stage"],row["repeat"]])
        new_graphs += row["counters"]["compile_graphs_after"]-row["counters"]["compile_graphs_before"]
        grouped[row["request_id"],row["arm"],row["stage"]].append(row)
    cases = []
    for rid,item in by_id.items():
        arms = {}
        for arm in ARMS:
            for stage in ("natural","prefix"):
                entries = grouped[rid,arm,stage]
                first = entries[0]
                if any((r["token_ids"],r["finish_reason"],r["counters"]["rounds"]) !=
                       (first["token_ids"],first["finish_reason"],first["counters"]["rounds"]) for r in entries):
                    raise ValueError("Repeated output/verification trajectory changed")
                values = [timing(r) for r in entries]
                arms[arm+"/"+stage] = dict(seconds=statistics.mean(v["seconds"] for v in values),
                    ttft_seconds=statistics.mean(v["ttft_seconds"] for v in values),post_first_seconds=statistics.mean(v["post_first_seconds"] for v in values),
                    tokens=values[0]["tokens"],rounds=values[0]["rounds"],finish_reason=first["finish_reason"],
                    quality=first["quality"],samples=[v["seconds"] for v in values])
        cases.append(dict(request_id=rid,task_id=item["task_id"],level=item["level"],prompt_tokens=item["prompt_tokens"],examples=item["examples"],arms=arms))
    summary = {}
    for level in sorted({r["level"] for r in items}):
        selected = [r for r in cases if r["level"] == level]
        common = [r for r in selected if all(r["arms"][arm+"/prefix"]["tokens"] == 128 for arm in ARMS)]
        stats = {}
        for arm in ARMS:
            natural = [r["arms"][arm+"/natural"] for r in selected]
            ratio = lambda group,stage,metric:statistics.mean(r["arms"][arm+"/"+stage][metric]/r["arms"]["ASD/"+stage][metric] for r in group) if group else None
            stats[arm] = dict(requests=len(natural),correct=sum(r["quality"]["correct"] for r in natural),
                natural_ratio_to_ar=statistics.mean(r["arms"][arm+"/natural"]["seconds"]/refs[r["request_id"]]["latency_seconds"] for r in selected),
                truncated=sum(r["finish_reason"] != "eos" for r in natural),tokens=sum(r["tokens"] for r in natural),rounds=sum(r["rounds"] for r in natural),
                natural_ratio_to_asd=ratio(selected,"natural","seconds"),post_first_ratio_to_asd=ratio(selected,"natural","post_first_seconds"),
                mean_ttft_seconds=statistics.mean(r["ttft_seconds"] for r in natural),mean_post_first_seconds=statistics.mean(r["post_first_seconds"] for r in natural),
                common_full_128=len(common),prefix_ratio_to_asd=ratio(common,"prefix","seconds"),
                prefix_post_first_ratio_to_asd=ratio(common,"prefix","post_first_seconds"),
                lost_vs_asd=[r["task_id"] for r in selected if r["arms"]["ASD/natural"]["quality"]["correct"] and not r["arms"][arm+"/natural"]["quality"]["correct"]],
                gained_vs_asd=[r["task_id"] for r in selected if not r["arms"]["ASD/natural"]["quality"]["correct"] and r["arms"][arm+"/natural"]["quality"]["correct"]])
        summary[str(level)] = stats
    return dict(rows=len(rows),references=len(refs),all_strict_match_ar=not strict_mismatches,strict_mismatches=strict_mismatches,
                formal_new_graphs=new_graphs,cases=cases,summary=summary)


def main():
    tasks,items,_ = load_data(ROOT)
    base = ROOT/f"results/{STUDY}"
    read = lambda path:json.loads(path.read_text(encoding="utf-8"))
    execution = read(base/"execution-verification.json")
    if execution["status"] != "complete":raise ValueError("Incomplete GPU execution")
    levels = read(base/"admission.json")["levels"]
    items = [r for r in items if r["level"] in levels]
    rows,refs,workers,diagnostics = [],[],[],[]
    for variant in ("base","position"):
        for shard in (0,1):
            directory = base/f"eval-{variant}-{shard}"
            meta = read(directory/"provenance.json")
            if (meta["status"] != "complete" or meta["compiler"]["allow_tf32"] or meta["peak_gpu_bytes"] > 46*2**30
                    or meta["compiler"]["float32_matmul_precision"] != "highest" or meta["compiler"]["graph_count"] > 8
                    or meta["compiler"]["backend_seconds"] > 300 or meta["cache_bytes"] > 4*2**30):
                raise ValueError("Worker status/precision/resources failed")
            workers.append(meta)
            rows.extend(json.loads(line) for line in (directory/"rows.jsonl").read_text(encoding="utf-8").splitlines())
            if variant == "base":refs.extend(json.loads(line) for line in (directory/"references.jsonl").read_text(encoding="utf-8").splitlines())
            if (directory/"diagnostics.jsonl").exists():diagnostics.extend(json.loads(line) for line in (directory/"diagnostics.jsonl").read_text(encoding="utf-8").splitlines())
    result = dict(source_commit=execution["commit"],**analyze(rows,refs,items,tasks),workers=workers,diagnostics=diagnostics,
                  note="Already observed eight GSM tasks with new few-shot inputs; number/content of demonstrations changes with length; no new blind-set claim")
    (base/"analysis.json").write_bytes((json.dumps(result,indent=2)+"\n").encode())
    lines = ["# 数学长上下文小规模评估", "", "八道已观察GSM题，使用公开训练集完整解题示例构造三个输入长度。各方法在每个条件下输入完全相同；不同长度也改变示例数量/内容，不当作纯长度因果实验。自然生成两次、128/EOS前缀一次，AR完整参考另列。", "",
             f"GPU源码`{execution['commit']}`；{result['rows']}正式结果、{result['references']} AR参考；strict逐token匹配AR：{result['all_strict_match_ar']}；正式新编译图{result['formal_new_graphs']}。", "",
             "|输入名义token|方法|正确/8|自然时延/ASD|自然时延/AR|首token后时延/ASD|首token秒|自然输出token|验证调用|共同128题|128时延/ASD|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for level,arms in result["summary"].items():
        for arm,s in arms.items():
            prefix = "—" if s["prefix_ratio_to_asd"] is None else f"{s['prefix_ratio_to_asd']:.4f}"
            lines.append(f"|{level}|{arm}|{s['correct']}/8|{s['natural_ratio_to_asd']:.4f}|{s['natural_ratio_to_ar']:.4f}|{s['post_first_ratio_to_asd']:.4f}|{s['mean_ttft_seconds']:.3f}|{s['tokens']}|{s['rounds']}|{s['common_full_128']}|{prefix}|")
    lines += ["", "比例为题均配对比；小于1表示耗时更低。自然输出长度/路径可能不同，首token后时延不是固定工作量吞吐。共同128仅纳入三方法都自然到达128 token的相同题目，不强制越过EOS。正确数不把计时重复当新增样本；截断、逐题长度/轮数和新增错答在analysis.json完整保留。", "",
              "TTFT包含请求内tokenize、Target预填充、首token处理；latency−TTFT是首token后的剩余请求耗时，包含首次Draft消费上下文，不称为纯decode kernel时间。主时延包含engine.generate内全部推理链路成本；离线数据准备、请求文件读取、模型加载、HTTP和排队未计入，不能称完整服务端到端延迟。模型加载/冷请求和profiler诊断另列。", "",
              "独立诊断在全部正式计时后执行，仅报告聚合CPU/GPU归属，不发布大型trace，也不把profiler时间作为性能结果。target_prefill和draft_first_forward分别包含Target预填充、Draft首次上下文及候选计算；GPU工作量与host等待不能相加当wall时间。", "",
              "数据/资源/配对规则见[协议](ASD_LONG_CONTEXT_PROTOCOL.md)。未训练新权重、未补seed、未生成reserve96答案，也未接入连续批处理。原研究收益及独立面试版保持。", ""]
    lines += ["", "|输入名义token|配置|方法|Target预填充GPU ms|Draft首次forward GPU ms|Target验证GPU ms|", "|---|---|---|---:|---:|---:|"]
    for d in diagnostics:
        if d.get("status") == "failed":
            lines.append(f"诊断失败：{d['variant']}/{d['level']}/{d['method']}：{d['error']}")
            continue
        gpu = lambda name:d["scopes"].get(name,{}).get("linked_device_us",0)/1000
        lines.append(f"|{d['level']}|{d['variant']}|{d['method']}|{gpu('target_prefill'):.3f}|{gpu('draft_first_forward'):.3f}|{gpu('target_verify'):.3f}|")
    (ROOT/"docs/ASD_LONG_CONTEXT_RESULTS.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(dict(rows=result["rows"],references=result["references"],strict_match=result["all_strict_match_ar"],formal_new_graphs=result["formal_new_graphs"])))


if __name__ == "__main__":
    main()
