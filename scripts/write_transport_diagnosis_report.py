"""Audit and summarize captured decisions; no instrumented speed comparisons."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from memorydraft.evaluation.runtime_migration import input_hashes, read_rows
from memorydraft.evaluation.transport_diagnosis import accepted_prefix, digest, failure_class, first_divergence
from scripts.diagnose_transport_online import METHODS, inputs, write


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    path = Path(path)
    data = path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    return json.loads(data)


def rebuild(source, output, *, root=ROOT):
    source, output = Path(source), Path(output)
    info = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    rows = read_json(source / "decisions.json")
    comparisons = json.loads((source / "comparisons.json").read_text(encoding="utf-8"))
    tasks, _ = inputs(root)
    definitions = {t.request.id: t for t in tasks}
    expected = {(t.request.id, method) for t in tasks for method in METHODS}
    if (info["status"] != "complete" or info["correct_runs"] != 24 or not info["original_unchanged"]
            or info["original_sha256"] != {name: input_hashes(root / "results" / name) for name in info["original_sha256"]}
            or len(rows) != 24 or {(r["request_id"], r["method"]) for r in rows} != expected):
        raise ValueError("Missing runs or changed diagnostic source evidence")
    for row in rows:
        task = definitions[row["request_id"]]
        if (not row["correct"] or row["token_ids"] != task.reference["token_ids"]
                or row["output_sha256"] != digest(row["token_ids"])
                or row["prompt_sha256"] != task.reference["prompt_sha256"] or row["snapshot_id"] != task.snapshot_id):
            raise ValueError("Diagnostic differs from frozen Target output/input/snapshot")
        aligned = [c for c in row["calls"] if c["alignment_called"]]
        counters = row["counters"]
        if (counters["rounds"] != len(row["rounds"])
                or counters.get("alignment_calls", 0) != len(aligned)):
            raise ValueError("Invalid captured round/call counters")
        if sum(c["selected_accepted"] for c in aligned) != counters.get("alignment_accepted_tokens", 0):
            raise ValueError("Invalid alignment acceptance counters")
        for call in row["calls"]:
            position = call["generated_before"]
            if call["prefix_sha256"] != digest(row["token_ids"][:position]):
                raise ValueError("Changed captured prefix")
            if not call["alignment_called"]:
                continue
            remaining = row["token_ids"][position:]
            pool_best = max((accepted_prefix(c["continuation"], remaining) for c in call["candidates"]), default=0)
            proposal = call["proposal"]
            selected = accepted_prefix(proposal["tokens"], remaining) if proposal else 0
            if (pool_best != call["pool_best"] or selected != call["selected_accepted"]
                    or not call["raw_best"] >= call["eligible_best"] >= pool_best >= selected
                    or any(c["accepted"] != accepted_prefix(c["continuation"], remaining) for c in call["candidates"])
                    or call["classification"] != failure_class(call["raw_best"], call["eligible_best"], pool_best,
                                                                selected, proposal is not None)
                    or proposal is not None and call["actual_round"]["accepted"] != selected):
                raise ValueError("Candidate or actual-verification audit changed")
    if len(comparisons) != 8 or {c["request_id"] for c in comparisons} != set(definitions):
        raise ValueError("Missing first-divergence comparisons")
    for comparison in comparisons:
        selected = {r["method"]: r for r in rows if r["request_id"] == comparison["request_id"]}
        if (comparison["suffix_ot"] != first_divergence(selected["Suffix-R"]["rounds"], selected["OT-R"]["rounds"])
                or comparison["edit_ot"] != first_divergence(selected["Edit-R"]["rounds"], selected["OT-R"]["rounds"])):
            raise ValueError("Changed first-divergence comparison")
        proposals_equal = [c["proposal"]["tokens"] if c["proposal"] else None for c in selected["Edit-R"]["calls"]] == [
            c["proposal"]["tokens"] if c["proposal"] else None for c in selected["OT-R"]["calls"]]
        if proposals_equal != comparison["edit_ot_proposals_equal"]:
            raise ValueError("Changed proposal equality comparison")
    prior_rows = read_rows(root / "results/ot-validation/rows.jsonl") + read_rows(root / "results/ot-r/rows.jsonl")
    measured = {}
    counter_keys = ("rounds", "draft_rounds", "reuse_rounds", "reuse_accepted_tokens", "alignment_calls",
                    "alignment_candidates", "alignment_proposals", "alignment_accepted_tokens")
    for row in rows:
        original = [r for r in prior_rows if r["request_id"] == row["request_id"] and r["method"] == row["method"]]
        if len(original) != 2 or any(any(r["counters"].get(k, 0) != row["counters"].get(k, 0)
                                         for k in counter_keys) for r in original):
            raise ValueError("Instrumented counters differ from prior uninstrumented runs")
        measured[row["request_id"], row["method"]] = sum(r["latency_seconds"] for r in original)
    summary = {}
    for method in METHODS:
        chosen = [r for r in rows if r["method"] == method]
        calls = [c for r in chosen for c in r["calls"] if c["alignment_called"]]
        summary[method] = dict(calls=len(calls), proposals=sum(c["proposal"] is not None for c in calls),
                               accepted=sum(c["selected_accepted"] for c in calls),
                               classes=dict(Counter(c["classification"] for c in calls)),
                               rounds=sum(r["counters"]["rounds"] for r in chosen))
    write(source / "diagnostic.json", summary)
    lines = ["# OT-R 实际触发位置的错误定位", "", "2026-10-19。八个已看过请求×三种检索，共24次插桩回放；不是性能测试或独立场景验证。", "",
             "全部输出与既有AR的token、Target输入和合法快照严格一致。参考答案仅在生成结束后评分。", "",
             "## 对齐调用", "", "| 方法 | 调用 | 提议 | 对齐接受token | 全部解码轮 |", "|---|---:|---:|---:|---:|"]
    for method, s in summary.items():
        lines.append(f"| {method} | {s['calls']} | {s['proposals']} | {s['accepted']} | {s['rounds']} |")
    lines += ["", "## 实际对齐决策", "", "raw/pool是事后最好连续接续长度，>=8只是诊断分层，不是盈亏线。", "",
              "| 请求 | 方法 | 已生成token | raw最好 | pool最好 | 实际接受 | 分类 |", "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        for c in row["calls"]:
            if c["alignment_called"]:
                lines.append(f"| {row['request_id']} | {row['method']} | {c['generated_before']} | {c['raw_best']} | {c['pool_best']} | {c['selected_accepted']} | {c['classification']} |")
    lines += ["", "## Suffix与OT的首次分歧", "", "同请求仅比较首个分歧；之后KV/特征累积路径可能不同。advance包含一个Target确认token，EOS处最终发布长度可截断。", "",
              "| 请求 | 位置 | Suffix路线/接受/推进 | OT路线/接受/推进 |", "|---|---:|---|---|"]
    for c in comparisons:
        value = c["suffix_ot"]
        if value is not None:
            a, b = value["left"], value["right"]
            lines.append(f"| {c['request_id']} | {a['generated_before']} | {a['route']}/{a['accepted']}/{a['advance']} | {b['route']}/{b['accepted']}/{b['advance']} |")
    lines += ["", "## 原未插桩计时的逐请求拆分", "", "下表直接汇总原两次重复，不使用此次插桩耗时。新请求在本次诊断前已经观察，不作新正例场景确认。", "",
              "| 请求 | Suffix秒 | Edit秒 | OT秒 | OT相对Suffix延迟 |", "|---|---:|---:|---:|---:|"]
    for task in tasks:
        a, b, c = (measured[task.request.id, method] for method in METHODS)
        lines.append(f"| {task.request.id} | {a:.6f} | {b:.6f} | {c:.6f} | {(c/a-1)*100:+.2f}% |")
    lines += ["", "## 机制解释与最小改进方向", "",
              "1. 新五请求的六次OT调用：四次不存在>=8-token原始接续，一次候选池漏召回，一次选中32-token接续。主要损失发生在编辑尚未完成或输出临近结束时，不能归为OT把池内长候选排错。",
              "2. parcel-queue:0在新值72只生成7时提议旧值的8，首token即拒绝；同状态Suffix转DFlash接受15。应先让模型跨过新内容，再寻找恢复复制的位置。仅提高求解精度不会创造不存在的连续span。",
              "3. payment-runbook:1插入Region行后，真实Severity接续可接受32，但现有池全为0。当前倒排只记录端点>=min_context的位置，早期Owner不能作旧记录锚点；稀有新词Region/eu/-west只命中当前修改指令。固定偏移把候选推入Severity中间。",
              "4. 该单点离线反事实shortlist-analysis.json：只把候选数8扩大32仍没有长候选；max_shift从2扩大5后池内出现32-token接续，但OT和Edit仍选择0-token接续。它同时涉及召回和排序，不能靠单项扩容修复。此反事实不修改线上参数、不作速度证据。",
              "5. payment-runbook:0确实恢复32-token并在原计时中更快；项目整体负结果来自另一个请求的较大损失。不能据项目均值否认局部成功，也不能据单请求成功确认稳定适用场景。Edit同样成功且更便宜。",
              "6. confirm-schema-0:0在共同状态下OT接受32，Edit接受0。两端点Edit分数同为0.9375，按位置取更早端点；OT分别约0.934891/0.941865，选对后一个。旧局部正例有真实消歧机制，但不证明OT优于所有便宜排序器。",
              "7. 先改触发决策与编辑后端点召回，再研究接续排序；保留旧正例作回归。不能硬性要求末token精确相同：旧正例成功OT的exact_tail=0，简单提高末尾匹配要求会删掉它。延迟构造索引和求解器加速可降低成本，但不能替代上述机制改动。"]
    lines += ["", "## 限制与复算", "", "原未插桩性能证据见OT_R_RESULTS.md和OT_R_VALIDATION_RESULTS.md。本报告不从插桩耗时推导加速。",
              "候选池诊断包含记录/特殊token边界、用户时间快照、历史最小上下文与实际阻塞。oracle只说明该位置存在什么，不能作为线上选择器。",
              "默认suffix和独立infra面试版保持原状；没有在本批答案上修改参数。", "",
              f"执行package SHA：`{info['source']['sha256']}`。协议SHA：`{info['protocol_sha256']}`。", "",
              "```bash", "python -m scripts.write_transport_diagnosis_report", "```", ""]
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/ot-diagnosis")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/OT_R_DIAGNOSIS_RESULTS.md")
    args = parser.parse_args()
    rebuild(args.source, args.output)


if __name__ == "__main__":
    main()
