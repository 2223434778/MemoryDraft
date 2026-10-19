"""Rebuild the five-request post-hoc cooldown repair check separately from v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.evaluation.splice_probe import COOLDOWN_IDS, METHODS, inputs
from scripts.diagnose_transport_online import write
from scripts.write_splice_report import ROOT, load_stage


def build(source):
    tasks, events = inputs(ROOT)
    tasks = [t for t in tasks if t.request.id in COOLDOWN_IDS]
    events = [e for e in events if e.user_id in {t.request.user_id for t in tasks}]
    manifest = json.loads((source / "execution-source.json").read_text())
    info, rows, summary = load_stage(source, "mechanism", tasks, events, manifest)
    timing_info, timing_rows, timing = load_stage(source, "timing", tasks, events, manifest)
    if not info["cooldown_check"] or not timing_info["cooldown_check"] or info["source"] != timing_info["source"]:
        raise ValueError("Wrong repair scope or changed execution package")
    prior = {(r["request_id"], r["method"]): r for r in read_rows(ROOT / "results/ot-splice/mechanism/rows.jsonl")}
    keyed = {(r["request_id"], r["method"]): r for r in rows}
    restored = {method: all(any(c["generated"] == position and c["accepted"] == 32 for c in
                                keyed[request, method]["calls"]) for request, position in
                            (("payment-runbook:1", 15), ("shipment-contract:1", 22))) for method in METHODS[3:]}
    unchanged_positives = all(keyed[t.request.id, m]["counters"]["rounds"] == prior[t.request.id, m]["counters"]["rounds"]
                             for t in tasks if t.request.id not in {"payment-runbook:1", "shipment-contract:1"} for m in METHODS)
    diagnostic = dict(restored_32_token_opportunities=restored, positive_round_counts_unchanged=unchanged_positives,
                      mechanisms=summary, timing=timing)
    lines = ["# 拼接前缀拒绝：冷却修复检查", "", "2026-10-19。五个已经观察过的请求，事后定位与修复验证，不是新holdout。首轮负结果完整保留于OT_SPLICE_RESULTS.md。", "",
             "最小修复：尚在保留Draft前缀内拒绝时不触发历史检索冷却；首个历史token或之后拒绝仍冷却。Target验证、候选池/阈值及其他方法规则不变。", "",
             "30/30机制回放、60/60未插桩交错计时与原AR逐token/prompt/snapshot严格一致；两阶段package/runner相同，全部旧证据包括首轮拼接SHA未变。", "",
             "| 请求 | 方法 | 首轮总轮 | 修复后总轮 |", "|---|---|---:|---:|"]
    for task in tasks:
        for method in METHODS[3:]:
            lines.append(f"| {task.request.id} | {method} | {prior[task.request.id, method]['counters']['rounds']} | {keyed[task.request.id, method]['counters']['rounds']} |")
    lines += ["", "两个退化请求下一轮32token机会恢复：" + "；".join(f"{m}={v}" for m, v in restored.items()) + "。",
              f"三个正例各方法总轮保持首轮计数：{unchanged_positives}。", "",
              "以下只比较本批五请求、两重复的配对总时延，不将首轮绝对时间跨运行相减，也不外推修复后的16请求总性能。", "",
              "| 方法 | 秒 | 相对同批Guided-Edit延迟 |", "|---|---:|---:|"]
    for method in METHODS:
        seconds = timing["all"][method]["seconds"]
        lines.append(f"| {method} | {seconds:.6f} | {(seconds / timing['all']['Guided-Edit']['seconds'] - 1) * 100:+.2f}% |")
    lines += ["", "| 请求 | " + " | ".join(METHODS) + " |", "|---|" + "---:|" * len(METHODS)]
    for task in tasks:
        times = {m: sum(r["latency_seconds"] for r in timing_rows if r["request_id"] == task.request.id and r["method"] == m) for m in METHODS}
        lines.append("| " + task.request.id + " | " + " | ".join(f"{times[m]:.6f}" for m in METHODS) + " |")
    count_keys = ("rounds", "draft_rounds", "reuse_rounds", "reuse_accepted_tokens", "draft_forward_rounds",
                  "alignment_calls", "alignment_candidates", "alignment_proposals", "alignment_accepted_tokens")
    same = all(all(r["counters"].get(k, 0) == keyed[r["request_id"], r["method"]]["counters"].get(k, 0)
                   for k in count_keys) for r in timing_rows)
    diagnostic["timing_counter_paths_match_mechanism"] = same
    lines += ["", f"60个未插桩观察与机制回放计数一致：{same}。", "",
              "解释：恢复被错误冷却跳过的检索机会是反馈策略修复的收益。SQL桥接收益仍需与同池Splice-Edit比较；这五个请求由已知故障/正例选取，不能证明OT的独立泛化优势。默认suffix与独立infra面试版保留。", "",
              f"执行提交：`{manifest['commit']}`；package SHA：`{info['source']['sha256']}`。", "",
              "```bash", "python -m scripts.write_splice_cooldown_report", "```", ""]
    return "\n".join(lines), diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/ot-splice-cooldown")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/OT_SPLICE_COOLDOWN_RESULTS.md")
    args = parser.parse_args()
    report, diagnostic = build(args.source)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    write(args.source / "diagnostic.json", diagnostic)


if __name__ == "__main__":
    main()
