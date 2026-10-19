"""Audit and rebuild the bounded splice development report without loading models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.evaluation.splice_probe import METHODS, audit, input_digest, inputs, mechanism_gate, protected_hashes
from scripts.diagnose_transport_online import write


ROOT = Path(__file__).resolve().parents[1]


def load_stage(source, stage, tasks, events, manifest):
    directory = source / stage
    info = json.loads((directory / "provenance.json").read_text())
    instrumented = stage == "mechanism"
    if (info["status"] != "complete" or info["stage"] != stage or info["instrumented"] != instrumented
            or info["inputs_sha256"] != input_digest(tasks, events) or info["original_sha256"] != protected_hashes(ROOT)
            or not info["original_unchanged"] or info["methods"] != list(METHODS)
            or info["runner_sha256"] != manifest["files"]["scripts/run_splice_probe.py"]
            or info["protocol_sha256"] != manifest["files"]["docs/OT_SPLICE_PROTOCOL.md"]
            or any(manifest["files"].get("memorydraft/" + key) != sha for key, sha in info["source"]["files"].items())):
        raise ValueError("Changed splice execution provenance")
    rows = read_rows(directory / "rows.jsonl")
    summary = audit(rows, tasks, 1 if instrumented else 2, instrumented=instrumented)
    if summary != json.loads((directory / "summary.json").read_text()):
        raise ValueError("Changed splice summary")
    if len(rows) != info["rows"]:
        raise ValueError("Wrong splice observation count")
    if instrumented and info["mechanism_gate"] != mechanism_gate(rows):
        raise ValueError("Changed mechanism gate")
    return info, rows, summary


def build(source):
    tasks, events = inputs(ROOT)
    manifest = json.loads((source / "execution-source.json").read_text())
    info, rows, summary = load_stage(source, "mechanism", tasks, events, manifest)
    diagnostic = dict(mechanism_gate=info["mechanism_gate"], mechanisms=summary, internal_calls=[])
    lines = ["# Draft内部锚点拼接：小成本开发检查", "", "2026-10-19。16个请求的Target答案均已在之前实验中观察，本轮不是独立holdout。", "",
             "精确suffix优先；失败后保留DFlash到内部4-token锚点的前缀，拼接历史原文尾部。三种Splice共用候选池、预算和严格Target验证。",
             "Splice-OT沿用原评分，TailOT只将整体与末8位置贡献各加权一半；没有实施单调OT或语义检索。", "",
             f"机制回放{len(rows)}/{len(rows)}输出token、Target输入和合法快照与既有AR严格一致。插桩延迟单独标记，不用于性能结论。", "",
             "| 方法 | 总轮 | 实际Draft前向 | 对齐接受token | 保留Draft接受 | 历史尾部接受 | 超出原Draft接受 | 内部拼接提议 | 查询秒（插桩） |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        c = summary["all"][method]["counters"]
        forwards = c["draft_forward_rounds"] if method != "Suffix-R" else c["draft_rounds"]
        lines.append(f"| {method} | {c['rounds']} | {forwards} | {c['alignment_accepted_tokens']} | {c['splice_prefix_accepted_tokens']} | {c['splice_history_accepted_tokens']} | {c['splice_extension_accepted_tokens']} | {c['splice_internal_proposals']} | {c['alignment_seconds']:.6f} |")
    lines += ["", "对齐接受量包含保留Draft前缀，不能全部归为历史收益；超出原Draft接受量也不能直接当作相对其他路线的净增量。", "",
              "| 请求 | " + " | ".join(METHODS) + " |", "|---|" + "---:|" * len(METHODS)]
    for task in tasks:
        keyed = {r["method"]: r for r in rows if r["request_id"] == task.request.id}
        lines.append("| " + task.request.id + " | " + " | ".join(str(keyed[m]["counters"]["rounds"]) for m in METHODS) + " |")
        for method in METHODS[3:]:
            for call in keyed[method]["calls"]:
                p = call["proposal"]
                if p and p["draft_prefix"] > 4:
                    diagnostic["internal_calls"].append(dict(request_id=task.request.id, method=method,
                        generated=call["generated"], draft_prefix=p["draft_prefix"], accepted=call["accepted"],
                        oracle_accepted=call["oracle_accepted"], draft_oracle_accepted=call["draft_oracle_accepted"]))
    lines += ["", "机制门槛：同一个Splice方法至少2请求比两个Guided基线都少轮，且内部拼接在同状态比原Draft连续接受更长。",
              "本轮：" + ("通过。" if info["mechanism_gate"]["passed"] else "未通过；停止该机制的GPU扩展，未运行正式计时。"), ""]
    for method, requests in info["mechanism_gate"]["improved_requests"].items():
        lines.append(f"- {method}：{len(requests)}请求；" + ("、".join(requests) if requests else "无") + "。")
    if info["mechanism_gate"]["passed"]:
        timing_info, timing_rows, timing = load_stage(source, "timing", tasks, events, manifest)
        if timing_info["source"] != info["source"]:
            raise ValueError("Timing package differs from mechanism package")
        diagnostic["timing"] = timing
        lines += ["", f"未插桩计时{len(timing_rows)}/{len(timing_rows)}严格一致；每请求两重复，全部在线成本计入。仍为已观察任务开发计时。", "",
                  "| 方法 | 总秒 | 相对Suffix延迟 | 相对Guided-Edit延迟 |", "|---|---:|---:|---:|"]
        for method in METHODS:
            seconds = timing["all"][method]["seconds"]
            lines.append(f"| {method} | {seconds:.6f} | {(seconds / timing['all']['Suffix-R']['seconds'] - 1) * 100:+.2f}% | {(seconds / timing['all']['Guided-Edit']['seconds'] - 1) * 100:+.2f}% |")
    pairs = 0
    for task in tasks:
        proposals = []
        for method in ("Splice-Edit", "Splice-OT"):
            proposals.append([c["proposal"]["tokens"] if c["proposal"] else None for c in
                              next(r for r in rows if r["request_id"] == task.request.id and r["method"] == method)["calls"]])
        pairs += int(proposals[0] == proposals[1])
    diagnostic["edit_ot_equal_proposal_sequences"] = pairs
    lines += ["", f"Splice-Edit与Splice-OT的完整引导调用候选token序列在{pairs}/16请求上相同；这描述实际路径，不等价于共同状态下的全候选排序能力。",
              "", "范围限制：作者受控任务、既有答案、固定8候选cheap预筛、相同数值阈值、短Draft桥接范围。不能据本轮判断自然工作负载泛化或OT最优能力。",
              "Target逐token验证、用户隔离、时间快照、全部在线成本口径不变，默认suffix和独立infra面试版保留。", "",
              f"执行提交：`{manifest['commit']}`。package SHA：`{info['source']['sha256']}`。", "",
              "```bash", "python -m scripts.write_splice_report", "```", ""]
    return "\n".join(lines), diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/ot-splice")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/OT_SPLICE_RESULTS.md")
    args = parser.parse_args()
    report, diagnostic = build(args.source)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    write(args.source / "diagnostic.json", diagnostic)


if __name__ == "__main__":
    main()
