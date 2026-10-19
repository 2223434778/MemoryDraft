"""Audit and rebuild the bounded splice development report without loading models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.evaluation.splice_probe import METHODS, audit, input_digest, inputs, mechanism_gate, prefix_acceptance, protected_hashes
from scripts.diagnose_transport_online import write


ROOT = Path(__file__).resolve().parents[1]


def audit_calls(rows, tasks):
    references = {t.request.id: t.reference["token_ids"] for t in tasks}
    for row in rows:
        rounds = {r["generated_before"]: r for r in row["rounds"]}
        c = row["counters"]
        if len(rounds) != c["rounds"] or sum(r["draft_forward"] for r in rounds.values()) != c.get("draft_forward_rounds", c["draft_rounds"]):
            raise ValueError("Changed decode-round accounting")
        prefix, historical, extension, internal, proposals = 0, 0, 0, 0, 0
        for call in row["calls"]:
            remaining = references[row["request_id"]][call["generated"]:]
            p = call["proposal"]
            if call["draft_oracle_accepted"] != prefix_acceptance(call["draft_tokens"], remaining):
                raise ValueError("Changed post-hoc Draft acceptance")
            if not p:
                if call["accepted"] is not None or call["oracle_accepted"] is not None:
                    raise ValueError("Acceptance without a proposal")
                continue
            r = rounds[call["generated"]]
            if (call["oracle_accepted"] != prefix_acceptance(p["tokens"], remaining)
                    or call["accepted"] != r["accepted"] or r["route"] != "R"
                    or len(p["tokens"]) != r["proposed"]):
                raise ValueError("Changed actual proposal or post-hoc acceptance")
            if row["method"].startswith("Splice-"):
                b, a = p["draft_prefix"], call["accepted"]
                if (not 4 <= b <= len(call["draft_tokens"]) or p["tokens"][:b] != call["draft_tokens"][:b]
                        or p["original_draft"] != len(call["draft_tokens"])):
                    raise ValueError("Changed retained Draft prefix")
                prefix += min(a, b)
                historical += max(0, a - b)
                extension += max(0, a - p["original_draft"])
                internal += int(b > 4)
                proposals += 1
        if row["method"].startswith("Splice-") and (prefix, historical, extension, internal, proposals) != tuple(c[k] for k in
                ("splice_prefix_accepted_tokens", "splice_history_accepted_tokens", "splice_extension_accepted_tokens",
                 "splice_internal_proposals", "splice_proposals")):
            raise ValueError("Changed historical vs Draft attribution")


def load_stage(source, stage, tasks, events, manifest):
    directory = source / stage
    info = json.loads((directory / "provenance.json").read_text())
    instrumented = stage == "mechanism"
    expected_files = {name.removeprefix("memorydraft/"): sha for name, sha in manifest["files"].items()
                      if name.startswith("memorydraft/") and name.endswith((".py", ".json", ".txt"))}
    if (info["status"] != "complete" or info["stage"] != stage or info["instrumented"] != instrumented
            or info["inputs_sha256"] != input_digest(tasks, events)
            or info["original_sha256"] != protected_hashes(ROOT, include_splice=info.get("cooldown_check", False))
            or not info["original_unchanged"] or info["methods"] != list(METHODS)
            or info["runner_sha256"] != manifest["files"]["scripts/run_splice_probe.py"]
            or info["protocol_sha256"] != manifest["files"]["docs/OT_SPLICE_COOLDOWN_PROTOCOL.md" if info.get("cooldown_check", False) else "docs/OT_SPLICE_PROTOCOL.md"]
            or info["source"]["files"] != expected_files
            or info["source"]["sha256"] != hashlib.sha256(json.dumps(expected_files, sort_keys=True).encode()).hexdigest()):
        raise ValueError("Changed splice execution provenance")
    rows = read_rows(directory / "rows.jsonl")
    summary = audit(rows, tasks, 1 if instrumented else 2, instrumented=instrumented)
    if summary != json.loads((directory / "summary.json").read_text()):
        raise ValueError("Changed splice summary")
    if len(rows) != info["rows"]:
        raise ValueError("Wrong splice observation count")
    if instrumented and info["mechanism_gate"] != mechanism_gate(rows):
        raise ValueError("Changed mechanism gate")
    if instrumented:
        audit_calls(rows, tasks)
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
        lines += ["", "| 项目 | " + " | ".join(METHODS) + " |", "|---|" + "---:|" * len(METHODS)]
        for project in sorted({t.project_id for t in tasks}):
            times = {m: sum(r["latency_seconds"] for r in timing_rows if r["project_id"] == project and r["method"] == m) for m in METHODS}
            lines.append("| " + project + " | " + " | ".join(f"{times[m]:.6f}" for m in METHODS) + " |")
        keyed = {(r["request_id"], r["method"]): r for r in rows}
        keys = ("rounds", "draft_rounds", "reuse_rounds", "reuse_accepted_tokens", "draft_forward_rounds",
                "alignment_calls", "alignment_candidates", "alignment_proposals", "alignment_accepted_tokens",
                "splice_internal_proposals", "splice_history_accepted_tokens", "splice_extension_accepted_tokens")
        same = all(all(r["counters"].get(k, 0) == keyed[r["request_id"], r["method"]]["counters"].get(k, 0)
                       for k in keys) for r in timing_rows)
        diagnostic["timing_counter_paths_match_mechanism"] = same
        lines += ["", "未插桩重复与机制回放的轮数/接受量/候选及真实Draft前向计数：" + ("192/192相同。" if same else "存在差异，需结合逐行记录分析。")]
    pairs = 0
    for task in tasks:
        proposals = []
        for method in ("Splice-Edit", "Splice-OT"):
            proposals.append([c["proposal"]["tokens"] if c["proposal"] else None for c in
                              next(r for r in rows if r["request_id"] == task.request.id and r["method"] == method)["calls"]])
        pairs += int(proposals[0] == proposals[1])
    diagnostic["edit_ot_equal_proposal_sequences"] = pairs
    lines += ["", f"Splice-Edit与Splice-OT的完整引导调用候选token序列在{pairs}/16请求上相同；这描述实际路径，不等价于共同状态下的全候选排序能力。",
              "", "解释与最小修复：", "",
              "- inventory-query两请求在位置8/10将同状态原Draft接受13/15提升到32，各少1轮；第二个请求的接续来自当前prompt，不能归为历史独特价值。三种Splice均成功，尚无OT独有收益。",
              "- payment-runbook:1在位置11保留11个Draft token但仅接受3；shipment-contract:1在位置14保留9或15个Draft token但仅接受7。两个位置原Draft也分别接受3/7，未尝试到历史尾部。现有R冷却仍被触发，跳过下一轮位置15/22可接受32token的原Guided机会，各多1轮。这是候选来源反馈策略缺陷，严格输出仍正确。",
              "- TailOT在旧confirm-schema-0:0选到不同候选，当前位置接受28而非32，但后续路径总轮11→10；不能将单次接受最大化直接等同于全请求最优。普通OT/Edit总轮相同；TailOT整体仍慢于原Guided-Edit。",
              "- 保存本轮负结果；最小修复只让保留Draft前缀内拒绝不触发历史检索冷却，然后对上述2负例、2 SQL正例及1旧JSON正例作事后开发复查，不扩大新场景。",
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
