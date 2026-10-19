"""Audit all continuation pairs while retaining failed Target task contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memorydraft.evaluation.prefix_risk_continuation import protected, reference_source
from memorydraft.evaluation.prefix_risk_probe import METHODS, audit, audit_decisions, inputs, timing_gate
from memorydraft.evaluation.runtime_migration import read_rows
from scripts.write_benefit_report import equivalent
from scripts.write_prefix_risk_report import check_provenance, comparisons
from scripts.write_splice_report import audit_calls, preserve_diagnostic

ROOT = Path(__file__).resolve().parents[1]


def build(source):
    manifest = json.loads((source / "source-manifest.json").read_text())
    tasks, _, calibration = inputs(ROOT)
    tasks, metadata, _ = reference_source(ROOT, tasks)
    references = read_rows(source / "mechanism/fresh-ar.jsonl")
    if references != [t.reference for t in tasks]:
        raise ValueError("Continuation dropped or changed an original AR reference")
    infos, stages, summaries = {}, {}, {}
    for stage, count, repeats in (("mechanism", 60, 1), ("timing", 120, 2)):
        directory = source / stage
        info = json.loads((directory / "provenance.json").read_text())
        check_provenance(info, manifest, protected_hashes=protected(ROOT))
        if (info["status"] != "complete" or info["stage"] != stage or info["rows"] != count
                or not info["continued_evaluation"]):
            raise ValueError("Incomplete all-request continuation")
        rows = read_rows(directory / "rows.jsonl")
        summary = audit(rows, tasks, repeats, instrumented=stage == "mechanism")
        if not equivalent(summary, json.loads((directory / "summary.json").read_text())):
            raise ValueError("Changed continuation summary")
        keyed = {t.request.id: t for t in tasks}
        if any(r["contract"] != keyed[r["request_id"]].reference["contract"] for r in rows):
            raise ValueError("Task quality differs from the unchanged Target reference")
        infos[stage], stages[stage], summaries[stage] = info, rows, summary
    if not equivalent(infos["mechanism"]["reference_source"], metadata):
        raise ValueError("Changed original reference provenance")
    expected_gate = {**timing_gate(stages["mechanism"]), "passed": True,
                     "policy": "Measure all completed strict pairs, including unchanged paths"}
    if infos["mechanism"]["timing_gate"] != expected_gate or infos["timing"]["paired_paths"] != 120:
        raise ValueError("Changed continuation timing policy")
    mechanisms, timed = stages["mechanism"], stages["timing"]
    audit_calls([{**r, "method": "Splice-" + r["method"]} if r["method"] in METHODS[3:] else r for r in mechanisms], tasks)
    audit_decisions(mechanisms, tasks, calibration)
    expected = {(r["request_id"], r["method"]): r["counters"] for r in mechanisms}
    for r in timed:
        if any(v != r["counters"][k] for k, v in expected[r["request_id"], r["method"]].items() if k != "alignment_seconds"):
            raise ValueError("Timing changed a strictly checked mechanism path")
    comparison = comparisons(timed, mechanisms)
    quality = {name: {m: sum(r["latency_seconds"] for r in timed if r["method"] == m
                            and r["contract"]["artifact_correct"] == correct) for m in METHODS}
               for name, correct in (("contract_correct", True), ("contract_incorrect", False))}
    decisions = []
    for r in mechanisms:
        for d in r["decisions"]:
            if d["baseline"] != d["selected"]:
                a, b = (d["candidates"][d[k]] for k in ("baseline", "selected"))
                decisions.append(dict(request_id=r["request_id"], method=r["method"], generated=d["generated"],
                    old_boundary=a["prefix_length"], new_boundary=b["prefix_length"],
                    old_survival=d["survival"][a["prefix_length"]-1], new_survival=d["survival"][b["prefix_length"]-1],
                    old_accepted=a["accepted"], new_accepted=b["accepted"]))
    diagnostic = dict(reference_source=metadata, mechanism=summaries["mechanism"], timing=summaries["timing"],
                      comparisons=comparison, quality_seconds=quality, switches=decisions)
    existing = source / "diagnostic.json"
    if existing.exists():
        saved = json.loads(existing.read_text())
        if not equivalent(saved, diagnostic):
            raise ValueError("Changed published continuation diagnostics")
        diagnostic = saved
        comparison, quality, decisions = (diagnostic[k] for k in ("comparisons", "quality_seconds", "switches"))
    lines = ["# 前缀风险拼接续测结果", "",
             "用户要求放宽原评测停止条件后，完整测试同一批已观察项目。风险模型、0.5阈值、候选池/预算、32上限及Target输入均未调整。原停止现场和原报告保留。",
             "", "复用原12个AR参考：11个完整编辑合同正确，vector-ingest:0仍遗漏正文。全部请求纳入主结果，不挑成功例；任务质量单列。项目与旧训练集隔离，但答案已观察，本轮不是新holdout或生产负载。",
             "", "60/60机制、120/120未插桩交错计时逐token、prompt、合法快照严格匹配AR；120/120计时计数路径与机制一致。Target接受条件未放宽，原错误正文也被严格复现。",
             "", f"原AR构造成本{metadata['historical_ar_seconds']:.6f}s为历史成本，本轮没有再生成AR。在线时间包含索引、检索/策略、置信度与熵、CPU/GPU同步、全部前向/KV和解码；加载/预热单列。", "",
             "| 方法 | 全部12请求秒 | 主场景8请求秒 | 控制4请求秒 | 合同正确11请求秒 | 错误1请求秒 |", "|---|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        s = comparison["seconds"]
        lines.append(f"| {m} | {s['all'][m]:.6f} | {s['edited-continuation'][m]:.6f} | {s['control'][m]:.6f} | {quality['contract_correct'][m]:.6f} | {quality['contract_incorrect'][m]:.6f} |")
    lines += ["", "表中为两次交错计时总和，不是单请求均值；错误任务的短答案计时不代表可用编辑收益。", "",
              "| 方法 | 机制总轮 | 实际Draft前向 | 历史尾部接受 | 风险评估 | 改选 |", "|---|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        c = summaries["mechanism"]["all"][m]["counters"]
        forwards = sum(r["counters"].get("draft_forward_rounds", r["counters"]["draft_rounds"]) for r in mechanisms if r["method"] == m)
        lines.append(f"| {m} | {c['rounds']} | {forwards} | {c['splice_history_accepted_tokens']} | {c['risk_evaluations']} | {c['risk_switches']} |")
    lines += ["", "| 实际改选：请求/位置/组 | 边界 | 存活概率 | 同状态实际接受 |", "|---|---|---|---|"]
    for d in decisions:
        lines.append(f"| {d['request_id']}/{d['generated']}/{d['method']} | {d['old_boundary']}→{d['new_boundary']} | {d['old_survival']:.3f}→{d['new_survival']:.3f} | {d['old_accepted']}→{d['new_accepted']} |")
    lines += ["", "同状态候选接受差异是事后诊断，不等于全请求净增token或净少轮数。", "",
              "| 请求 | " + " | ".join(METHODS) + " |", "|---|" + "---:|" * len(METHODS)]
    for task in tasks:
        lines.append("| " + task.request.id + " | " + " | ".join(
            f"{sum(r['latency_seconds'] for r in timed if r['request_id'] == task.request.id and r['method'] == m):.6f}" for m in METHODS) + " |")
    p = comparison["seconds"]["edited-continuation"]
    interval = comparison["risk_over_splice_project_bootstrap_95"]
    lines += ["", f"主场景Risk相对Splice-Edit延迟变化{p['Risk-Splice']/p['Splice-Edit']-1:+.2%}，相对Guided-Edit {p['Risk-Splice']/p['Guided-Edit']-1:+.2%}，相对Suffix {p['Risk-Splice']/p['Suffix-R']-1:+.2%}；负值为更快。",
              f"4主项目中{comparison['projects_faster']}个比Splice-Edit快，Risk/Splice项目bootstrap 95%区间[{interval[0]:.4f}, {interval[1]:.4f}]。四项目、两重复不足以支持广泛显著性或生产结论。",
              "", "原继续投入参考指标仅作描述，没有阻断本轮120次计时：", "", "```json", json.dumps(comparison["gate"], indent=2, sort_keys=True), "```", "",
              "本批风险评估和改选均为0，Splice三组均无内部拼接历史尾部接受。唯一的拼接候选池没有达到准入分数；现有精确suffix/普通Draft已覆盖实际路径，风险模型未被调用。因此约0.01%量级的总时差不能作为风险决策收益，也不能据此判断激活后的风险排序无效。", "",
              "另有两个已知训练项目用于检查风险分支覆盖，单独见[PREFIX_RISK_CONTROLS_RESULTS.md](PREFIX_RISK_CONTROLS_RESULTS.md)。补充结果不合入本表，也不作为泛化证据。", "",
              f"冻结执行`{manifest['commit']}`，package SHA `{infos['mechanism']['source']['sha256']}`。旧结果含首次停止现场在执行前后SHA不变。", "",
              "```bash", "python -m scripts.write_prefix_risk_continuation_report", "```", ""]
    return "\n".join(lines), diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/prefix-risk/continuation")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/PREFIX_RISK_CONTINUATION_RESULTS.md")
    args = parser.parse_args()
    report, diagnostic = build(args.source)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    if not (args.source / "diagnostic.json").exists():
        preserve_diagnostic(args.source / "diagnostic.json", diagnostic)


if __name__ == "__main__":
    main()
