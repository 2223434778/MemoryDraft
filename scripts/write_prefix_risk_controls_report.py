"""Audit the explicitly training-project mechanism coverage supplement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memorydraft.evaluation.prefix_risk_continuation import protected
from memorydraft.evaluation.prefix_risk_probe import METHODS, audit, audit_decisions
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.evaluation.splice_probe import input_digest
from scripts.run_prefix_risk_controls import inputs, primary_hashes
from scripts.write_benefit_report import equivalent
from scripts.write_prefix_risk_report import check_provenance
from scripts.write_splice_report import audit_calls, preserve_diagnostic

ROOT = Path(__file__).resolve().parents[1]


def build(source):
    tasks, events, calibration = inputs(ROOT)
    manifest = json.loads((source / "source-manifest.json").read_text())
    info = json.loads((source / "provenance.json").read_text())
    checked = {**info, "frozen_sha256": {"docs/PREFIX_RISK_CONTROLS_PROTOCOL.md": info["protocol_sha256"],
                "data/research/prefix-risk/calibration.json": info["calibration_sha256"]}}
    check_provenance(checked, manifest, protected_hashes=protected(ROOT), runner="scripts/run_prefix_risk_controls.py")
    actual_primary = primary_hashes(ROOT)
    if (info["status"] != "complete" or info["mechanism_rows"] != 10 or info["timing_rows"] != 20
            or not info["primary_unchanged"] or any(actual_primary.get(n) != h for n, h in info["primary_sha256"].items())
            or info["inputs_sha256"] != input_digest(tasks, events)):
        raise ValueError("Changed known-control inputs, coverage or primary evidence")
    primary_info = json.loads((ROOT / "results/prefix-risk/continuation/mechanism/provenance.json").read_text())
    if info["source"] != primary_info["source"]:
        raise ValueError("Coverage controls changed the measured runtime package")
    stages, summaries = {}, {}
    for stage, repeats in (("mechanism", 1), ("timing", 2)):
        rows = read_rows(source / stage / "rows.jsonl")
        summary = audit(rows, tasks, repeats, instrumented=stage == "mechanism")
        if not equivalent(summary, json.loads((source / stage / "summary.json").read_text())):
            raise ValueError("Changed known-control summary")
        stages[stage], summaries[stage] = rows, summary
    mechanisms, timed = stages["mechanism"], stages["timing"]
    audit_calls([{**r, "method": "Splice-" + r["method"]} if r["method"] in METHODS[3:] else r for r in mechanisms], tasks)
    audit_decisions(mechanisms, tasks, calibration)
    expected = {(r["request_id"], r["method"]): r["counters"] for r in mechanisms}
    if any(v != r["counters"][k] for r in timed for k, v in expected[r["request_id"], r["method"]].items() if k != "alignment_seconds"):
        raise ValueError("Control timing differs from mechanism")
    requests, changes = {}, []
    for task in tasks:
        requests[task.request.id] = {m: dict(seconds=sum(r["latency_seconds"] for r in timed if r["request_id"] == task.request.id and r["method"] == m),
                rounds=expected[task.request.id, m]["rounds"], switches=expected[task.request.id, m].get("risk_switches", 0)) for m in METHODS}
    for r in mechanisms:
        for d in r["decisions"]:
            if d["baseline"] != d["selected"]:
                a, b = (d["candidates"][d[k]] for k in ("baseline", "selected"))
                changes.append(dict(request_id=r["request_id"], method=r["method"], generated=d["generated"],
                                    old_boundary=a["prefix_length"], new_boundary=b["prefix_length"],
                                    old_accepted=a["accepted"], new_accepted=b["accepted"]))
    same_paths = all(
        next(r for r in mechanisms if r["request_id"] == t.request.id and r["method"] == "Risk-Splice")["calls"]
        == next(r for r in mechanisms if r["request_id"] == t.request.id and r["method"] == "Risk-NoConfidence")["calls"]
        for t in tasks)
    diagnostic = dict(mechanism=summaries["mechanism"], timing=summaries["timing"], requests=requests, changes=changes)
    if (source / "diagnostic.json").exists():
        saved = json.loads((source / "diagnostic.json").read_text())
        if not equivalent(diagnostic, saved):
            raise ValueError("Changed published control diagnostic")
        diagnostic, requests, changes = saved, saved["requests"], saved["changes"]
    lines = ["# 风险拼接：两个已有训练项目的机制覆盖", "",
             "这两个案例为已知训练项目，特意选择以覆盖风险边界分支。答案/候选早已观察，不能作为独立泛化证据，不并入12请求续测主结果；没有调整模型、阈值、长度或候选池。",
             "", "10/10机制、20/20计时严格匹配各自旧AR token/prompt/快照；20/20计数路径匹配机制。全部在线成本计入，未重新构造AR。", "",
             "| 请求 | 方法 | 验证轮 | 两次总秒 | 改选 |", "|---|---|---:|---:|---:|"]
    for q, values in requests.items():
        for m in METHODS:
            v = values[m]
            lines.append(f"| {q} | {m} | {v['rounds']} | {v['seconds']:.6f} | {v['switches']} |")
    lines += ["", "| 实际改选 | 位置 | 原边界→新边界 | 原候选→新候选同状态实际接受 |", "|---|---:|---|---|"]
    for d in changes:
        lines.append(f"| {d['request_id']}/{d['method']} | {d['generated']} | {d['old_boundary']}→{d['new_boundary']} | {d['old_accepted']}→{d['new_accepted']} |")
    for q, values in requests.items():
        risk = values["Risk-Splice"]["seconds"]
        lines += ["", f"{q}：Risk相对Splice-Edit延迟变化{risk/values['Splice-Edit']['seconds']-1:+.2%}，相对Guided-Edit {risk/values['Guided-Edit']['seconds']-1:+.2%}。"]
    lines += ["", f"两控制中有/无置信度的实际逐轮候选提议完全相同：{same_paths}。payment的局部改善不需要概率/熵信息也能由该消融取得，尚无置信度的独立增量证据。Risk仍未超过该例的Guided-Edit，不能把修复内部拼接退化说成超过所有合理基线。"]
    lines += ["", "解释边界：同状态接受增加不等于整条路径收益，必须结合表中总轮与全成本时延；无置信度消融若取得相同行为，也不能把收益独归于概率/熵信息。两个训练案例只回答机制是否能工作，不能说明新项目可泛化。", "",
              f"冻结执行`{manifest['commit']}`，package SHA `{info['source']['sha256']}`；原结果及12请求主证据的已记录文件SHA未变。", "",
              "```bash", "python -m scripts.write_prefix_risk_controls_report", "```", ""]
    return "\n".join(lines), diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/prefix-risk/continuation/controls")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/PREFIX_RISK_CONTROLS_RESULTS.md")
    args = parser.parse_args()
    report, diagnostic = build(args.source)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    if not (args.source / "diagnostic.json").exists():
        preserve_diagnostic(args.source / "diagnostic.json", diagnostic)


if __name__ == "__main__":
    main()
