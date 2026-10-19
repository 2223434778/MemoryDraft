"""Rebuild the frozen prefix-risk experiment, including its possible stop gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from memorydraft.evaluation.prefix_risk_probe import METHODS, audit, audit_decisions, inputs, protected, timing_gate
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.evaluation.transport_validation import attach_references, contract_check
from scripts.write_benefit_report import equivalent
from scripts.write_splice_report import audit_calls, preserve_diagnostic

ROOT = Path(__file__).resolve().parents[1]


def check_provenance(info, manifest):
    files = {n.removeprefix("memorydraft/"): h for n, h in manifest["files"].items()
             if n.startswith("memorydraft/") and n.endswith((".py", ".json", ".txt"))}
    old = protected(ROOT)
    if set(old) != set(info["original_sha256"]):
        raise ValueError("Changed protected evidence coverage")
    for name, digest in info["original_sha256"].items():
        if digest != manifest["files"][name]:
            raise ValueError("Execution changed a frozen evidence blob")
        if old[name] != digest:
            # These two early audit JSONs predate evidence -text attributes.
            # Windows checkout uses CRLF; deployment exports canonical Git LF.
            if (name not in {"results/research-audit/model_preflight.json", "results/research-audit/pilot_summary.json"}
                    or hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != digest):
                raise ValueError("Changed protected evidence bytes")
    if (not info["original_unchanged"]
            or info["source"]["files"] != files
            or info["source"]["sha256"] != hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
            or info["runner_sha256"] != manifest["files"]["scripts/run_prefix_risk_probe.py"]
            or any(h != manifest["files"][name] or h != file_hash(ROOT / name)
                   for name, h in info["frozen_sha256"].items())):
        raise ValueError("Changed frozen prefix-risk execution provenance")


def comparisons(rows, mechanisms):
    groups = {}
    for group in ("all", "edited-continuation", "control"):
        selected = rows if group == "all" else [r for r in rows if r["group"] == group]
        groups[group] = {m: sum(r["latency_seconds"] for r in selected if r["method"] == m) for m in METHODS}
    primary = [r for r in rows if r["group"] == "edited-continuation"]
    projects = sorted({r["project_id"] for r in primary})
    table = {p: {m: sum(r["latency_seconds"] for r in primary if r["project_id"] == p and r["method"] == m)
                 for m in METHODS} for p in projects}
    rng = np.random.default_rng(20261019)
    boot = []
    for indices in rng.integers(0, len(projects), size=(5000, len(projects))):
        sample = [projects[i] for i in indices]
        boot.append(sum(table[p]["Risk-Splice"] for p in sample) / sum(table[p]["Splice-Edit"] for p in sample))
    wins = sum(table[p]["Risk-Splice"] < table[p]["Splice-Edit"] for p in projects)
    counters = {m: sum(r["counters"]["rounds"] for r in mechanisms if r["method"] == m) for m in METHODS}
    improved = []
    for q in sorted({r["request_id"] for r in mechanisms}):
        keyed = {r["method"]: r for r in mechanisms if r["request_id"] == q}
        if keyed["Risk-Splice"]["counters"]["rounds"] < keyed["Splice-Edit"]["counters"]["rounds"]:
            improved.append(q)
    p, c = groups["edited-continuation"], groups["control"]
    gate = dict(primary_splice_3_percent=p["Risk-Splice"] <= .97 * p["Splice-Edit"],
                primary_guided_1_percent=p["Risk-Splice"] <= .99 * p["Guided-Edit"],
                projects_3_of_4=wins >= 3,
                fewer_rounds=counters["Risk-Splice"] < counters["Splice-Edit"] and len(improved) >= 2,
                better_than_no_confidence=p["Risk-Splice"] < p["Risk-NoConfidence"],
                control_within_3_percent=c["Risk-Splice"] <= 1.03 * c["Splice-Edit"])
    gate["passed"] = all(gate.values())
    return dict(seconds=groups, projects=table, projects_faster=wins, fewer_round_requests=improved,
                risk_over_splice_project_bootstrap_95=np.quantile(boot, [.025, .975]).tolist(), gate=gate)


def build(source):
    manifest = json.loads((source / "source-manifest.json").read_text())
    tasks, _, calibration = inputs(ROOT)
    directory = source / "mechanism"
    info = json.loads((directory / "provenance.json").read_text())
    check_provenance(info, manifest)
    references = read_rows(directory / "fresh-ar.jsonl")
    tasks = attach_references(tasks, references)
    contracts = all(r["contract"]["artifact_correct"] and r["finish_reason"] == "eos" for r in references)
    diagnostic = dict(status=info["status"], references=len(references), contract_passed=contracts,
                      ar_seconds=sum(r["latency_seconds"] for r in references))
    lines = ["# 前缀风险辅助拼接：独立项目检查", "",
             "2026-10-19。仅用旧56状态/9项目训练前缀生存，固定0.5边界风险规则；新6项目/12请求与训练项目隔离。四项目为短修改后恢复复制，另两个为复制/全文替换控制。全部为作者受控输入，不是生产流量；当前prompt含源文，不主张长期历史不可替代。",
             "", "精确suffix优先。风险策略只在原Edit候选边界不安全、另有安全边界时改选；保持原长度上限32、候选池、预算与严格Target验证，不预测长尾、不求解OT。无置信度消融采用相同训练和规则。",
             "", f"12个新AR参考构造成本{diagnostic['ar_seconds']:.6f}s，单列而非在线计时。", "",
             "| 请求 | 文本合同 | 结束原因 |", "|---|---|---|"]
    for r in references:
        if r["contract"] != contract_check(r["text"], next(t.expected_text for t in tasks if t.request.id == r["request_id"])):
            raise ValueError("Changed Target artifact contract")
        lines.append(f"| {r['request_id']} | {r['contract']['artifact_correct']} | {r['finish_reason']} |")
    if not contracts:
        if info["status"] != "stopped_target_contract" or info["rows"] != 0 or (source / "timing").exists():
            raise ValueError("Experiment continued after the frozen Target contract gate")
        lines += ["", "Target未通过预先固定的完整输出合同，按协议停止；未运行60次机制或120次计时。这不能判断风险策略有无性能优势。本轮不调整提示、不挑成功项目继续，保留实现和失败参考。",
                  "", "失败项vector-ingest:0要求只将batch_size从160改为224并返回完整文件，Target只输出`batch_size = 224`，遗漏完整文件的其余配置内容。这是实质性完整性失败；其EOS结束，非输出预算耗尽。其余11项合同通过（忽略末尾LF数量），但不能事后仅选它们作为本轮完整验证。",
                  "", "固定Target与严格接受规则下，加速结果必须与AR相同，候选风险策略无法纠正这次遗漏。应保留原实验的停止结论；未来若重新验证，需要事先另立公共输出合同和参考基线，不能将本轮改提示后重新运行叫独立确认。"]
    else:
        if info["status"] != "complete" or info["rows"] != 60:
            raise ValueError("Incomplete strict mechanism experiment")
        rows = read_rows(directory / "rows.jsonl")
        mechanism = audit(rows, tasks, 1, instrumented=True)
        audit_calls([{**r, "method": "Splice-" + r["method"]} if r["method"] in METHODS[3:] else r for r in rows], tasks)
        audit_decisions(rows, tasks, calibration)
        if not equivalent(mechanism, json.loads((directory / "summary.json").read_text())) or info["timing_gate"] != timing_gate(rows):
            raise ValueError("Changed mechanism summary/gate")
        diagnostic.update(mechanism=mechanism, timing_gate=timing_gate(rows), switches=[])
        lines += ["", "60/60机制生成逐token、Target prompt、合法快照严格匹配新AR。下面计数来自插桩机制，时间不作性能结论。", "",
                  "| 方法 | 总验证轮 | 实际Draft前向 | 历史尾部接受 | 风险评估 | 改选 |", "|---|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            c = mechanism["all"][method]["counters"]
            forwards = sum(r["counters"].get("draft_forward_rounds", r["counters"]["draft_rounds"]) for r in rows if r["method"] == method)
            lines.append(f"| {method} | {c['rounds']} | {forwards} | {c['splice_history_accepted_tokens']} | {c['risk_evaluations']} | {c['risk_switches']} |")
        lines += ["", "| 请求/位置/组 | 原边界→新边界 | 边界存活概率 | 原候选→新候选实际接受 |", "|---|---|---|---|"]
        for row in rows:
            for d in row["decisions"]:
                if d["baseline"] == d["selected"]:
                    continue
                a, b = (d["candidates"][d[k]] for k in ("baseline", "selected"))
                value = dict(request_id=row["request_id"], method=row["method"], generated=d["generated"],
                             old_boundary=a["prefix_length"], new_boundary=b["prefix_length"],
                             old_survival=d["survival"][a["prefix_length"] - 1], new_survival=d["survival"][b["prefix_length"] - 1],
                             old_accepted=a["accepted"], new_accepted=b["accepted"])
                diagnostic["switches"].append(value)
                lines.append(f"| {row['request_id']}/{d['generated']}/{row['method']} | {a['prefix_length']}→{b['prefix_length']} | {value['old_survival']:.3f}→{value['new_survival']:.3f} | {a['accepted']}→{b['accepted']} |")
        if not timing_gate(rows)["passed"]:
            if (source / "timing").exists():
                raise ValueError("Timing ran despite a no-switch stop gate")
            lines += ["", "有置信度策略没有改选候选，按冻结门槛停止计时。该场景没有体现风险辅助决策的增量；不把相同路径算作性能成功，也不据此证明所有场景无效。不增加新输入或调整阈值继续本轮。"]
        else:
            timed_info = json.loads((source / "timing/provenance.json").read_text())
            check_provenance(timed_info, manifest)
            timed = read_rows(source / "timing/rows.jsonl")
            summary = audit(timed, tasks, 2)
            if timed_info["status"] != "complete" or timed_info["rows"] != 120 or timed_info["paired_paths"] != 120:
                raise ValueError("Incomplete uninstrumented timing experiment")
            expected = {(r["request_id"], r["method"]): r["counters"] for r in rows}
            for r in timed:
                if any(v != r["counters"][k] for k, v in expected[r["request_id"], r["method"]].items() if k != "alignment_seconds"):
                    raise ValueError("Timing changed a checked path")
            if not equivalent(summary, json.loads((source / "timing/summary.json").read_text())):
                raise ValueError("Changed paired timing summary")
            diagnostic.update(timing=summary, comparisons=comparisons(timed, rows))
            value = diagnostic["comparisons"]
            lines += ["", "120/120未插桩交错计时严格匹配AR，120/120计数路径与机制一致。请求时间包含置信度/熵、CPU/GPU同步、检索/策略、索引、全部前向、KV及解码；加载/预热/新AR单列。", "",
                      "| 方法 | 主场景秒 | 控制秒 | 全部秒 |", "|---|---:|---:|---:|"]
            for m in METHODS:
                lines.append(f"| {m} | {value['seconds']['edited-continuation'][m]:.6f} | {value['seconds']['control'][m]:.6f} | {value['seconds']['all'][m]:.6f} |")
            lines += ["", "| 主项目 | " + " | ".join(METHODS) + " |", "|---|" + "---:|" * len(METHODS)]
            for p, table in value["projects"].items():
                lines.append("| " + p + " | " + " | ".join(f"{table[m]:.6f}" for m in METHODS) + " |")
            p = value["seconds"]["edited-continuation"]
            interval = value["risk_over_splice_project_bootstrap_95"]
            lines += ["", f"主场景Risk相对Splice-Edit延迟变化{p['Risk-Splice']/p['Splice-Edit']-1:+.2%}，相对Guided-Edit {p['Risk-Splice']/p['Guided-Edit']-1:+.2%}，相对Suffix {p['Risk-Splice']/p['Suffix-R']-1:+.2%}；负值为更快。",
                      f"4个主项目中{value['projects_faster']}个比Splice-Edit快，Risk/Splice项目bootstrap 95%区间[{interval[0]:.4f}, {interval[1]:.4f}]。只描述四项目小样本，不称广泛显著性。",
                      "", "冻结的继续投入门槛：", "", "```json", json.dumps(value["gate"], indent=2), "```", "",
                      "门槛未达则结束本轮，保留局部正负结果；不修改本轮阈值/特征后重复确认。即使局部连续接受增加，也不能自动推导全请求轮数减少，后续suffix机会可能改变。"]
    lines += ["", f"执行提交`{manifest['commit']}`；package SHA `{info['source']['sha256']}`。原始旧证据递归SHA未变，独立infra版本和默认suffix保留。", "",
              "交付时的真实tiny Target组合测试另发现均匀分布FP32熵可略超1，已增加[0,1]数值裁剪与回归；这是风险特征的边界修复，未改Target验证。本次冻结GPU检查停在AR阶段，未执行新风险策略的机制/性能路径；不能将组合单测当作其GPU性能验证。", "",
              "来源核验使用冻结Git blob。仅两份早期research-audit JSON在本地Windows检出为CRLF、服务器Git原始导出为LF，报告只读兼容此已知差异，不修改原文件；其余证据字节SHA精确检查。", "",
              "```bash", "python -m scripts.write_prefix_risk_report", "```", ""]
    return "\n".join(lines), diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/prefix-risk")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/PREFIX_RISK_RESULTS.md")
    args = parser.parse_args()
    report, diagnostic = build(args.source)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    existing = args.source / "diagnostic.json"
    if existing.exists() and not equivalent(diagnostic, json.loads(existing.read_text())):
        raise ValueError("Changed published prefix-risk diagnostics")
    if not existing.exists():
        preserve_diagnostic(existing, diagnostic)


if __name__ == "__main__":
    main()
