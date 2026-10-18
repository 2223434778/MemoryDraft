"""Rebuild the new-case OT validation report from strict, complete evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from memorydraft.evaluation.runtime_migration import file_hash, input_hashes, read_rows
from memorydraft.evaluation.transport_probe import METHODS
from memorydraft.evaluation.transport_validation import GROUPS, attach_references, audit_rows, comparisons, load_inputs


ROOT = Path(__file__).resolve().parents[1]


def rebuild(source: Path, output: Path, *, cases=None):
    cases = cases or ROOT / "data/research/ot-validation/cases.json"
    tasks, _ = load_inputs(cases)
    info = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    references = read_rows(source / "fresh-ar.jsonl")
    tasks = attach_references(tasks, references)
    rows = read_rows(source / "rows.jsonl")
    summary = audit_rows(rows, tasks)
    comparison = comparisons(summary, tasks)
    if (info["status"] != "complete" or not info["original_unchanged"] or info["rows"] != 192
            or info["fresh_ar_correct"] != 24 or info["cases_sha256"] != file_hash(cases)
            or info["fresh_ar_artifact_correct"] != summary["contracts"]["all"]["correct"]
            or summary != json.loads((source / "summary.json").read_text(encoding="utf-8"))
            or comparison != json.loads((source / "comparisons.json").read_text(encoding="utf-8"))
            or set(info["original_sha256"]) != {"ot-r", "reuse-confirmation", "infra-v0.3", "runtime-migration"}
            or any(input_hashes(ROOT / "results" / name) != hashes for name, hashes in info["original_sha256"].items())):
        raise ValueError("Changed or incomplete validation evidence")
    for task in tasks:
        identity = info["identities"][task.request.id]
        if any(identity[key] != task.reference[key] for key in ("prompt_sha256", "snapshot_id", "prompt_tokens")):
            raise ValueError("AR differs from the input identity frozen before generation")
    failed = source / "failed"
    if failed.exists():
        failure = json.loads((failed / "provenance.json").read_text(encoding="utf-8"))
        original_references = read_rows(failed / "fresh-ar.jsonl")
        previous = {t.request.id: t for t in attach_references(tasks, original_references)}
        if failure["status"] != "failed" or failure["rows"] != 0 or failure["cases_sha256"] != info["cases_sha256"]:
            raise ValueError("Incorrect failed-attempt history")
        for task in tasks:
            if any(task.reference[k] != previous[task.request.id].reference[k]
                   for k in ("token_ids", "text", "prompt_sha256", "snapshot_id", "finish_reason", "contract")):
                raise ValueError("AR changed after the audit-only interface correction")
    passed = [g for g in GROUPS if comparison[g]["engineering_gate"]]
    paired = {(r["request_id"], r["repeat"], r["method"]): r for r in rows}
    count_keys = ("rounds", "draft_rounds", "reuse_rounds", "reuse_accepted_tokens", "alignment_calls",
                  "alignment_candidates", "alignment_proposals", "alignment_accepted_tokens")
    equal_counts = sum(all(r["counters"].get(k, 0) == paired[r["request_id"], r["repeat"], "Edit-R"]["counters"].get(k, 0)
                           for k in count_keys) for r in rows if r["method"] == "OT-R")
    active_projects = sorted({r["project_id"] for r in rows if r["method"] == "OT-R" and r["counters"].get("alignment_calls", 0)})
    lines = ["# OT-R：24个新请求的场景验证", "", "2026-10-18。12个新编写受控项目，每项目两请求；不是自然生产流量。",
             "检索实现、参数、任务及工程门槛在首个Target答案前冻结。评测字段修复与重跑记录见下文。原infra面试版继续独立保存。", "",
             "## 结果", "", f"24个AR输入/快照核对通过，{len(rows)}/192计时输出逐token严格一致。",
             f"任务契约通过{summary['contracts']['all']['correct']}/24，长度截断{summary['contracts']['all']['length_stops']}/24。",
             "严格一致表示保持Target行为；任务契约检查用于判断生成内容是否完成指定修订。", "",
             "| 预定场景 | DFlash秒 | Suffix-R秒 | Edit-R秒 | OT-R秒 | OT相对Suffix延迟 | OT相对Edit延迟 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for group in ("all", *GROUPS):
        times = summary["groups"][group]
        c = comparison[group]
        lines.append(f"| {group} | " + " | ".join(f"{times[m]['seconds']:.6f}" for m in METHODS)
                     + f" | {(c['Suffix-R']['ot_over_baseline']-1)*100:+.2f}% | {(c['Edit-R']['ot_over_baseline']-1)*100:+.2f}% |")
    lines += ["", "对齐查询和每请求初始化已包含在端到端耗时，不另外扣除。加载、预热和AR参考单列。", "",
              "| 场景 | OT优于Suffix项目数 | OT优于Edit项目数 | OT/Suffix项目bootstrap 95% | OT/Edit项目bootstrap 95% | 契约 | 工程门槛 |",
              "|---|---:|---:|---|---|---:|---|"]
    for group in GROUPS:
        c = comparison[group]
        s, e = c["Suffix-R"], c["Edit-R"]
        bounds = lambda x: f"[{x['project_bootstrap_95'][0]:.4f}, {x['project_bootstrap_95'][1]:.4f}]"
        contract = summary["contracts"][group]
        lines.append(f"| {group} | {s['projects_faster']}/4 | {e['projects_faster']}/4 | {bounds(s)} | {bounds(e)} | {contract['correct']}/8 | {'通过' if c['engineering_gate'] else '未通过'} |")
    lines += ["", "继续投入门槛：候选场景降低>=5% Suffix延迟、>=2% Edit延迟，分别至少3/4项目更快，且8/8任务正确/无截断。",
              "四项目成组bootstrap仅作描述；小样本和区间跨1均不支持稳定性宣称。control不是可挑选的正例场景。", "",
              "## 逐项目结果", "", "| 场景/项目 | Suffix秒 | Edit秒 | OT秒 | OT相对Suffix | OT对齐接受token | 契约 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for project, values in summary["projects"].items():
        selected = [t for t in tasks if t.project_id == project]
        s, e, o = (values[m] for m in ("Suffix-R", "Edit-R", "OT-R"))
        contract = sum(t.reference["contract"]["artifact_correct"] for t in selected)
        lines.append(f"| {selected[0].group}/{project} | {s['seconds']:.6f} | {e['seconds']:.6f} | {o['seconds']:.6f} | {(o['seconds']/s['seconds']-1)*100:+.2f}% | {o['counters']['alignment_accepted_tokens']} | {contract}/2 |")
    lines += ["", "## 在线计算与对齐成本", "", "| 场景/方法 | D轮 | R轮 | R接受token | 对齐调用 | 提议 | 对齐接受token | 对齐查询秒 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        for method in METHODS:
            c = summary["groups"][group][method]["counters"]
            lines.append(f"| {group}/{method} | {c['draft_rounds']} | {c['reuse_rounds']} | {c['reuse_accepted_tokens']} | {c['alignment_calls']} | {c['alignment_proposals']} | {c['alignment_accepted_tokens']} | {c['alignment_seconds']:.6f} |")
    totals = summary["groups"]["all"]
    suffix, edit, aligned = (totals[m]["counters"] for m in ("Suffix-R", "Edit-R", "OT-R"))
    lines += ["", "## 原因与归因边界", "",
              f"OT/Edit的D/R轮数、R接受量和对齐调用/候选/提议/接受量在{equal_counts}/48个请求-重复配对中完全相同，未观察到OT带来的额外接受优势。",
              f"两组对齐接受token均为{aligned['alignment_accepted_tokens']}；实际对齐查询OT={aligned['alignment_seconds']:.6f}s，Edit={edit['alignment_seconds']:.6f}s。",
              f"触发OT的项目仅{len(active_projects)}/12：{', '.join(active_projects) if active_projects else '无'}。",
              f"Suffix到OT：D轮{suffix['draft_rounds']}→{aligned['draft_rounds']}，R轮{suffix['reuse_rounds']}→{aligned['reuse_rounds']}，总轮{suffix['rounds']}→{aligned['rounds']}。",
              f"对齐接受量不等于相对Suffix新增的R接受量：全部R接受token为{suffix['reuse_accepted_tokens']}→{aligned['reuse_accepted_tokens']}。",
              "这说明接受更多对齐token、减少部分D轮，并不足以保证净加速；新增验证轮和查询成本必须同时考虑。"]
    if not summary["groups"]["ambiguity"]["OT-R"]["counters"]["alignment_calls"]:
        lines += ["ambiguity组未触发OT/Edit回退，且解码计数相同；小幅时间差不能归因于OT对齐。",
                  "多数项目在实际解码路径上没有进入对齐回退；未记录逐轮候选，因此不能进一步区分精确后缀已足够与触发时机的作用。"]
    lines += ["", "## 决定与限制", ""]
    if passed:
        lines += [f"预先指定工程门槛通过场景：{', '.join(passed)}。仅支持继续验证该受控场景，不能宣布通用OT优势。",
                  "下一阶段才考虑独立开发成本门控与另一个留出集；本轮没有新门控或生产保护。"]
    else:
        lines += ["两个预定候选场景均未达到继续投入门槛。本轮不扩大同类测试，不在这些答案上调参。",
                  "保留OT显式实验选项与负例；默认suffix。若近似对齐有局部信号，应先判断便宜Edit是否已足够。"]
    lines += ["", "这些新任务没有参与首轮调参，但仍是作者编写的受控任务，不证明真实流量、更多模型或并发性能。",
              "所有方法共享完整当前输入；本实验不证明历史知识不可替代或长期记忆的独特价值。",
              "运行时索引也包含当前prompt，因此无历史/无关历史控制仍可能复用请求中的源文，并不等于没有R机会。",
              "OT与Edit共享候选池/预算和数值阈值，分数尺度没有按各自最优点校准；本轮比较的是冻结实现。",
              "未保存在线逐轮候选轨迹，不把离线对齐机会当作实际在线触发位置或接受率。", "",
              "首次运行226399c在24个AR后因评测误用stop_reason字段停止，有效计时行0；原始失败证据保留。",
              "修复45ec2ef仅改评测为运行时实际finish_reason字段并补真实数据结构测试，任务/参数/门槛未变。",
              "修复后从新目录重跑全部参考与计时；两次24个AR的token/text/prompt/snapshot/结束原因/契约逐项相同。", "",
              "## 来源与复算", "", f"新AR总耗时：{sum(r['latency_seconds'] for r in references):.6f}s。",
              f"模型加载与输入身份预计算：{info['load_and_identity_seconds']:.6f}s。",
              f"实际GPU package SHA：`{info['source']['sha256']}`。",
              f"任务SHA：`{info['cases_sha256']}`。",
              f"运行前协议SHA：`{info['protocol_sha256']}`。",
              f"执行脚本SHA：`{info['runner_sha256']}`。", "",
              "原始参考、计时行、执行来源、完整测试结果与发布SHA清单保存在results/ot-validation。权重/数据库/大trace不发布。", "",
              "```bash", "python -m scripts.write_transport_validation_report", "```", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    diagnostic = {"comparisons": comparison, "contracts": summary["contracts"], "rows": len(rows),
                  "ot_edit_counter_pairs_equal": equal_counts, "alignment_active_projects": active_projects,
                  "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (source / "diagnostic.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/ot-validation")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/OT_R_VALIDATION_RESULTS.md")
    args = parser.parse_args()
    print(json.dumps(rebuild(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
