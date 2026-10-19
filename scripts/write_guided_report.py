"""Rebuild the guided-retrieval report from frozen references and paired results."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from memorydraft.evaluation.guided_validation import METHODS, audit, load_inputs
from memorydraft.evaluation.runtime_migration import file_hash, input_hashes, read_rows
from memorydraft.evaluation.transport_validation import attach_references
from memorydraft.runtime.benchmark import percentile
from scripts.diagnose_transport_online import inputs as development_inputs, write


ROOT = Path(__file__).resolve().parents[1]


def checked_info(path, stage):
    info = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
    if (info["stage"] != stage or info["cases_sha256"] != file_hash(ROOT / "data/research/ot-guided/cases.json")
            or info["protocol_sha256"] != file_hash(ROOT / "docs/OT_GUIDED_PROTOCOL.md")
            or set(info["original_sha256"]) != {"ot-r", "ot-validation", "ot-diagnosis", "reuse-confirmation", "infra-v0.3"}
            or any(input_hashes(ROOT / "results" / name) != hashes for name, hashes in info["original_sha256"].items())):
        raise ValueError("Changed frozen source evidence")
    return info


def compare_projects(rows, tasks, baseline):
    projects = sorted({t.project_id for t in tasks if t.group == "edited-continuation"})
    times = {p: {m: sum(r["latency_seconds"] for r in rows if r["project_id"] == p and r["method"] == m)
                 for m in ("Guided-OT", baseline)} for p in projects}

    def ratio(selected):
        return sum(times[p]["Guided-OT"] for p in selected) / sum(times[p][baseline] for p in selected)

    distribution = [ratio(sample) for sample in itertools.product(projects, repeat=len(projects))]
    return dict(ratio=ratio(projects), projects_faster=sum(times[p]["Guided-OT"] < times[p][baseline] for p in projects),
                project_bootstrap_95=[percentile(distribution, .025), percentile(distribution, .975)])


def rebuild(source, output):
    source, output = Path(source), Path(output)
    development, holdout = source / "development", source / "holdout"
    dev_info, info = checked_info(development, "development"), checked_info(holdout, "holdout")
    if (dev_info["status"] != "complete" or dev_info["rows"] != 32 or not dev_info["original_unchanged"]
            or info["source"] != dev_info["source"] or info["runner_sha256"] != dev_info["runner_sha256"]):
        raise ValueError("Development and holdout used different or incomplete implementations")
    dev_tasks, _ = development_inputs(ROOT)
    dev_rows = read_rows(development / "rows.jsonl")
    dev_summary = audit(dev_rows, dev_tasks, METHODS[1:], 1)
    if dev_summary != json.loads((development / "summary.json").read_text(encoding="utf-8")):
        raise ValueError("Changed development summary")
    tasks, _ = load_inputs(ROOT / "data/research/ot-guided/cases.json")
    tasks = attach_references(tasks, read_rows(holdout / "fresh-ar.jsonl"))
    contracts = sum(t.reference["contract"]["artifact_correct"] for t in tasks)
    stops = sum(t.reference["finish_reason"] != "eos" for t in tasks)
    lines = ["# DFlash前缀引导的OT/Edit：机制验证", "", "2026-10-19。旧八请求开发回放与新八个受控项目验证分开；不是自然生产流量。", "",
             "精确suffix优先。失败后用已有DFlash前4候选检索一致接续，再作OT或Edit上下文排序；候选需长于原Draft块。",
             "引导hint未确认，Target仍逐token严格验证。DFlash前向成本、hint同步、检索和验证均计入。", "",
             "## 开发检查", "", "32/32输出与原AR的token/prompt/snapshot严格一致。以下时间为已看过任务上的开发记录，不当作独立确认。", "",
             "| 方法 | 秒 | 解码轮 | 对齐提议 | 对齐接受token |", "|---|---:|---:|---:|---:|"]
    for method, value in dev_summary["all"].items():
        c = value["counters"]
        lines.append(f"| {method} | {value['seconds']:.6f} | {c['rounds']} | {c['alignment_proposals']} | {c['alignment_accepted_tokens']} |")
    lines += ["", "## 新请求验证", "", f"16个新AR完成：契约{contracts}/16，非EOS结束{stops}/16。"]
    if info["status"] == "stopped_target_contract":
        if contracts == 16 and stops == 0 or info["rows"] != 0:
            raise ValueError("Invalid Target-contract stop")
        lines += ["按预定协议停止正式计时，没有160次性能结果；不改题或调参后重报本批确认。"]
        diagnostic = dict(status=info["status"], contracts=contracts, non_eos=stops, engineering_gate=False)
    else:
        if (info["status"] != "complete" or info["rows"] != 160 or not info["original_unchanged"]
                or info["fresh_ar_correct"] != 16 or contracts != 16 or stops):
            raise ValueError("Incomplete holdout experiment")
        rows = read_rows(holdout / "rows.jsonl")
        summary = audit(rows, tasks, METHODS, 2)
        if summary != json.loads((holdout / "summary.json").read_text(encoding="utf-8")):
            raise ValueError("Changed holdout summary")
        comparisons = {baseline: compare_projects(rows, tasks, baseline) for baseline in ("Suffix-R", "OT-R", "Guided-Edit")}
        gate = (comparisons["Suffix-R"]["ratio"] <= .97 and comparisons["Guided-Edit"]["ratio"] <= .99
                and comparisons["Suffix-R"]["projects_faster"] >= 3
                and comparisons["Guided-Edit"]["projects_faster"] >= 3)
        diagnostic = dict(status="complete", contracts=contracts, non_eos=stops, engineering_gate=gate,
                          comparisons=comparisons, summary=summary)
        paired = {(r["request_id"], r["repeat"], r["method"]): r for r in rows}
        count_keys = ("rounds", "draft_rounds", "reuse_rounds", "reuse_accepted_tokens", "alignment_calls",
                      "alignment_candidates", "alignment_proposals", "alignment_accepted_tokens",
                      "guided_lookups", "draft_forward_rounds")
        equal_pairs = sum(all(r["counters"].get(k, 0) == paired[r["request_id"], r["repeat"], "Guided-Edit"]["counters"].get(k, 0)
                              for k in count_keys) for r in rows if r["method"] == "Guided-OT")
        diagnostic["guided_equal_counter_pairs"] = equal_pairs
        lines += ["160/160计时输出与新AR逐token/prompt/snapshot严格一致；每请求两重复，方法交错。", "",
                  "| 场景 | DFlash秒 | Suffix秒 | 原OT秒 | Guided-Edit秒 | Guided-OT秒 | 引导OT相对Suffix | 引导OT相对引导Edit |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for group in ("all", "edited-continuation", "control"):
            values = summary[group]
            times = [values[m]["seconds"] for m in METHODS]
            lines.append(f"| {group} | " + " | ".join(f"{s:.6f}" for s in times)
                         + f" | {(times[4]/times[1]-1)*100:+.2f}% | {(times[4]/times[3]-1)*100:+.2f}% |")
        lines += ["", "edited-continuation工程门槛：降低>=3% Suffix延迟、>=1% Guided-Edit延迟，分别至少3/4项目更快，且8/8契约通过。",
                  f"本轮门槛：{'通过' if gate else '未通过'}。四项目bootstrap仅作描述，不作稳定性或生产泛化声明。", "",
                  "| 对照 | Guided-OT/对照 | 更快项目 | 项目bootstrap 95% |", "|---|---:|---:|---|"]
        for baseline, value in comparisons.items():
            low, high = value["project_bootstrap_95"]
            lines.append(f"| {baseline} | {value['ratio']:.6f} | {value['projects_faster']}/4 | [{low:.4f}, {high:.4f}] |")
        lines += ["", "## 在线计算", "", "D路线数与实际Draft前向数分开。原OT的R跳过Draft；引导R已支付该轮Draft成本。", "",
                  "| 场景/方法 | D路线 | R路线 | 实际Draft前向 | R接受token | 对齐调用 | 对齐提议 | 对齐接受token | 查询秒 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for group in ("edited-continuation", "control"):
            for method in METHODS:
                c = summary[group][method]["counters"]
                forwards = c["draft_forward_rounds"] if method.startswith("Guided-") else c["draft_rounds"]
                lines.append(f"| {group}/{method} | {c['draft_rounds']} | {c['reuse_rounds']} | {forwards} | {c['reuse_accepted_tokens']} | {c['alignment_calls']} | {c['alignment_proposals']} | {c['alignment_accepted_tokens']} | {c['alignment_seconds']:.6f} |")
        lines += ["", "## 逐项目", "", "| 项目 | Suffix秒 | 原OT秒 | Guided-Edit秒 | Guided-OT秒 | 引导OT相对Suffix |", "|---|---:|---:|---:|---:|---:|"]
        for project in sorted({t.project_id for t in tasks}):
            times = {m: sum(r["latency_seconds"] for r in rows if r["project_id"] == project and r["method"] == m) for m in METHODS[1:]}
            lines.append(f"| {project} | " + " | ".join(f"{times[m]:.6f}" for m in METHODS[1:])
                         + f" | {(times['Guided-OT']/times['Suffix-R']-1)*100:+.2f}% |")
        c, s = summary["edited-continuation"]["Guided-OT"]["counters"], summary["edited-continuation"]["Suffix-R"]["counters"]
        lines += ["", "## 解释与决定", "",
                  f"主场景总解码轮{s['rounds']}→{c['rounds']}，实际Draft前向仍为{s['draft_rounds']}→{c['draft_forward_rounds']}。收益对应较长接续减少Target验证轮，不能说节省了这些Draft前向。",
                  f"引导OT与引导Edit的轮数、接受量、调用/候选/提议及实际Draft前向在{equal_pairs}/32个配对中相同；未保存逐候选trace，不据计数相同声称候选内容完全相同。",
                  "新主场景的TOML和JSON项目出现局部收益，另两个没有净收益；格式拆分是事后观察，不能作为新场景确认。",
                  "本批没有显示OT排序相对便宜Edit的额外接受收益。开发阶段OT曾比Edit多接受32token，但这没有在新请求中重现。",
                  "保留前缀引导作为显式工程实验选项；默认suffix。不扩大同机制OT试验，也不将引导带来的收益归为OT独有贡献。"]
    lines += ["", "## 来源与限制", "", "所有任务为作者编写；显式当前源文也进入检索资源，不能据此证明长期历史不可替代。",
              "开发样例已用于机制设计，新请求输出此前未观察。未在新答案上调整参数。OT/Edit共用数值阈值，比较固定实现而非各自调优上限。",
              "引导机制可能改善定位，也可能因错误hint排除好候选；候选更长不保证净加速。只依据全部在线成本判断。",
              "默认suffix和独立infra面试版保留。", "",
              f"实际package SHA：`{info['source']['sha256']}`。任务SHA：`{info['cases_sha256']}`。协议SHA：`{info['protocol_sha256']}`。", "",
              "```bash", "python -m scripts.write_guided_report", "```", ""]
    write(source / "diagnostic.json", diagnostic)
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/ot-guided")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/OT_GUIDED_RESULTS.md")
    args = parser.parse_args()
    rebuild(args.source, args.output)


if __name__ == "__main__":
    main()
