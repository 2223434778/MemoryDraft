"""Rebuild OT-R pilot results from strictly audited small evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from memorydraft.evaluation.runtime_migration import input_hashes, read_rows
from memorydraft.evaluation.transport_probe import METHODS, audit, probe_inputs


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    path = Path(path)
    data = path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    return json.loads(data)


def rebuild(source, output, root=ROOT):
    source, output, root = Path(source), Path(output), Path(root)
    tasks, _ = probe_inputs(root)
    info = read_json(source / "provenance.json")
    originals = {name: input_hashes(root / "results" / name) for name in ("reuse-confirmation", "infra-v0.3")}
    if (info["status"] != "complete" or not info["original_unchanged"] or info["fresh_ar_correct"] != len(tasks)
            or info["original_sha256"] != originals):
        raise ValueError("OT pilot is incomplete or original evidence differs")
    references = read_rows(source / "fresh-ar.jsonl")
    definitions = {t.request.id: t for t in tasks}
    if len(references) != len(tasks) or {r["request_id"] for r in references} != set(definitions):
        raise ValueError("Fresh AR checks are missing or duplicated")
    for row in references:
        task = definitions[row["request_id"]]
        if (row["token_ids"] != task.reference["token_ids"] or row["prompt_sha256"] != task.reference["prompt_sha256"]
                or row["snapshot_id"] != task.snapshot_id):
            raise ValueError("Fresh AR differs from frozen tokens/input/snapshot")
    rows = read_rows(source / "rows.jsonl")
    summary = audit(rows, tasks)
    if summary != read_json(source / "summary.json") or info["rows"] != len(rows):
        raise ValueError("Saved summary differs from audited pairs")
    offline = read_json(source / "offline-local-v1.json")
    if offline["original_sha256"] != originals or set(offline["per_request"]) != set(definitions):
        raise ValueError("Offline evidence uses different original workloads")
    lines = ["# OT-R 首轮接入与实验结果", "", "2026-10-18。已查看旧工作负载上的探索性工程实验，不是独立研究确认。", "",
             "infra 面试交付已单独保存：固定 `27e3619`，分支 `codex/infra-interview-v0.3`、标签 `infra-v0.3-interview`。",
             "本机独立仓库、ZIP、bundle 与校验清单见 [INFRA_INTERVIEW](INFRA_INTERVIEW.md)。OT 在独立研究分支与服务器目录运行。", "",
             "## 实现与正确性", "",
             "保留精确后缀快速路径；缺失时，OT 对最近32个已确认token与最多8个历史窗口作词法/位置对齐。",
             "双dustbin允许部分内容不匹配；POT 0.9.6.post1求解。每请求最多4次。对照edit使用相同候选池与预算。",
             "这不是语义embedding检索、完整OPW、SpecTr或新的接受条件；只提议一段历史原始连续token。",
             f"10/10新AR符合原token、Target输入和合法快照；{len(rows)}/{len(rows)}计时输出也严格一致。",
             "未修改Target prompt、greedy逐token验证、记录/特殊token边界、用户隔离及时间快照；不把当前答案写回过去。", "",
             "## A6000 同批端到端结果", "",
             "四方法、十请求、两重复，方法在请求/重复内随机交错。下面为同批总秒数，越低越好。", "",
             "| 场景 | DFlash | Suffix-R | Edit-R | OT-R | OT相对Suffix延迟变化 |",
             "|---|---:|---:|---:|---:|---:|"]
    diagnostics = {}
    for group, values in summary.items():
        seconds = [values[method]["seconds"] for method in METHODS]
        delta = seconds[3] / seconds[1] - 1
        diagnostics[group] = {"ot_vs_suffix_latency_change": delta,
                              "ot_vs_edit_latency_change": seconds[3] / seconds[2] - 1}
        lines.append(f"| {group} | " + " | ".join(f"{s:.6f}" for s in seconds) + f" | {delta:+.2%} |")
    lines += ["", "下面按旧JSON项目作事后拆分，展示收益是否集中；不是新场景确认。", "",
              "| 项目（两请求各两重复） | Suffix-R秒 | Edit-R秒 | OT-R秒 | OT相对Suffix | OT对齐接受token |",
              "|---|---:|---:|---:|---:|---:|"]
    projects = {}
    for project in sorted({r["project_id"] for r in rows if r["project_id"].startswith("confirm-")}):
        selected = [r for r in rows if r["project_id"] == project]
        seconds = {m: sum(r["latency_seconds"] for r in selected if r["method"] == m) for m in METHODS}
        accepted = sum(r["counters"].get("alignment_accepted_tokens", 0) for r in selected if r["method"] == "OT-R")
        delta = seconds["OT-R"] / seconds["Suffix-R"] - 1
        projects[project] = {"seconds": seconds, "ot_vs_suffix_latency_change": delta, "ot_alignment_accepted_tokens": accepted}
        lines.append(f"| {project} | {seconds['Suffix-R']:.6f} | {seconds['Edit-R']:.6f} | "
                     f"{seconds['OT-R']:.6f} | {delta:+.2%} | {accepted} |")
    lines += ["", "| 场景/方法 | Draft轮数 | R轮数 | R接受token | 对齐提议 | 对齐接受token | 对齐查询秒数 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for group in ("old-json", "infra-explain", "infra-copy"):
        for method in METHODS:
            counter = summary[group][method]["counters"]
            lines.append(f"| {group}/{method} | {counter['draft_rounds']} | {counter['reuse_rounds']} | "
                         f"{counter['reuse_accepted_tokens']} | {counter['alignment_proposals']} | "
                         f"{counter['alignment_accepted_tokens']} | {counter['alignment_seconds']:.6f} |")
    lines += ["", "对齐查询秒数已经包含在端到端时间内，不另外相减。另有每请求位置索引构造成本，也已计入总时间。",
              "旧JSON仅4个项目；两个demo也不是生产流量。重复不是独立请求，不据小时间差宣称稳定优势或普遍收益。", "",
              "## 离线机会诊断", "",
              "使用已冻结答案作事后评分，所有长度>=8前缀独立选择。在线四次预算、冷却与反馈未模拟。",
              "oracle枚举最好原始span或候选池span，只是机会诊断，不是接受率、在线token总量或速度界。", "",
              "| 场景 | 后缀缺失位置 | 原始span>=8 | 池中span>=8 | OT找回>=8 | edit找回>=8 | OT零接受提议 | edit零接受提议 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    groups = {"old-json": [v for k, v in offline["per_request"].items() if k.startswith("confirm-")],
              "infra-explain": [offline["per_request"]["explain"]], "infra-copy": [offline["per_request"]["copy"]]}
    for group, values in groups.items():
        columns = [sum(v[key] for v in values) for key in ("misses", "raw_ge8", "pool_ge8")]
        columns += [sum(v[method]["recovered_ge8"] for v in values) for method in ("ot", "edit")]
        columns += [sum(v[method]["zero_accept"] for v in values) for method in ("ot", "edit")]
        lines.append(f"| {group} | " + " | ".join(map(str, columns)) + " |")
    lines += ["", "OT能在部分编辑后找回精确后缀缺失的连续片段，但同池edit也能找回大部分。",
              "词袋式质量传输加位置先验仍可能对重复token选错接续位置；相似上下文并不保证后续相同。",
              "换行样例的Target会添加空格，OT没有生成这些缺失token的能力。",
              "不能把此前19.4%受控复用收益归于新OT，或借这批任务重新宣称长期历史必要性。", "",
              "## 首轮判断", "",
              "本轮显示局部接续恢复确实可能减少模型计算；不能仅根据离线总覆盖判断实际收益。",
              "收益集中在哪个项目应以上表为准，不将已看过数据的子集当成独立发现。",
              "即使R接受量为零，一轮失败R也会由Target补出一个token，因此Draft轮数下降本身不保证加速。",
              "换行负例中OT增加验证且没有减少Draft轮数，仍需要退化保护。",
              "保留OT为显式实验选项，默认仍为suffix，面试版不改。停止在这十个旧请求上调阈值/扩大重复。",
              "若继续研究，下一步是固定机制后用未参与调参的项目检验收益与误触发；尚无通用替换R的证据。", "",
              "## 成本、来源与复算", "",
              "相同Qwen3-8B/DFlash、本地固定权重、A6000 GPU0、FP32/eager、4线程、TF4.57.3。",
              "所有在线快照、tokenize、冷索引、对齐setup/lookup、起草、验证、KV维护、同步与decode计入。",
              "POT库与模型预加载、预热及新AR检查单列，无HTTP/profiler。",
              f"新AR总耗时{sum(r['latency_seconds'] for r in references):.6f}s；模型加载{info['load_seconds']:.6f}s。", "",
              f"GPU实际package SHA：`{info['source']['sha256']}`。",
              f"执行脚本SHA：`{info['runner_sha256']}`。",
              f"运行前协议SHA：`{info['protocol_sha256']}`。",
              "实际部署清单execution-source.json、原始测试XML、CPU诊断及压缩计时行分别保存；最终报告不覆盖运行来源。", "",
              "```bash", "python -m scripts.write_transport_report", "```", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    diagnostic = {"comparisons": diagnostics, "projects": projects, "rows": len(rows), "fresh_ar": len(references),
                  "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (source / "diagnostic.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8", newline="\n")
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/ot-r")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/OT_R_RESULTS.md")
    args = parser.parse_args()
    print(json.dumps(rebuild(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
