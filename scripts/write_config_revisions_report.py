"""Rebuild public CI revision evidence, including legal history and task quality."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from memorydraft.evaluation.config_revisions import (METHODS_BY_MODE, answer_event, load_projects,
                                                     quality, working_event)
from memorydraft.memory.store import Snapshot
from scripts.run_config_revisions import save, summaries
from scripts.write_reuse_report import ratio_interval, read


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/research/config-revisions"


def audit(projects, references, rows, *, repeats, quality_version=2):
    definitions = {p["id"]: p for p in projects}
    keys = {(p["id"], mode, turn) for p in projects for mode in METHODS_BY_MODE
            for turn in range(2 if mode == "full" else 1)}
    labels = {(r["project_id"], r["mode"], r["turn"]): r for r in references}
    if len(labels) != len(references) or set(labels) != keys:
        raise ValueError("Missing, duplicate or unexpected references")
    for key, reference in labels.items():
        project_id, mode, turn = key
        project = definitions[project_id]
        events = [working_event(project, mode, turn)]
        if turn:
            previous = labels[(project_id, mode, turn - 1)]
            events.append(answer_event(project, mode, turn - 1, previous["text"]))
        order = 10 + 10 * turn
        snapshot = Snapshot(project_id + ":" + mode, order, tuple(sorted(events, key=lambda event: event.record_id)))
        if reference["snapshot_id"] != snapshot.snapshot_id or reference["order"] != order:
            raise ValueError("Reference used a future, cross-user or incorrect file/answer version")
        if reference["quality"] != quality(reference["text"], project, turn, mode, version=quality_version):
            raise ValueError("Task completion does not reproduce")
    seen = set()
    for row in rows:
        key = row["project_id"], row["mode"], row["turn"]
        identity = (*key, row["method"], row["repeat"])
        if (key not in keys or identity in seen or row["method"] not in METHODS_BY_MODE[row["mode"]]
                or row["repeat"] not in range(repeats)):
            raise ValueError("Unexpected, duplicate or unpaired method row")
        seen.add(identity)
        reference = labels[key]
        if (not row["correct"] or row["token_ids"] != reference["token_ids"] or row["text"] != reference["text"]
                or row["prompt_sha256"] != reference["prompt_sha256"] or row["snapshot_id"] != reference["snapshot_id"]
                or row["order"] != reference["order"] or row["budget"] != (3072 if row["mode"] == "full" else 768)
                or not 0 <= row["write_seconds"] <= row["latency_seconds"]):
            raise ValueError("Replay differs from the fixed Target or legal snapshot/cost contract")
        for event in row["rounds"]:
            before = event["target_cache_after"] - event["advance"]
            if (not 0 <= event["accepted"] <= event["proposed"] or event["advance"] != event["accepted"] + 1
                    or event["draft_cache_before"] + event["pending_before"] != before
                    or event["pending_after"] != event["advance"] + (event["pending_before"] if event["route"] == "R" else 0)):
                raise ValueError("Invalid accepted-prefix or KV frontier")
            if event["route"] == "R" and event["source"] not in {"artifact", "working-file", "context:current-prompt", "arctic:legal-resources"}:
                raise ValueError("Unexpected candidate resource")
    expected_rows = sum(len(METHODS_BY_MODE[mode]) for _, mode, _ in keys) * repeats
    if len(rows) != expected_rows:
        raise ValueError("Incomplete paired replay")
    return summaries(rows)


def write_report(source: Path, output: Path):
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    if provenance["status"] != "complete":
        raise ValueError("Do not report an incomplete experiment as performance evidence")
    if hashlib.sha256((DATA / "projects.json").read_bytes()).hexdigest() != provenance["data_sha256"]:
        raise ValueError("Frozen dataset changed since inference")
    projects = load_projects(DATA, provenance["split"])
    if projects != json.loads((source / "projects.json").read_text(encoding="utf-8")):
        raise ValueError("Project selection changed")
    references, rows = read(source / "references.jsonl"), read(source / "replay.jsonl")
    recorded_quality_version = provenance.get("quality_version", 1)
    summary = audit(projects, references, rows, repeats=provenance["repeats"], quality_version=recorded_quality_version)
    if summary != json.loads((source / "summary.json").read_text()):
        raise ValueError("Summary does not reproduce")
    definitions = {p["id"]: p for p in projects}
    corrected_references = [dict(r, quality=quality(r["text"], definitions[r["project_id"]], r["turn"], r["mode"]))
                            for r in references]
    corrections = [dict(project_id=old["project_id"], mode=old["mode"], turn=old["turn"],
                        recorded=old["quality"], corrected=new["quality"])
                   for old, new in zip(references, corrected_references) if old["quality"] != new["quality"]]
    if recorded_quality_version == 1:
        result = dict(recorded_quality_version=1, quality_version=2, reason="Append missing terminal LF before git apply; no output or timing changes",
                      runtime_source_sha256=provenance["source"]["source_sha256"], corrections=corrections)
        path = source / "quality-v2.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != result:
            raise ValueError("Quality correction evidence changed")
        save(path, result)
    references = corrected_references
    full = [r for r in rows if r["mode"] == "full"]
    label_full = [r for r in references if r["mode"] == "full"]
    label_diff = [r for r in references if r["mode"] == "diff"]
    t = summary["full"]["times"]
    lines = ["# 公开CI配置连续修订回放结果", "", "2026-10-18。协议：[CONFIG_REVISIONS_PROTOCOL](CONFIG_REVISIONS_PROTOCOL.md)。", "",
             f"阶段：{provenance['split']}；{len(projects)}个公开项目，每项目两轮完整文件修订和一项diff边界，"
             f"每方法{provenance['repeats']}重复，共{len(rows)}次计时。全部与各自固定Target AR参考逐token一致。",
             "这是基于真实仓库文件/提交构造的编辑API回放；请求和输出合同由实验指定，不是自然用户或部署服务日志。",
             "", "## 完整文件：全部在线成本", "", "| 方法 | 次数 | 秒 | DFlash/方法 |", "|---|---:|---:|---:|"]
    for method in METHODS_BY_MODE["full"]:
        lines.append(f"| {method} | {sum(r['method']==method for r in full)} | {t[method]:.3f} | {t['DFlash']/t[method]:.3f}× |")
    lines += ["", "| 对照/Global | 配对总时间比 | 按项目bootstrap95%区间 |", "|---|---:|---|"]
    for method in ("DFlash", "Adapted", "Context", "Arctic", "AR"):
        low, high = ratio_interval(full, method, "Global")
        lines.append(f"| {method}/Global | {t[method]/t['Global']:.3f}× | [{low:.3f}, {high:.3f}] |")
    lines += ["", "Global/Context使用相同算法，仅资源不同；当前输入已含完整旧文件和所需patch。"
              "因此Global快于DFlash本身不能证明额外历史有独立贡献，须直接看Context/Global。",
              "Arctic为官方线性后缀候选组件接入同一严格验证器；保留原生动态候选长度，但统一最大长度、准入门槛和资源。"
              "没有复现完整vLLM、树验证、在线输出树增长或其论文全部系统配置。", "",
              "## 逐项目结果", "", "| 项目 | 文件 | DFlash秒 | Global秒 | DFlash/Global | Context/Global | Arctic/Global |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for project in projects:
        group = [r for r in full if r["project_id"] == project["id"]]
        times = {m: sum(r["latency_seconds"] for r in group if r["method"] == m) for m in METHODS_BY_MODE["full"]}
        lines.append(f"| {project['repository']} | `{project['path']}` | {times['DFlash']:.3f} | {times['Global']:.3f} | "
                     f"{times['DFlash']/times['Global']:.3f}× | {times['Context']/times['Global']:.3f}× | {times['Arctic']/times['Global']:.3f}× |")
    lines += ["", "## 任务完成与截断", "",
              f"完整文件：原文件修改后文本一致 {sum(r['quality']['exact_file_or_patch'] for r in label_full)}/{len(label_full)}；"
              f"YAML映射可解析 {sum(r['quality']['yaml_mapping'] for r in label_full)}/{len(label_full)}；"
              f"YAML结构相等 {sum(r['quality']['yaml_equivalent'] for r in label_full)}/{len(label_full)}。",
              f"diff：patch文本一致 {sum(r['quality']['exact_file_or_patch'] for r in label_diff)}/{len(label_diff)}；"
              f"实际应用后得到期望文件 {sum(r['quality']['patch_applies_to_expected'] for r in label_diff)}/{len(label_diff)}。",
              f"参考输出达到预算 {sum(r['budget_hit'] for r in references)}/{len(references)}；所有计时达到预算 "
              f"{sum(r['budget_hit'] for r in rows)}/{len(rows)}。任务检查在计时外进行，不删除失败样本。",
              "YAML结构相等不保证注释/文本保留；不执行工作流，不声明CI或部署测试通过。所有方法输出相同，速度优化不改善Target任务能力。",
              f"原运行评分版本{recorded_quality_version}；本报告版本2。旧版未为无结尾LF的diff补patch终止符，"
              f"离线修正{len(corrections)}项应用检查；原参考/计时未修改，差异保存在quality-v2.json（仅旧版运行）。",
              "", "未完全符合文本合同的请求："]
    failures = [r for r in references if not r["quality"]["exact_file_or_patch"]]
    lines += [f"- `{r['project_id']}` / {r['mode']} / turn={r['turn']}：{r['quality']}" for r in failures] or ["- 无。"]
    lines += ["", "## diff边界：单列", "", "| 方法 | 秒 | DFlash/方法 |", "|---|---:|---:|"]
    td = summary["diff"]["times"]
    for method in METHODS_BY_MODE["diff"]:
        lines.append(f"| {method} | {td[method]:.3f} | {td['DFlash']/td[method]:.3f}× |")
    lines += ["", "## 机制及资源来源", "", "| 方法 | R/D轮数 | R接受/提议 | 来源轮数 | 写回秒 |", "|---|---:|---:|---|---:|"]
    for method in ("Adapted", "Global", "Context", "Arctic"):
        group = [r for r in full if r["method"] == method]
        events = [e for r in group for e in r["rounds"]]
        r_events = [e for e in events if e["route"] == "R"]
        counts = Counter(e["route"] for e in events)
        resources = dict(Counter(e["source"] for e in r_events))
        lines.append(f"| {method} | {counts['R']}/{counts['D']} | {sum(e['accepted'] for e in r_events)}/{sum(e['proposed'] for e in r_events)} | "
                     f"{resources} | {sum(r['write_seconds'] for r in group):.4f} |")
    lines += ["", "## 可复现性与边界", "",
              "固定Qwen3-8B/DFlash、FP32/eager、greedy、Transformers4.57.3、A6000 GPU0。"
              "每个方法/重复独立SQLite，首轮库创建、用户文件版本更新、冷索引、检索、所有前向、缓存追赶、解码和最终回答写入均计入。",
              f"独立参考构造时间 {sum(r['latency_seconds'] for r in references):.3f}s；模型加载 {provenance['load_seconds']:.3f}s，单列。",
              "历史从真实旧文件及已经完成的同方法回答重建，逐请求审核版本、用户、Target prompt、token、配对、KV前沿与summary。"
              "方法间的结果重复不增加独立项目数；项目间文件规模不同，主结果为配对总时间比。",
              "这批工作流维护及统一diff指令不能代表自然语言修改请求、一般部署配置或真实长期用户。"
              "给定完整diff本可由git apply直接完成；本回放衡量既定LLM输出路径的性能，不证明该任务必须调用LLM。"
              "未测生产精度、并发、TTFT或稳定尾延迟。CPU索引数组载荷与C++树估计的口径不同，不能据它们宣称内存优势。",
              "源码SHA-256：`" + provenance["source"]["source_sha256"] + "`。",
              "冻结数据SHA-256：`" + provenance["data_sha256"] + "`。", "",
              f"复算：`python -m scripts.write_config_revisions_report --source results/config-revisions-{provenance['split']}`。", ""]
    output.write_bytes("\n".join(lines).encode())
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/config-revisions-development")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    write_report(args.source, args.output or ROOT / ("docs/CONFIG_REVISIONS_" + args.source.name.rsplit("-", 1)[-1].upper() + "_RESULTS.md"))
