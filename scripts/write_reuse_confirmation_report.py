"""Audit the frozen independent structured-revision confirmation and rebuild its report."""
import argparse
from collections import Counter
import json
from pathlib import Path

from memorydraft.evaluation.reuse_cases import confirmation_projects
from scripts.run_reuse_pilot import CONFIRMATION_METHODS
from scripts.write_reuse_report import audit, ratio_interval, read


ROOT = Path(__file__).resolve().parents[1]


def write_report(source, output):
    provenance = json.loads((source / "provenance.json").read_text())
    if provenance["status"] != "complete" or provenance["stage"] != "confirmation":
        raise ValueError("Confirmation is not complete")
    rows, references = read(source / "replay.jsonl"), read(source / "references.jsonl")
    initial = json.loads((source / "initial-history.json").read_text())
    projects = json.loads((source / "projects.json").read_text())
    summary = audit(projects, initial, references, rows, expected_projects=confirmation_projects(),
                    methods=CONFIRMATION_METHODS, repeats=3)
    if summary != json.loads((source / "summary.json").read_text()):
        raise ValueError("Confirmation summary differs from audited rows")
    past = read(source / "past-generations.jsonl")
    times = summary["all"]["times"]
    lines = ["# 结构化产物连续修订：独立项目小规模确认", "", "2026-10-18。运行前协议：[REUSE_CONFIRMATION_PROTOCOL](REUSE_CONFIRMATION_PROTOCOL.md)。",
             "", "四个新合成项目、四种JSON schema，每项目两轮修订、三重复、四方法，共96次计时。项目未用于开发选择；规模小，不代表自然用户数据或通用场景泛化。",
             "核心循环和R参数沿用开发阶段；选用简单全局匹配，未使用游标、局部重匹配或新训练。所有方法与相同Target AR参考逐token一致。",
             "", "## 完整在线成本", "", "| 方法 | 次数 | 总秒数 | 官方DFlash/方法 |",
             "|---|---:|---:|---:|"]
    for method in CONFIRMATION_METHODS:
        lines.append(f"| {method} | 24 | {times[method]:.3f} | {times['DFlash']/times[method]:.3f}× |")
    lines += ["", "Adapted关闭R，用于检查接入开销；Global读取合法过去回答和当前prompt；Context使用完全相同全局匹配策略，仅移除过去回答。每次在线重新构建冷索引。",
              "", "| 直接比较 | baseline/方法 | 项目配对bootstrap95%区间 |", "|---|---:|---|"]
    intervals = {}
    for left, right in (("DFlash", "Global"), ("Adapted", "Global"), ("Context", "Global"), ("DFlash", "Adapted")):
        low, high = ratio_interval(rows, left, right)
        intervals[(left, right)] = low, high
        lines.append(f"| {left}/{right} | {times[left]/times[right]:.3f}× | [{low:.3f}, {high:.3f}] |")
    lines += ["", "区间以4个项目为重采样单位，每项目包含两轮与全部重复。重复不是独立用户；样本量仍很小。",
              "", "## 各项目及机制", "", "| 项目 | DFlash秒 | Adapted秒 | Global秒 | Context秒 | DFlash/Global | Context/Global |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for project in projects:
        subset = [r for r in rows if r["project_id"] == project["id"]]
        t = {m: sum(r["latency_seconds"] for r in subset if r["method"] == m) for m in CONFIRMATION_METHODS}
        lines.append(f"| {project['id']} | " + " | ".join(f"{t[m]:.3f}" for m in CONFIRMATION_METHODS)
                     + f" | {t['DFlash']/t['Global']:.3f}× | {t['Context']/t['Global']:.3f}× |")
    lines += ["", "| 方法 | R/D轮数 | R接受/提议 | 历史/当前prompt来源R轮数 |",
              "|---|---:|---:|---:|"]
    for method in ("Adapted", "Global", "Context"):
        events = [e for row in rows if row["method"] == method for e in row["rounds"]]
        c = Counter(e["route"] for e in events)
        r = [e for e in events if e["route"] == "R"]
        sources = Counter(e["source"] for e in r)
        lines.append(f"| {method} | {c['R']}/{c['D']} | {sum(e['accepted'] for e in r)}/{sum(e['proposed'] for e in r)}"
                     f" | {sources['artifact']}/{sources['context:current-prompt']} |")
    parseable = 0
    for reference in references:
        try:
            json.loads(reference["text"])
            parseable += 1
        except json.JSONDecodeError:
            pass
    supported = intervals[("DFlash", "Global")][0] > 1 and intervals[("Adapted", "Global")][0] > 1
    memory = intervals[("Context", "Global")][0] > 1
    lines += ["", "## 解释与限制", "",
              ("本小批次支持该合成结构化修订场景中Global相对官方DFlash及关闭R适配的净收益。" if supported else
               "本小批次尚未同时证明Global相对官方DFlash及关闭R适配的稳定净收益。"),
              ("同策略Context消融支持过去完整回答在当前事实描述之外的独立时间收益。" if memory else
               "同策略Context消融尚未证明过去完整回答的稳定独立时间收益。"),
              "这不是新概率反馈模型的收益，也不是已击败全部检索起草方法；Global为现有接续复用与DFlash的场景适配。独立于开发项目，但四种schema仍属于受控JSON工作负载。",
              f"当前AR参考JSON可解析：{parseable}/{len(references)}；预算命中：{sum(r['budget_hit'] for r in references)}/{len(references)}；过去回答预算命中：{sum(r['budget_hit'] for r in past)}/{len(past)}。",
              "可解析只说明格式完整，不替代全面任务语义评测；由于输出与固定AR一致，性能优化不改善Target自身的任务质量。",
              "所有在线tokenization、快照、冷索引、查找、前向、特征维护、回切和解码计入。未额外测TTFT；CPU索引bytes为数组载荷而非完整进程RSS。",
              f"离线过去回答构造{sum(r['latency_seconds'] for r in past):.3f}s，当前AR参考{sum(r['latency_seconds'] for r in references):.3f}s，另行记录。",
              "每轮参考仅用于评分，全部方法及重复结束后才成为下一轮过去记录；离线重新构造快照并核对全部token、prompt、配对、场景和KV前沿。",
              "", "确认运行源码SHA-256：`" + provenance["source"]["source_sha256"] + "`。",
              "接入检查来源SHA-256：`" + provenance["integration_reference_source"] + "`。",
              "", "复算：`python -m scripts.write_reuse_confirmation_report`。结果目录：`results/reuse-confirmation/`。", ""]
    output.write_bytes("\n".join(lines).encode())
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/reuse-confirmation")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/REUSE_CONFIRMATION_RESULTS.md")
    args = parser.parse_args()
    write_report(args.source, args.output)
