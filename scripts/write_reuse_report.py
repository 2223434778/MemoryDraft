"""Rebuild and audit the small synthetic R/DFlash development experiment offline."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from memorydraft.evaluation.reuse_cases import development_projects
from memorydraft.memory.store import MemoryEvent, Snapshot
from scripts.run_reuse_pilot import METHODS, summarize


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    path = Path(path)
    data = path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def audit(projects, initial, references, rows):
    if projects != development_projects():
        raise ValueError("Development tasks differ from the predeclared source")
    expected = {(p["id"], turn) for p in projects for turn in range(len(p["currents"]))}
    definitions = {p["id"]: p for p in projects}
    labels = {(r["project_id"], r["turn"]): r for r in references}
    if len(labels) != len(references) or set(labels) != expected:
        raise ValueError("Missing or duplicate frozen references")
    events = {event["user_id"]: MemoryEvent(**event) for event in initial}
    if len(events) != len(initial) or set(events) != {p["id"] for p in projects}:
        raise ValueError("Initial user history differs from the project set")
    snapshots = {}
    for project in projects:
        event = events[project["id"]]
        for turn in range(len(project["currents"])):
            order = 10 + 10 * turn
            if event.order >= order or event.user_id != project["id"]:
                raise ValueError("Future or cross-user memory")
            snapshot = Snapshot(project["id"], order, (event,))
            key = project["id"], turn
            snapshots[key] = snapshot.snapshot_id
            label = labels[key]
            if label["snapshot_id"] != snapshot.snapshot_id or label["order"] != order:
                raise ValueError("Frozen reference snapshot mismatch")
            event = MemoryEvent(project["id"], "artifact", order + 1, label["text"], project_id=project["id"])
    seen = set()
    for row in rows:
        key = row["project_id"], row["turn"]
        identity = (*key, row["method"], row["repeat"])
        if key not in expected or identity in seen or row["method"] not in METHODS or row["repeat"] != 0:
            raise ValueError("Unpaired, duplicate or unexpected replay row")
        seen.add(identity)
        label = labels[key]
        definition = definitions[row["project_id"]]
        if (not row["correct"] or row["token_ids"] != label["token_ids"]
                or row["prompt_sha256"] != label["prompt_sha256"] or row["snapshot_id"] != snapshots[key]
                or row["scenario"] != definition["scenario"] or row["order"] != label["order"]
                or row["budget"] != definition.get("budget", 384)):
            raise ValueError("Replay differs from fixed teacher prompt, tokens or legal snapshot")
        for event in row.get("rounds", []):
            before = event["target_cache_after"] - event["advance"]
            if (event["accepted"] > event["proposed"] or event["advance"] != event["accepted"] + 1
                    or event["draft_cache_before"] + event["pending_before"] != before
                    or event["pending_after"] != event["advance"] + (event["pending_before"] if event["route"] == "R" else 0)):
                raise ValueError("Invalid KV or accepted-prefix frontier")
            if event["route"] == "R" and event["source"] not in {"artifact", "context:current-prompt"}:
                raise ValueError("Unexpected retrieval source")
    if len(seen) != len(expected) * len(METHODS):
        raise ValueError("Incomplete paired development replay")
    return summarize(rows)


def ratio_interval(rows, left, right):
    ids = sorted({r["project_id"] for r in rows})
    pairs = np.array([[sum(r["latency_seconds"] for r in rows if r["project_id"] == p and r["method"] == m)
                       for m in (left, right)] for p in ids])
    samples = np.random.default_rng(20261018).integers(0, len(ids), size=(2000, len(ids)))
    totals = pairs[samples].sum(axis=1)
    low, high = np.quantile(totals[:, 0] / totals[:, 1], [.025, .975])
    return float(low), float(high)


def write_report(source, output):
    provenance = json.loads((source / "provenance.json").read_text())
    if provenance["status"] != "complete":
        raise ValueError("Experiment is not complete")
    projects = json.loads((source / "projects.json").read_text())
    initial = json.loads((source / "initial-history.json").read_text())
    references, rows = read(source / "references.jsonl"), read(source / "replay.jsonl")
    summary = audit(projects, initial, references, rows)
    if summary != json.loads((source / "summary.json").read_text()):
        raise ValueError("Saved summary does not reproduce")
    smoke, smoke_labels = read(source / "smoke.jsonl"), read(source / "smoke-references.jsonl")
    smoke_map = {r["case"]: r for r in smoke_labels}
    if len(smoke_map) != 8 or len(smoke) != 24 or len({(r['case'], r['method']) for r in smoke}) != 24:
        raise ValueError("Incomplete integration smoke")
    if any(not r["correct"] or r["token_ids"] != smoke_map[r["case"]]["token_ids"]
           or r["prompt_sha256"] != smoke_map[r["case"]]["prompt_sha256"] for r in smoke):
        raise ValueError("Integration smoke differs from AR")
    past = read(source / "past-generations.jsonl")
    lines = ["# R + DFlash 接入与开发场景结果", "", "2026-10-18。协议：[REUSE_PILOT_PROTOCOL](REUSE_PILOT_PROTOCOL.md)。",
             "", "本轮是合成开发探索：12 个项目、18 个当前请求、每方法1次计时。没有独立测试集，不能据此宣称稳定泛化或论文最终性能。",
             "", "## 正确性与计时", "", "A阶段8个请求×官方/适配关闭R/混合3方法=24次均与固定AR逐token一致；B阶段90次计时全部一致。",
             "服务器环境为固定Qwen3-8B/DFlash-b16、FP32/eager、Transformers4.57.3、A6000单GPU。",
             "", "| 场景 | 项目/请求 | AR秒 | DFlash秒 | Global秒 | Cursor秒 | Context秒 | DFlash/Cursor | Global/Cursor | Context/Cursor |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, group in summary.items():
        subset = [r for r in rows if name == "all" or r["scenario"] == name]
        t = group["times"]
        count = len(subset) // len(METHODS)
        lines.append(f"| {name} | {group['projects']}/{count} | " + " | ".join(f"{t[m]:.3f}" for m in METHODS)
                     + f" | {t['DFlash']/t['Cursor']:.3f}× | {t['Global']/t['Cursor']:.3f}× | {t['Context']/t['Cursor']:.3f}× |")
    lines += ["", "Global和Cursor读取相同合法历史+当前prompt，只有游标/局部重匹配不同。Context沿用Cursor策略，只读取当前prompt。",
              "官方DFlash与混合适配在输出尾部预算处理等方面存在工程差异；关闭R控制本轮只在A测量，不能把B中全部DFlash差值归为历史。Global/Cursor及Context/Cursor为同适配后端的机制比较。",
              "", "## 逐轮机制", "", "| 方法 | R/D轮数 | R接受/提议 | Draft前向次数 | 游标/局部/全局查找 | 特征缓冲强制消费 |",
              "|---|---:|---:|---:|---|---:|"]
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        events = [event for row in group for event in row.get("rounds", [])]
        r_events = [event for event in events if event["route"] == "R"]
        counts = Counter(event["route"] for event in events)
        lookups = [sum(row.get("lookups", {}).get(mode, 0) for row in group)
                   for mode in ("cursor", "local", "global_lookup")]
        if method in {"AR", "DFlash"}:
            lines.append(f"| {method} | 未插桩 | — | 未插桩 | — | — |")
        else:
            lines.append(f"| {method} | {counts['R']}/{counts['D']} | {sum(e['accepted'] for e in r_events)}/{sum(e['proposed'] for e in r_events)}"
                         f" | {counts['D']} | {'/'.join(map(str, lookups))} | {sum(r['buffer_flushes'] for r in group)} |")
    lines += ["", "## 不确定性与资源归因", "", "| 场景 | DFlash/Cursor 请求序列按项目重采样95%区间 |",
              "|---|---|"]
    for name in summary:
        subset = [r for r in rows if name == "all" or r["scenario"] == name]
        low, high = ratio_interval(subset, "DFlash", "Cursor")
        lines.append(f"| {name} | [{low:.3f}, {high:.3f}] |")
    lines += ["", "每类仅3个合成项目，单重复bootstrap不能刻画GPU重复运行方差；区间仅描述这批项目，开发结果用于选择后续场景。",
              "P1/P3已在当前prompt包含旧产物，其收益不能自动称为长期记忆收益。P2每项目顺序三轮，当前prompt有完整配置事实但不包含过去完整回答。每轮全部方法结束后才提交同一最终答案供下一轮使用。",
              "所有快照从初始合法过去记录及已经结束的先前轮次重建，离线核验prompt hash、token、配对、KV前沿和summary。",
              "", "## 完整性与成本边界", "",
              f"过去回答预算命中：{sum(r['budget_hit'] for r in past)}/{len(past)}；当前AR参考预算命中：{sum(r['budget_hit'] for r in references)}/{len(references)}。",
              f"已保存过去回答生成时间：{sum(r['latency_seconds'] for r in past):.3f}s；独立当前参考生成：{sum(r['latency_seconds'] for r in references):.3f}s，另行记录。",
              "在线快照、每请求冷索引、查询、prefill、起草、验证、有效特征复制/拼接、缓存恢复和解码均计入。索引bytes仅为数组载荷，不是完整CPU进程内存；未额外测TTFT。",
              "最初一次零命中位置查询失败发生于A阶段，修复后另起本批次；失败证据保留，没有并入性能。",
              "", "运行源码SHA-256：`" + provenance["source"]["source_sha256"] + "`。",
              "", "复算：`python scripts/write_reuse_report.py`。结果：`results/reuse-pilot/`。独立测试与场景结论需下一阶段冻结协议。", ""]
    output.write_bytes("\n".join(lines).encode())
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/reuse-pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/REUSE_PILOT_RESULTS.md")
    args = parser.parse_args()
    write_report(args.source, args.output)
