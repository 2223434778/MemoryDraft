"""Rebuild the small approximate-decoding report from completed probe rows."""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from memorydraft.evaluation.runtime_migration import read_rows, file_hash
from scripts.run_asd_splice_probe import IDS, METHODS, summarize

ROOT = Path(__file__).resolve().parents[1]


def analyze(source):
    info = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    if info["status"] != "complete" or not info["original_unchanged"]:
        raise ValueError("Incomplete probe or altered evidence")
    stages = {stage: read_rows(source / stage / "rows.jsonl") for stage in ("mechanism", "natural", "fixed")}
    references = json.loads((source / "references.json").read_text(encoding="utf-8"))
    for stage, rows in stages.items():
        repeats = 1 if stage == "mechanism" else 2
        keys = {(r["request_id"], r["method"], r["repeat"]) for r in rows}
        expected = {(q, m, i) for q in IDS for m in METHODS for i in range(repeats)}
        if len(rows) != len(expected) or keys != expected:
            raise ValueError("Missing or duplicate probe pairs")
        for row in rows:
            ref, state = references[row["request_id"]], row["asd"]
            if row["prompt_sha256"] != ref["prompt_sha256"] or row["snapshot_id"] != ref["snapshot_id"]:
                raise ValueError("Changed inputs or snapshots")
            if not 0 <= state["spent"] <= state["budget"] + 1e-9:
                raise ValueError("Invalid request budget")
            if row["strict_identity"] != (row["token_ids"] == ref["token_ids"]):
                raise ValueError("Incorrect strict-identity label")
            if stage == "fixed" and len(row["token_ids"]) != 64:
                raise ValueError("Unequal output work")
            if stage != "fixed" and "ASD" not in row["method"] and not row["strict_identity"]:
                raise ValueError("Failed zero-budget/strict control")
    mechanism = {(r["request_id"], r["method"]): r for r in stages["mechanism"]}
    for row in stages["natural"]:
        reference = mechanism[row["request_id"], row["method"]]
        if row["token_ids"] != reference["token_ids"] or row["contract"] != reference["contract"]:
            raise ValueError("Mechanism and timing outcomes differ")
    bridges, changes, blockers = {}, [], {}
    for method in METHODS:
        counts = dict(bridge_exceptions=0, history_exceptions=0, unlocked_blocks=0, unlocked_exact_history=0)
        for row in stages["mechanism"]:
            if row["method"] != method:
                continue
            for event in row["events"]:
                boundary = event.get("draft_prefix")
                if boundary is None:
                    continue
                accepted = event["accepted"]
                counts["bridge_exceptions"] += sum(event["mismatch"][:min(boundary, accepted)])
                counts["history_exceptions"] += sum(event["mismatch"][boundary:accepted])
                if event["strict_accepted"] < boundary < accepted:
                    counts["unlocked_blocks"] += 1
                    counts["unlocked_exact_history"] += sum(not v for v in event["mismatch"][boundary:accepted])
            if "ASD" in method and not row["strict_identity"]:
                old = references[row["request_id"]]["text"]
                changes.append(dict(request_id=row["request_id"], method=method,
                                    contract=row["contract"], old_tokens=len(references[row["request_id"]]["token_ids"]),
                                    new_tokens=len(row["token_ids"]),
                                    diff=list(difflib.unified_diff(old.splitlines(), row["text"].splitlines(),
                                                                  fromfile="strict", tofile="approximate", lineterm=""))))
        bridges[method] = counts
    for method in ("DFlash", "Suffix", "Guided", "Splice"):
        entries = []
        for row in stages["mechanism"]:
            if row["method"] != method:
                continue
            for event in row["events"]:
                i = event["strict_accepted"]
                if i >= len(event["mismatch"]) or event["generated_before"] + i >= len(row["token_ids"]):
                    continue  # Scores after an already emitted EOS are not opportunities.
                entries.append(dict(request_id=row["request_id"], generated_before=event["generated_before"],
                                    position=i, regret=event["regrets"][i], ratio=event["regrets"][i]/event["values"][i],
                                    draft_prefix=event.get("draft_prefix"), lookup=event["lookup"]))
        blockers[method] = dict(count=len(entries), minimum_regret=min((e["regret"] for e in entries), default=None),
                                minimum_ratio=min((e["ratio"] for e in entries), default=None),
                                budget8_eligible=sum(e["regret"] <= 8 and e["ratio"] <= .25 for e in entries),
                                entries=entries)
    return dict(stages={s: summarize(rows) for s, rows in stages.items()}, bridges=bridges,
                changes=changes, blockers=blockers, requests={q: {m: summarize([r for r in stages["natural"] if r["request_id"] == q and r["method"] == m])[m]
                                              for m in METHODS} for q in IDS}), stages, info


def render(analysis, info):
    natural, fixed, mechanism = (analysis["stages"][s] for s in ("natural", "fixed", "mechanism"))
    lines = ["# ASD 与历史拼接：直接组合实验", "", "2026-10-20。8个已观察请求/5项目；开发机制检查，不是独立泛化或无损加速结论。", "",
             "已完成72次机制、144次自然EOS计时、144次固定64token计时。全部32个零预算机制对照和64次自然EOS严格计时匹配旧AR；近似组允许偏离。自然计时与机制输出完全一致。", "",
             "ASD B=2、g=0.25、每块最多2例外。Run用已打分的连续匹配后缀取代剩余块长作为价值；无新模型、OT或风险预测器。所有在线成本计入，插桩时间不进入表。", "",
             "| 方法 | 自然EOS总秒（两重复） | 输出token（两重复） | 合同正确（8请求） | AR一致（8请求） | 机制验证轮 | 机制例外token | 固定64token TPS |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        n, f, a = natural[m], fixed[m], mechanism[m]
        lines.append(f"| {m} | {n['seconds']:.6f} | {n['tokens']} | {a['contract_correct']}/8 | {a['strict_identity']}/8 | {a['rounds']} | {a['relaxed_tokens']} | {f['tps']:.3f} |")
    lines += ["", "固定工作量每方法1024 token，忽略EOS；短控制中会越过自然结束，只作等工作量吞吐检查。自然EOS质量/时延另列，输出长度改变不能伪装成纯推理提速。", "",
              "| 比较 | 自然EOS时延变化 | 固定工作量TPS变化 |", "|---|---:|---:|"]
    for a, b in (("DFlash-ASD", "DFlash"), ("Suffix-ASD", "Suffix"), ("Guided-ASD", "Guided"),
                 ("Splice-ASD", "Splice"), ("Splice-ASD-Run", "Splice"), ("Splice-ASD", "Guided-ASD"),
                 ("Splice-ASD-Run", "Guided-ASD")):
        lines.append(f"| {a} / {b} | {(natural[a]['seconds']/natural[b]['seconds']-1)*100:+.2f}% | {(fixed[a]['tps']/fixed[b]['tps']-1)*100:+.2f}% |")
    lines += ["", "| 拼接方法 | Draft桥段例外 | 历史段例外 | 跨过桥段释放历史的块 | 这些块中新轨迹下精确匹配历史token |", "|---|---:|---:|---:|---:|"]
    for m in ("Splice", "Splice-ASD", "Splice-ASD-Run"):
        b = analysis["bridges"][m]
        lines.append(f"| {m} | {b['bridge_exceptions']} | {b['history_exceptions']} | {b['unlocked_blocks']} | {b['unlocked_exact_history']} |")
    if all(mechanism[m]["relaxed_tokens"] == 0 for m in METHODS):
        lines += ["", "本批自然EOS下全部方法均未接受近似例外，输出全部匹配AR；ASD没有改变验证轮数，时差只能体现额外计算/同步及测量波动，不能算作组合收益。", "",
                  "| 严格候选来源 | EOS前首次阻断数 | 最小regret | 最小regret/剩余长度 | B=8且g=0.25可放行 |", "|---|---:|---:|---:|---:|"]
        for m, values in analysis["blockers"].items():
            lines.append(f"| {m} | {values['count']} | {values['minimum_regret']:.3f} | {values['minimum_ratio']:.3f} | {values['budget8_eligible']} |")
        lines += ["", "这里的B=8是同状态分数检查，没有伪称重新跑了B=8计时。排除了自然EOS之后已不会交付的候选；不能把这些位置算成可挽救的尾部。", "",
                  "| 拼接失败请求 | 已生成 | Draft边界 | 严格接受 | 首次阻断regret |", "|---|---:|---:|---:|---:|"]
        for e in analysis["blockers"]["Splice"]["entries"]:
            if e["draft_prefix"] is not None:
                lines.append(f"| {e['request_id']} | {e['generated_before']} | {e['draft_prefix']} | {e['position']} | {e['regret']:.3f} |")
        lines += ["", "解释：被挡住的拼接不是低差距例外，单纯提高请求预算也受局部门槛阻挡。当前代码/配置任务没有支持ASD与拼接互补；这不否定ASD在其他输出分布/任务上的效果。"]
    lines += ["", "释放计数按实际提交轨迹定义；若接受例外，后缀精确匹配将以包含该例外的新前缀为条件，不保证原AR路径，也不能直接换算为全请求加速。", "",
              "| 请求 | Guided秒 | Guided-ASD秒 | Splice秒 | Splice-ASD秒 | Splice-ASD-Run秒 |", "|---|---:|---:|---:|---:|---:|"]
    for q, values in analysis["requests"].items():
        lines.append("| " + q + " | " + " | ".join(f"{values[m]['seconds']:.6f}" for m in ("Guided", "Guided-ASD", "Splice", "Splice-ASD", "Splice-ASD-Run")) + " |")
    lines += ["", "输出变化（同请求不同方法仍分别保留，不当作独立样本）：", ""]
    for change in analysis["changes"]:
        lines += [f"- `{change['request_id']}` / `{change['method']}`：合同正确={change['contract']['artifact_correct']}，token {change['old_tokens']}→{change['new_tokens']}。"]
    if not analysis["changes"]:
        lines += ["本批自然EOS输出无变化。"]
    lines += ["", "逐行差异及汇总见 `results/asd-splice/analysis.json`，原始逐请求结果保留，不删除任务失败。小规模两重复结果不作统计显著性或生产质量承诺。", "",
              f"运行 package SHA `{info['source']['sha256']}`。官方ASD核验提交 `{info['upstream_asd_commit']}`；当前为DFlash适配，不声称复现官方DSpark实验。", "",
              "重建：`python -m scripts.write_asd_splice_report`。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/asd-splice")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/ASD_SPLICE_RESULTS.md")
    args = parser.parse_args()
    analysis, _, info = analyze(args.source)
    (args.source / "analysis.json").write_bytes((json.dumps(analysis, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    args.report.write_bytes(render(analysis, info).encode("utf-8"))
    print(json.dumps({"report": str(args.report), "sha256": file_hash(args.report)}))


if __name__ == "__main__":
    main()
