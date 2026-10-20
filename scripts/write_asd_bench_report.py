"""Report ASD on public tasks; retain original scores and explicit parser correction."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from memorydraft.evaluation.asd_bench import load_tasks, score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from scripts.run_asd_bench_probe import METHODS, summarize

ROOT = Path(__file__).resolve().parents[1]


def analyze(source):
    info = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    if info["status"] != "complete" or not info["original_unchanged"]:
        raise ValueError("Incomplete public-task probe")
    records = {r["id"]: r for r in load_tasks(ROOT / "data/research/asd-bench/records.json")}
    if info["data_sha256"] != file_hash(ROOT / "data/research/asd-bench/records.json"):
        raise ValueError("Changed public data slice")
    stages = {stage: read_rows(source / stage / "rows.jsonl") for stage in ("mechanism", "natural", "fixed")}
    cache, corrections = {}, []
    for stage, rows in stages.items():
        repeats = 1 if stage == "mechanism" else 2
        keys = {(r["request_id"], r["method"], r["repeat"]) for r in rows}
        if len(rows) != 16 * len(METHODS) * repeats or keys != {(q, m, i) for q in records for m in METHODS for i in range(repeats)}:
            raise ValueError("Missing or duplicate task/method/repeat")
        for row in rows:
            if not 0 <= row["asd"]["spent"] <= row["asd"]["budget"] + 1e-9:
                raise ValueError("Overspent request budget")
            if stage == "fixed":
                if len(row["token_ids"]) != 128:
                    raise ValueError("Unequal fixed output work")
                continue
            key = row["request_id"], row["text"]
            if key not in cache:
                cache[key] = score(records[row["request_id"]], row["text"])
            corrected = cache[key]
            if corrected != row["quality"]:
                corrections.append(dict(stage=stage, request_id=row["request_id"], method=row["method"], repeat=row["repeat"],
                                        original=row["quality"], corrected=corrected))
            row["quality"] = corrected
    references = json.loads((source / "references.json").read_text())
    mechanism = {(r["request_id"], r["method"]): r for r in stages["mechanism"]}
    for stage, rows in stages.items():
        for row in rows:
            ref = references[row["request_id"]]
            if (row["prompt_sha256"], row["snapshot_id"]) != (ref["prompt_sha256"], ref["snapshot_id"]):
                raise ValueError("Changed prompt or isolated history")
            if row["strict_identity"] != (row["token_ids"] == ref["token_ids"]):
                raise ValueError("Incorrect trajectory identity")
            if stage != "fixed" and "ASD" not in row["method"] and not row["strict_identity"]:
                raise ValueError("Strict retrieval changed output")
            if stage == "natural" and row["token_ids"] != mechanism[row["request_id"], row["method"]]["token_ids"]:
                raise ValueError("Mechanism and timed trajectories differ")
    attribution = {}
    for method in METHODS:
        counts = Counter()
        for row in stages["mechanism"]:
            if row["method"] != method:
                continue
            for event in row["events"]:
                counts["exception_blocks"] += int(event["relaxed"] > 0)
                boundary = event.get("draft_prefix")
                if boundary is not None:
                    a = event["accepted"]
                    counts["bridge_exceptions"] += sum(event["mismatch"][:min(a, boundary)])
                    counts["history_exceptions"] += sum(event["mismatch"][boundary:a])
                    counts["unlocked_history_blocks"] += int(event["strict_accepted"] < boundary < a)
        attribution[method] = dict(counts)
    per_request = {q: {m: dict(quality=mechanism[q, m]["quality"], tokens=len(mechanism[q, m]["token_ids"]),
                               rounds=mechanism[q, m]["counters"]["rounds"], asd=mechanism[q, m]["asd"],
                               seconds=sum(r["latency_seconds"] for r in stages["natural"] if r["request_id"] == q and r["method"] == m))
                       for m in METHODS} for q in records}
    continuation = {}
    for method in METHODS:
        selected = [r for r in stages["fixed"] if r["benchmark"] == "mbpp" and r["method"] == method]
        prefixes_match = all(r["token_ids"][:len(mechanism[r["request_id"], method]["token_ids"])]
                             == mechanism[r["request_id"], method]["token_ids"] for r in selected)
        forced = sum(len(r["token_ids"]) - len(mechanism[r["request_id"], method]["token_ids"]) for r in selected)
        continuation[method] = dict(rows=len(selected), natural_prefixes_match=prefixes_match,
                                    tokens_beyond_natural_eos=forced,
                                    fraction=forced / sum(len(r["token_ids"]) for r in selected))
    return dict(stages={s: summarize(rows) for s, rows in stages.items()}, attribution=attribution,
                requests=per_request, score_corrections=corrections,
                fixed_mbpp_continuation=continuation,
                grader_sha256=file_hash(ROOT / "memorydraft/evaluation/asd_bench.py")), info


def render(a, info):
    lines = ["# ASD 在 GSM8K / MBPP 上的组合小实验", "", "2026-10-20。GSM8K test前8题、MBPP官方test编号11–18；64机制+128自然EOS+128固定128token，共320次生成。", "",
             "Qwen3-8B/DFlash、FP32/eager、greedy、非思考，ASD B=8/g=0.25/M=2。任务来自ASD使用的benchmark，模型/起草器/提示不等同其原论文复现。", "",
             "每题独立空历史，拼接只检索当前题目文本/公开测试；参考答案及其他生成不入索引。DFlash strict为本批参考，没有额外生成AR。", "",
             "| 任务 | 方法 | 自然EOS秒（两重复） | 输出token（两重复） | 正确（8题） | strict一致（8题） | 机制轮数 | 机制例外token | 截断（8题） | 固定128token TPS |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in ("gsm8k", "mbpp"):
        for m in METHODS:
            n, f, d = (a["stages"][s][group][m] for s in ("natural", "fixed", "mechanism"))
            lines.append(f"| {group} | {m} | {n['seconds']:.6f} | {n['tokens']} | {d['quality_correct']}/8 | {d['strict_identity']}/8 | {d['rounds']} | {d['relaxed_tokens']} | {d['length_stops']} | {f['tps']:.3f} |")
    lines += ["", "正确性按自然EOS的最终数值/MBPP公开测试评分；不混合两类质量指标。固定128token忽略EOS，仅报告等工作量吞吐，尤其短代码可能越过正常结束，不作任务质量证据。全部在线生成成本计入，评分/加载/插桩时间不计入正式生成时延。", "",
              "| 任务 | 比较 | 自然EOS时延变化 | 固定工作量TPS变化 |", "|---|---|---:|---:|"]
    for group in ("gsm8k", "mbpp"):
        for candidate, base in (("DFlash-ASD", "DFlash"), ("Splice-ASD", "Splice"), ("Splice-ASD", "DFlash-ASD"), ("Splice", "DFlash")):
            n, f = (a["stages"][s][group] for s in ("natural", "fixed"))
            lines.append(f"| {group} | {candidate}/{base} | {(n[candidate]['seconds']/n[base]['seconds']-1)*100:+.2f}% | {(f[candidate]['tps']/f[base]['tps']-1)*100:+.2f}% |")
    strict_tail, asd_tail = (a["fixed_mbpp_continuation"][m] for m in ("DFlash", "DFlash-ASD"))
    lines += ["", "自然输出长度改变时，总时延变化包含路径/长度变化，不可全归为每token加速；固定吞吐另看。小样本两重复不做显著性或质量非劣性保证。", "",
              f"MBPP固定工作量中，strict有{strict_tail['fraction']*100:.1f}%、ASD有{asd_tail['fraction']*100:.1f}%的token在自然EOS之后；64/64固定行的自然输出前缀与相应自然EOS方法一致。该口径下的大吞吐增幅覆盖了大量强制续写，不能解读为正常代码请求获得同等加速；正常请求以自然EOS时延为准。", "",
              "| 方法 | 机制R轮 | 内部拼接提议 | 机制历史段提交token | 含例外的块 | Draft桥段例外 | 历史段例外 | 跨桥释放历史块 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        c, d = a["attribution"][m], a["stages"]["mechanism"]["all"][m]
        lines.append(f"| {m} | {d['reuse_rounds']} | {d['splice_proposals']} | {d['splice_history_committed']} | {c.get('exception_blocks',0)} | {c.get('bridge_exceptions',0)} | {c.get('history_exceptions',0)} | {c.get('unlocked_history_blocks',0)} |")
    if not any(a["stages"]["mechanism"]["all"][m]["splice_proposals"] for m in METHODS):
        lines += ["", "本批内部拼接提议为0，没有验证出拼接与ASD的互补收益。若出现加速，应归于ASD验证策略；普通精确R的少量触发另计在R轮，不能混称内部拼接。"]
    lines += ["", "这里的历史段指拼接结构中的复制段，资源本轮仅当前输入；近似接受后的精确匹配基于新前缀，不保证strict轨迹。", "",
              "GSM8K评分器修正：原提示中的占位符使部分输出包含 `<answer>…</answer>`，初版解析漏识别标签及boxed货币数值。离线修正仅接受已明确输出的数字，未改输入、生成、阈值或标签；原始评分仍在rows中，修正明细在quality-v2.json。", "",
              f"本轮共{len(a['score_corrections'])}条机制/重复记录的评分结构得到修正（重复不增加独立题数）。最终报告使用修正后的评分，grader SHA `{a['grader_sha256']}`。", "",
              "| 请求 | strict正确 | ASD正确 | strict token→ASD token | strict轮→ASD轮 | ASD例外 |", "|---|---|---|---|---|---:|"]
    for q, rows in a["requests"].items():
        s, d = rows["DFlash"], rows["DFlash-ASD"]
        lines.append(f"| {q} | {s['quality']['correct']} | {d['quality']['correct']} | {s['tokens']}→{d['tokens']} | {s['rounds']}→{d['rounds']} | {d['asd']['relaxed_tokens']} |")
    lines += ["", f"执行runtime package SHA `{info['source']['sha256']}`，来源提交/数据SHA/旧证据保护见provenance与source-manifest。", "",
              "重建：`python -m scripts.write_asd_bench_report`。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/asd-bench")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/ASD_BENCH_RESULTS.md")
    args = parser.parse_args()
    a, info = analyze(args.source)
    (args.source / "analysis.json").write_bytes((json.dumps(a, indent=2, ensure_ascii=False) + "\n").encode())
    (args.source / "quality-v2.json").write_bytes((json.dumps(a["score_corrections"], indent=2, ensure_ascii=False) + "\n").encode())
    args.report.write_bytes(render(a, info).encode())
    print(json.dumps({"report": str(args.report), "sha256": file_hash(args.report)}))


if __name__ == "__main__":
    main()
