"""Audit the complete frozen ASD budget probe and publish natural-EOS results."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from memorydraft.evaluation.asd_budget import load_tasks, score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from scripts.run_asd_budget_probe import POLICIES

ROOT = Path(__file__).resolve().parents[1]


def audit_events(row):
    spent, exceptions, shortened = 0., 0, 0
    local = Counter()
    threshold = None
    for event in row["events"]:
        a, s = event["accepted"], event["strict_accepted"]
        mismatch, regrets = event["mismatch"], event["regrets"]
        charge = sum(r for r,m in zip(regrets[:a], mismatch[:a]) if m)
        count = sum(mismatch[:a])
        if count > 2 or any(r / v > .25 + 1e-9 for r,v,m in zip(regrets[:a], event["values"][:a], mismatch[:a]) if m):
            raise ValueError("Violated local gate or exception cap")
        if abs(charge - event["regret"]) > 1e-8 or count != event["relaxed"]:
            raise ValueError("Incorrect charged prefix")
        if not 0 <= a <= event["feasible_accepted"] <= len(mismatch):
            raise ValueError("Accepted outside feasible prefix")
        running = spent
        for i in range(a):
            if mismatch[i]:
                running += regrets[i]
            if threshold is None and running >= 7:
                threshold = (event["generated_before"] + i + 1) / len(row["token_ids"])
        spent += charge
        exceptions += count
        shortened += a < event["feasible_accepted"]
        if abs(spent - event["spent"]) > 1e-8 or spent > row["asd"]["budget"] + 1e-8:
            raise ValueError("Incorrect persistent ledger")
        if count:
            gain = max(0, a - s)
            local["paid_blocks"] += 1
            local["extra_proposal_tokens_same_state"] += gain
            local["paid_blocks_gain_le_2"] += gain <= 2
            if gain <= 2:
                local["low_gain_regret"] += charge
    if (abs(spent - row["asd"]["spent"]) > 1e-8 or exceptions != row["asd"]["relaxed_tokens"]
            or shortened != row["asd"]["shortened_blocks"] or len(row["events"]) != row["counters"]["rounds"]):
        raise ValueError("Event/summary mismatch")
    return {**dict(local), "fraction_at_spent7": threshold}


def analyze(source):
    data = ROOT / "data/research/asd-budget/records.json"
    records = {r["id"]: r for r in load_tasks(data)}
    output, provenance, requests = {}, {}, {}
    for benchmark in ("gsm8k", "math500"):
        directory = source / benchmark
        info = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        if info["status"] != "complete" or not info["original_unchanged"] or info["data_sha256"] != file_hash(data):
            raise ValueError("Incomplete probe or changed frozen inputs")
        if info["policies"] != {k: asdict(v) for k,v in POLICIES.items()}:
            raise ValueError("Changed frozen policies")
        provenance[benchmark] = info
        ids = {q for q,r in records.items() if r["benchmark"] == benchmark}
        stages = {s: read_rows(directory / s / "rows.jsonl") for s in ("mechanism", "natural")}
        refs = json.loads((directory / "references.json").read_text(encoding="utf-8"))
        mechanism = {(r["request_id"],r["method"]):r for r in stages["mechanism"]}
        for stage, rows in stages.items():
            repeats = 1 if stage == "mechanism" else 2
            expected = {(q,m,i) for q in ids for m in POLICIES for i in range(repeats)}
            if len(rows) != len(expected) or {(r["request_id"],r["method"],r["repeat"]) for r in rows} != expected:
                raise ValueError("Missing or duplicate request/method/repeat")
            for row in rows:
                q, m = row["request_id"], row["method"]
                ref = refs[q]
                if (row["prompt_sha256"],row["snapshot_id"]) != (ref["prompt_sha256"],ref["snapshot_id"]):
                    raise ValueError("Changed Target input or isolated history")
                if row["strict_identity"] != (row["token_ids"] == ref["token_ids"]):
                    raise ValueError("Incorrect identity label")
                if m == "DFlash" and not row["strict_identity"]:
                    raise ValueError("Strict reference changed")
                if not 0 <= row["asd"]["spent"] <= row["asd"]["budget"] + 1e-8:
                    raise ValueError("Exceeded request budget")
                if row["asd"]["budget"] != POLICIES[m].budget:
                    raise ValueError("Changed method budget")
                if score(records[q], row["text"]) != row["quality"]:
                    raise ValueError("Changed quality score; retain raw data and audit separately")
                if stage == "natural":
                    anchor = mechanism[q,m]
                    if (row["token_ids"], row["asd"]) != (anchor["token_ids"], anchor["asd"]):
                        # Timed strict avoids the diagnostic B=0 verifier.
                        if m != "DFlash" or row["token_ids"] != anchor["token_ids"]:
                            raise ValueError("Timed and mechanism trajectories differ")
        table = {}
        for m in POLICIES:
            selected = [r for r in stages["mechanism"] if r["method"] == m]
            timed = [r for r in stages["natural"] if r["method"] == m]
            diagnostics = {r["request_id"]: audit_events(r) for r in selected}
            seconds = sum(r["latency_seconds"] for r in timed)
            tokens = sum(len(r["token_ids"]) for r in timed)
            rounds = sum(r["counters"]["rounds"] for r in selected)
            table[m] = dict(requests=len(selected), seconds=seconds, tokens=tokens, tps=tokens/seconds,
                            quality_correct=sum(r["quality"]["correct"] for r in selected),
                            length_stops=sum(r["finish_reason"] == "length" for r in selected),
                            strict_identity=sum(r["strict_identity"] for r in selected), rounds=rounds,
                            rounds_per_100_tokens=rounds / sum(len(r["token_ids"]) for r in selected) * 100,
                            regret=sum(r["asd"]["spent"] for r in selected),
                            exceptions=sum(r["asd"]["relaxed_tokens"] for r in selected),
                            shortened_blocks=sum(r["asd"]["shortened_blocks"] for r in selected),
                            foregone_proposal_tokens=sum(r["asd"]["foregone_proposal_tokens"] for r in selected),
                            local_diagnostics=diagnostics)
        output[benchmark] = table
        for q in records:
            if q not in ids:
                continue
            requests[q] = {m: dict(quality=mechanism[q,m]["quality"], tokens=len(mechanism[q,m]["token_ids"]),
                                    finish_reason=mechanism[q,m]["finish_reason"],
                                    rounds=mechanism[q,m]["counters"]["rounds"], asd=mechanism[q,m]["asd"],
                                    seconds=sum(r["latency_seconds"] for r in stages["natural"] if (r["request_id"],r["method"]) == (q,m)))
                           for m in POLICIES}
    if provenance["gsm8k"]["source"]["sha256"] != provenance["math500"]["source"]["sha256"]:
        raise ValueError("Benchmarks executed different runtime sources")
    comparisons = {}
    for benchmark, table in output.items():
        comparisons[benchmark] = {}
        for m in ("ASD-Run", "ASD-Utility"):
            keys = [q for q in requests if records[q]["benchmark"] == benchmark]
            comparisons[benchmark][m] = dict(
                latency_change_percent=(table[m]["seconds"]/table["ASD"]["seconds"]-1)*100,
                natural_tps_change_percent=(table[m]["tps"]/table["ASD"]["tps"]-1)*100,
                quality_losses=[q for q in keys if requests[q]["ASD"]["quality"]["correct"] and not requests[q][m]["quality"]["correct"]],
                quality_gains=[q for q in keys if not requests[q]["ASD"]["quality"]["correct"] and requests[q][m]["quality"]["correct"]])
    termination = {}
    for benchmark in output:
        groups = {name: [] for name in ("all_eos", "all_length", "mixed")}
        for q,rows in requests.items():
            if records[q]["benchmark"] != benchmark:
                continue
            reasons = {r["finish_reason"] for r in rows.values()}
            group = "all_eos" if reasons == {"eos"} else "all_length" if reasons == {"length"} else "mixed"
            groups[group].append(q)
        termination[benchmark] = {}
        for group, ids in groups.items():
            if not ids:
                continue
            table = {}
            for m in POLICIES:
                seconds = sum(requests[q][m]["seconds"] for q in ids)
                tokens = sum(requests[q][m]["tokens"] for q in ids) * 2
                table[m] = dict(seconds=seconds, tokens=tokens, tps=tokens/seconds)
            termination[benchmark][group] = dict(ids=ids, methods=table)
    return dict(benchmarks=output, comparisons=comparisons, requests=requests,
                termination_diagnostics=termination,
                grader_sha256=file_hash(ROOT / "memorydraft/evaluation/asd_budget.py")), provenance


def render(a, infos):
    lines = ["# ASD 实际前缀收益预算验证", "", "2026-10-20。新GSM8K 16题、MATH-500 8题；96机制+192自然EOS计时，288次生成。", "",
             "同一Qwen3-8B/DFlash、FP32/eager、greedy、非思考、最多512token。近似组B=8/g=0.25/M=2；Utility价格2冻结，不按本批调参。",
             "同题四组同卡交错；GSM8K物理GPU0、MATH-500物理GPU1。所有在线策略成本计入，机制插桩及加载不计正式时延。", "",
             "| 任务 | 方法 | 自然EOS秒（两重复） | token（两重复） | 自然TPS | 正确 | 截断 | 机制轮数 | 每100token轮数 | regret总支出 | 例外token | 主动缩短块 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for benchmark, table in a["benchmarks"].items():
        for m,t in table.items():
            lines.append(f"| {benchmark} | {m} | {t['seconds']:.6f} | {t['tokens']} | {t['tps']:.3f} | {t['quality_correct']}/{t['requests']} | {t['length_stops']} | {t['rounds']} | {t['rounds_per_100_tokens']:.3f} | {t['regret']:.3f} | {t['exceptions']} | {t['shortened_blocks']} |")
    lines += ["", "自然EOS路径长度可能改变；TPS也反映不同生成分布，不能当成相同工作量或逐token等价加速。本轮没有忽略EOS的强制续写。", "",
              "GSM8K按最终数值评分；MATH-500为最后boxed答案的保守规范化精确匹配，不是一般符号等价。失败、截断、零激活均保留，两重复不支持显著性或质量非劣性保证。", "",
              "| 任务 | 相对原ASD | 自然时延变化 | 自然TPS变化 | 正确→不匹配 | 不匹配→正确 |", "|---|---|---:|---:|---|---|"]
    for benchmark, comparisons in a["comparisons"].items():
        for m,c in comparisons.items():
            lines.append(f"| {benchmark} | {m} | {c['latency_change_percent']:+.2f}% | {c['natural_tps_change_percent']:+.2f}% | {', '.join(c['quality_losses']) or '无'} | {', '.join(c['quality_gains']) or '无'} |")
    lines += ["", "按全部四组的终止原因作事后诊断（保留上面的全量主结果，不用子集替代）：", "",
              "| 任务 | 终止分组 | 题数 | 方法/原ASD | 时延变化 | 自然TPS变化 |", "|---|---|---:|---|---:|---:|"]
    for benchmark, groups in a["termination_diagnostics"].items():
        for group, partition in groups.items():
            table = partition["methods"]
            for m in ("ASD-Run", "ASD-Utility"):
                latency = (table[m]["seconds"]/table["ASD"]["seconds"]-1)*100
                tps = (table[m]["tps"]/table["ASD"]["tps"]-1)*100
                lines.append(f"| {benchmark} | {group} | {len(partition['ids'])} | {m} | {latency:+.2f}% | {tps:+.2f}% |")
    lines += ["", "all_eos=四组均完整结束；all_length=四组均达长度上限；mixed=终止原因不一致。分类只作诊断，未作为在线决策或选题依据。"]
    completed = a["termination_diagnostics"]["math500"].get("all_eos")
    if completed:
        table = completed["methods"]
        run = (table["ASD-Run"]["tps"]/table["ASD"]["tps"]-1)*100
        utility = (table["ASD-Utility"]["tps"]/table["ASD"]["tps"]-1)*100
        lines.append(f"MATH完整结束{len(completed['ids'])}题中，Run/Utility的自然TPS变化{run:+.2f}%/{utility:+.2f}%。完整结束与上限截断分组须分别解释，尚不能据此宣称完整长解答的稳定每token加速。")
    lines += ["", "负时延变化表示更快。Utility额外缩短是在它自身轨迹/剩余预算下相对本轮最长可行前缀；不能当作与原ASD整条轨迹逐块配对。", "",
              "| 任务 | 方法 | 付费块 | 局部额外候选token | 增量≤2块 | 这些块regret | 每regret局部增量 |", "|---|---|---:|---:|---:|---:|---:|"]
    for benchmark, table in a["benchmarks"].items():
        for m in ("ASD", "ASD-Run", "ASD-Utility"):
            t=table[m]; d=t["local_diagnostics"].values()
            paid=sum(x.get("paid_blocks",0) for x in d); gain=sum(x.get("extra_proposal_tokens_same_state",0) for x in d)
            low=sum(x.get("paid_blocks_gain_le_2",0) for x in d); cost=sum(x.get("low_gain_regret",0) for x in d)
            lines.append(f"| {benchmark} | {m} | {paid} | {gain} | {low} | {cost:.3f} | {gain/t['regret'] if t['regret'] else 0:.3f} |")
    lines += ["", "预算效率与请求效果：", ""]
    for benchmark, table in a["benchmarks"].items():
        base, utility = table["ASD"], table["ASD-Utility"]
        regret_change = (utility['regret']/base['regret']-1)*100 if base['regret'] else 0.
        lines.append(f"- {benchmark}: Utility regret支出变化{regret_change:+.2f}%，验证轮变化{(utility['rounds']/base['rounds']-1)*100:+.2f}%；正确数{base['quality_correct']}→{utility['quality_correct']}。regret不是错误概率，减少支出不保证质量提升。")
    if "gsm8k:21" in a["comparisons"]["gsm8k"]["ASD-Utility"]["quality_losses"]:
        lines.append("- gsm8k:21人工核对：题目需由当前年龄37减去生育年龄23，正确14；原ASD输出14，Utility输出23，属于真实推理错误，不是解析漏识别。未做反事实重放，不能断言由哪一个例外单独造成。")
    math_rows = [rows for q,rows in a["requests"].items() if q.startswith("math500:")]
    truncated = sum(all(r["finish_reason"] == "length" and not r["quality"]["correct"] for r in rows.values()) for rows in math_rows)
    lines.append(f"- MATH-500有{truncated}题在全部四组中均截断且不匹配；这些题没有完整解答质量结论。")
    lines += ["", "局部增量=同一已评分块的近似接受长度−严格接受长度；不等于全请求新增输出或节约验证轮。预算到7的位置分数保存在analysis.json，不用最终长度作为在线输入。", "",
              "| 请求 | 原ASD正确 | Run正确 | Utility正确 | ASD→Run→Utility token | ASD→Run→Utility轮数 |", "|---|---|---|---|---|---|"]
    for q, rows in a["requests"].items():
        chosen=[rows[m] for m in ("ASD","ASD-Run","ASD-Utility")]
        quality=[str(x["quality"]["correct"]) for x in chosen]
        lines.append(f"| {q} | {' | '.join(quality)} | {'→'.join(str(x['tokens']) for x in chosen)} | {'→'.join(str(x['rounds']) for x in chosen)} |")
    lines += ["", "MATH不匹配的原始解析值（需结合完整输出复查）：", ""]
    for q, rows in a["requests"].items():
        if not q.startswith("math500:"): continue
        for m,r in rows.items():
            if not r["quality"]["correct"]:
                lines.append(f"- {q}/{m}: {json.dumps(r['quality'],ensure_ascii=False,sort_keys=True)}")
    lines += ["", f"执行runtime package SHA `{infos['gsm8k']['source']['sha256']}`。所有既有发布结果在执行前后SHA一致；冻结源码前后校验见execution-verification.json。",
              "", "重建：`python -m scripts.write_asd_budget_report`。", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"results/asd-budget")
    parser.add_argument("--report",type=Path,default=ROOT/"docs/ASD_BUDGET_RESULTS.md")
    args=parser.parse_args()
    a,infos=analyze(args.source)
    (args.source/"analysis.json").write_bytes((json.dumps(a,ensure_ascii=False,indent=2)+"\n").encode())
    args.report.write_bytes(render(a,infos).encode())
    print(json.dumps(a["comparisons"],ensure_ascii=False))


if __name__=="__main__":
    main()
