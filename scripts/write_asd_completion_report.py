"""Audit extended natural-EOS decoding of the three previously truncated MATH tasks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from memorydraft.evaluation.asd_budget import load_tasks, score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from scripts.run_asd_budget_probe import POLICIES, compare_previous
from scripts.write_asd_budget_report import audit_events

ROOT = Path(__file__).resolve().parents[1]
PARTS = {"part-a": ("math500:1", "math500:4"), "part-b": ("math500:7",)}


def analyze(source):
    data = ROOT / "data/research/asd-budget/records.json"
    records = {r["id"]:r for r in load_tasks(data)}
    old_root = ROOT / "results/asd-budget/math500"
    old_rows = read_rows(old_root / "mechanism/rows.jsonl")
    old = {(r["request_id"],r["method"]):r for r in old_rows}
    previous_hash = file_hash(old_root / "mechanism/rows.jsonl.gz")
    old_info = json.loads((old_root / "provenance.json").read_text(encoding="utf-8"))
    mechanism, natural, infos = {}, [], {}
    for part, ids in PARTS.items():
        directory = source / part
        info = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        if (info["status"] != "complete" or not info["original_unchanged"] or info["task_ids"] != list(ids)
                or info["max_new_tokens"] != 2048 or info["data_sha256"] != file_hash(data)
                or info["source"]["sha256"] != old_info["source"]["sha256"]
                or info["protocol_sha256"] != file_hash(ROOT / "docs/ASD_COMPLETION_PROTOCOL.md")
                or info["previous_rows_sha256"] != previous_hash
                or info["policies"] != {m:asdict(c) for m,c in POLICIES.items()}):
            raise ValueError("Changed or incomplete completion probe")
        infos[part] = info
        refs = json.loads((directory / "references.json").read_text(encoding="utf-8"))
        for stage, repeats in (("mechanism", 1), ("natural", 2)):
            rows = read_rows(directory / stage / "rows.jsonl")
            keys = {(r["request_id"],r["method"],r["repeat"]) for r in rows}
            expected = {(q,m,i) for q in ids for m in POLICIES for i in range(repeats)}
            if len(rows) != len(expected) or keys != expected:
                raise ValueError("Missing or duplicate request/method/repeat")
            for row in rows:
                q, m = row["request_id"], row["method"]
                ref = refs[q]
                if (row["prompt_sha256"],row["snapshot_id"]) != (ref["prompt_sha256"],ref["snapshot_id"]):
                    raise ValueError("Changed Target input")
                identity = row["token_ids"] == ref["token_ids"]
                if row["strict_identity"] != identity or m == "DFlash" and not identity:
                    raise ValueError("Incorrect strict identity")
                if row["asd"]["budget"] != POLICIES[m].budget or not 0 <= row["asd"]["spent"] <= POLICIES[m].budget+1e-8:
                    raise ValueError("Changed or exceeded request budget")
                if score(records[q],row["text"]) != row["quality"]:
                    raise ValueError("Changed quality score")
                comparison = compare_previous(SimpleNamespace(**row),old[q,m])
                if comparison != row["previous_prefix"]:
                    raise ValueError("Incorrect previous prefix audit")
                if stage == "mechanism":
                    audit_events(row)
                    mechanism[q,m] = row
                else:
                    anchor = mechanism[q,m]
                    if row["token_ids"] != anchor["token_ids"] or m != "DFlash" and row["asd"] != anchor["asd"]:
                        raise ValueError("Timed and instrumented trajectories differ")
                    natural.append(row)
    table = {}
    for m in POLICIES:
        selected = [r for (q,method),r in mechanism.items() if method == m]
        timed = [r for r in natural if r["method"] == m]
        seconds = sum(r["latency_seconds"] for r in timed)
        tokens = sum(len(r["token_ids"]) for r in timed)
        table[m] = dict(seconds=seconds,tokens=tokens,tps=tokens/seconds,
                        correct=sum(r["quality"]["correct"] for r in selected),
                        truncated=sum(r["finish_reason"]=="length" for r in selected),
                        rounds=sum(r["counters"]["rounds"] for r in selected),
                        regret=sum(r["asd"]["spent"] for r in selected),
                        exceptions=sum(r["asd"]["relaxed_tokens"] for r in selected),
                        shortened=sum(r["asd"]["shortened_blocks"] for r in selected))
        table[m]["rounds_per_100_tokens"] = table[m]["rounds"]/(tokens/2)*100
    comparisons = {m:dict(latency_change_percent=(table[m]["seconds"]/table["ASD"]["seconds"]-1)*100,
                         tps_change_percent=(table[m]["tps"]/table["ASD"]["tps"]-1)*100)
                   for m in ("ASD-Run","ASD-Utility")}
    per_request = {}
    for ids in PARTS.values():
        for q in ids:
            per_request[q] = {}
            for m in POLICIES:
                r=mechanism[q,m]
                per_request[q][m] = dict(tokens=len(r["token_ids"]),quality=r["quality"],finish_reason=r["finish_reason"],
                                        rounds=r["counters"]["rounds"],asd=r["asd"],previous_prefix=r["previous_prefix"],
                                        seconds=sum(t["latency_seconds"] for t in natural if (t["request_id"],t["method"])==(q,m)))
    return dict(methods=table,comparisons=comparisons,requests=per_request,
                previous_rows_sha256=previous_hash,grader_sha256=file_hash(ROOT/"memorydraft/evaluation/asd_budget.py")),infos


def render(a,infos):
    lines=["# ASD 截断MATH题的完整解答复查", "", "2026-10-20。已观察math500:1/4/7三题，统一自然EOS/2048上限；12机制+24计时，共36生成。", "",
           "同一Qwen3-8B/DFlash、FP32/eager、greedy、非思考；B8/g0.25/M2、Utility价格2均保持。重新从原始prompt生成，预算不在中途重置。",
           "前两题物理GPU0，第三题物理GPU1；同题四组同卡交错。全部在线成本计入，加载与机制插桩另计。", "",
           "| 方法 | 总秒（两重复） | token（两重复） | 自然TPS | 正确 | 截断 | 机制轮 | 每100token轮 | regret | 例外 | 主动缩短块 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m,t in a["methods"].items():
        lines.append(f"| {m} | {t['seconds']:.6f} | {t['tokens']} | {t['tps']:.3f} | {t['correct']}/3 | {t['truncated']} | {t['rounds']} | {t['rounds_per_100_tokens']:.3f} | {t['regret']:.3f} | {t['exceptions']} | {t['shortened']} |")
    lines += ["", "| 相对原ASD | 时延变化 | 自然TPS变化 |", "|---|---:|---:|"]
    for m,c in a["comparisons"].items():
        lines.append(f"| {m} | {c['latency_change_percent']:+.2f}% | {c['tps_change_percent']:+.2f}% |")
    lines += ["", "本批解释：", ""]
    for m in ("ASD-Run","ASD-Utility"):
        t=a["methods"][m]
        change=(t["seconds"]/a["methods"]["DFlash"]["seconds"]-1)*100
        lines.append(f"- {m}相对严格DFlash的自然时延变化{change:+.2f}%；完整输出仍有长度变化，不能把全部时延收益归于每token执行效率。")
    for m in ("ASD-Run","ASD-Utility"):
        slower=[q for q,rows in a["requests"].items() if rows[m]["seconds"]>rows["ASD"]["seconds"]]
        if slower:
            lines.append(f"- {m}存在逐题时延退化：{', '.join(slower)}。总体正结果不代表每题均改善。")
    lines += ["- 本轮排除了这三题的截断限制，完整解答下仍有总体收益；只有三个已观察请求，不能作为新题泛化或质量非劣性结论。",
              "- 上一轮Utility在gsm8k:21的真实错误仍然成立，本轮不覆盖或修正那条负结果。当前优先保留实际后缀价值作为候选，不将Utility升级为默认策略。"]
    lines += ["", "自然TPS和时延都受输出路径/长度影响，不等于相同工作量加速；本轮没有强制越过EOS的续写。质量沿用保守boxed规范化精确匹配，三题补全诊断不是泛化或非劣性证明。", "",
              "| 请求 | 方法 | 秒（两重复） | 相对同题ASD时延 | token | 正确 | 结束 | 机制轮 | 旧512公共前缀 | 必须一致的前缀 |", "|---|---|---:|---:|---:|---|---|---:|---:|---:|"]
    for q,rows in a["requests"].items():
        for m,r in rows.items():
            p=r["previous_prefix"]
            change = (r['seconds']/rows['ASD']['seconds']-1)*100
            lines.append(f"| {q} | {m} | {r['seconds']:.6f} | {change:+.2f}% | {r['tokens']} | {r['quality']['correct']} | {r['finish_reason']} | {r['rounds']} | {p['common_prefix_tokens']} | {p['required_prefix_tokens']} |")
    exact=sum(r["previous_prefix"]["exact_previous_prefix"] for rows in a["requests"].values() for r in rows.values())
    lines += ["", f"全部12条机制满足旧上限裁短验证块之前的前缀一致；{exact}/12保留完整旧512前缀。输入、合法快照、runtime package、数据和策略身份均核对通过。",
              "旧上限可能改变最后块长度及局部门槛，边界附近的差异单列，不把预算内未评分的后缀直接拼接提交。", "",
              "旧512实验中的这3题均为截断；新旧轮次在后续轨迹分叉后不能逐块配对。旧结果保留在results/asd-budget；比较按同题同卡配对，不与旧批次绝对时延相除。", "",
              "未通过评分的输出：", ""]
    for q,rows in a["requests"].items():
        for m,r in rows.items():
            if not r["quality"]["correct"]:
                lines.append(f"- {q}/{m}: {json.dumps(r['quality'],ensure_ascii=False,sort_keys=True)}")
    if all(t["correct"]==3 for t in a["methods"].values()):
        lines.append("- 本轮全部方法3/3通过保守答案评分。")
    lines += ["", f"执行runtime package SHA `{infos['part-a']['source']['sha256']}`；冻结源码与旧发布证据执行前后SHA校验见provenance和execution-verification.json。",
              "", "重建：`python -m scripts.write_asd_completion_report`。", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"results/asd-completion")
    parser.add_argument("--report",type=Path,default=ROOT/"docs/ASD_COMPLETION_RESULTS.md")
    args=parser.parse_args()
    a,infos=analyze(args.source)
    (args.source/"analysis.json").write_bytes((json.dumps(a,ensure_ascii=False,indent=2)+"\n").encode())
    args.report.write_bytes(render(a,infos).encode())
    print(json.dumps(a["comparisons"],ensure_ascii=False))


if __name__=="__main__":
    main()
