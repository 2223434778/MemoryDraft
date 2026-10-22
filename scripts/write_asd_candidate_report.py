"""Frozen paired natural-latency and mechanism report for stage four candidates."""
from collections import defaultdict
import argparse
import json
import math
from pathlib import Path
import statistics

from memorydraft.evaluation.runtime_migration import read_rows
from scripts.diagnose_transport_online import write
from scripts.write_asd_allocation_report import audit_resources
from scripts.run_asd_candidate_probe import METHODS, REPEATS

ROOT = Path(__file__).resolve().parents[1]


def analyze(rows, methods=METHODS):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["request_id"]][row["candidate"]].append(row)
    if not grouped or "native" not in methods or len(set(methods)) != len(methods):
        raise ValueError("Missing candidate comparison")
    samples = []
    for request_id, arms in sorted(grouped.items()):
        if set(arms) != set(methods):
            raise ValueError("Incomplete candidate method matrix")
        base = arms["native"][0]
        details = {}
        for method in methods:
            entries = sorted(arms[method], key=lambda r: r["repeat"])
            if [r["repeat"] for r in entries] != list(range(REPEATS)):
                raise ValueError("Missing or duplicate candidate repeats")
            times = [r["latency_seconds"] for r in entries]
            if any(not math.isfinite(t) or t <= 0 for t in times):
                raise ValueError("Invalid candidate latency")
            safe = all(r["token_match"] and r["text_match"] and r["finish_match"]
                       and r["token_ids"] == base["token_ids"] and r["prompt_sha256"] == base["prompt_sha256"]
                       and r["snapshot_id"] == base["snapshot_id"] for r in entries)
            stable = all(r["rounds"] == entries[0]["rounds"] and r["token_ids"] == entries[0]["token_ids"] for r in entries)
            rounds = entries[0]["rounds"]
            repair = [r for r in rounds if r.get("lookup") == "suffix-repair"]
            details[method] = dict(seconds=statistics.mean(times), times=times, safe=safe, stable=stable,
                tokens=len(entries[0]["token_ids"]), target_calls=len(rounds),
                draft_calls=sum(r["draft_forward"] for r in rounds),
                verify_nodes=sum(r.get("tree_nodes", r["proposed"]+1) for r in rounds),
                tree_gain=sum(r.get("tree_gain", 0) for r in rounds),
                branch_rounds=sum(any(i != j for i, j in enumerate(r.get("tree_path", []))) for r in rounds),
                repair_calls=len(repair), repair_accepted=sum(r["accepted"] for r in repair),
                repair_zero=sum(r["accepted"] == 0 for r in repair), finish_reason=entries[0]["finish_reason"],
                quality=entries[0]["quality"])
        native = details["native"]
        for method, item in details.items():
            item["latency_ratio"] = item["seconds"]/native["seconds"]
            item["repeat_ratios"] = [a/b for a, b in zip(item["times"], native["times"])]
        samples.append(dict(request_id=request_id, benchmark=base["benchmark"], methods=details))
    scopes = {"overall": samples, **{b: [s for s in samples if s["benchmark"] == b] for b in sorted({s["benchmark"] for s in samples})}}
    result = {}
    passed = defaultdict(list)
    globally_safe = {method: all(s["methods"][method]["safe"] and s["methods"][method]["stable"]
                                 and s["methods"]["native"]["safe"] and s["methods"]["native"]["stable"]
                                 for s in samples) for method in methods}
    for scope, items in scopes.items():
        summary = {}
        for method in methods:
            ms = [s["methods"][method] for s in items]
            ratio = statistics.mean(s["latency_ratio"] for s in ms)
            repeats = [statistics.mean(s["repeat_ratios"][i] for s in ms) for i in range(REPEATS)]
            safe = all(s["safe"] and s["stable"] for s in ms) and all(s["methods"]["native"]["safe"] and s["methods"]["native"]["stable"] for s in items)
            eligible = (method not in {"native", "packed-linear"} and globally_safe[method] and len(items) >= 2
                        and ratio <= (.97 if scope == "overall" else .95) and all(v < 1 for v in repeats)
                        and (scope == "overall" or all(s["latency_ratio"] < 1 for s in ms)))
            if eligible:
                passed[method].append(scope)
            summary[method] = dict(latency_ratio=ratio, repeat_ratios=repeats, safe=safe,
                faster=sum(s["latency_ratio"] < 1 for s in ms), requests=len(items), eligible=eligible,
                target_calls=sum(s["target_calls"] for s in ms), draft_calls=sum(s["draft_calls"] for s in ms),
                verify_nodes=sum(s["verify_nodes"] for s in ms), tree_gain=sum(s["tree_gain"] for s in ms),
                branch_rounds=sum(s["branch_rounds"] for s in ms), repair_calls=sum(s["repair_calls"] for s in ms),
                repair_accepted=sum(s["repair_accepted"] for s in ms), repair_zero=sum(s["repair_zero"] for s in ms))
        result[scope] = summary
    return dict(scopes=result, samples=samples, globally_safe=globally_safe, expand_candidates=dict(passed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("pilot", "expanded"), default="pilot")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    base = ROOT/f"results/asd-candidate-{args.scope}"
    execution = json.loads((base/"execution-verification.json").read_text())
    if execution["status"] != "complete" or execution["files_verified_before"] != execution["files_verified_after"]:
        raise ValueError("Incomplete frozen-source candidate execution")
    infos = [json.loads((base/f"part-{i}/provenance.json").read_text()) for i in (0, 1)]
    if any(p["status"] != "complete" or not p["original_unchanged"] or p["scope"] != args.scope
           or p["split"] != "development" or p["methods"] != args.methods for p in infos):
        raise ValueError("Incomplete or changed candidate probe")
    rows = [r for i in (0, 1) for r in read_rows(base/f"part-{i}/rows.jsonl")]
    from scripts.run_asd_execution_probe import selected_records
    from memorydraft.evaluation.asd_suite import load_suite
    records = selected_records() if args.scope == "pilot" else load_suite(ROOT/"data/research/asd-suite", "development")
    if len(rows) != len(records)*len(args.methods)*REPEATS or {r["request_id"] for r in rows} != {r["id"] for r in records}:
        raise ValueError("Changed frozen candidate coverage")
    result = analyze(rows, tuple(args.methods))
    result["execution_audit"] = audit_resources(infos, rows)
    write(base/"analysis.json", result)
    lines = ["# 阶段4：多候选树和局部后缀修复", "", "FP32/RMSNorm/原生b16，严格greedy、原Target输入，4096上限并遵守自然EOS。题级三次均值相对同批native归一后等权；重复只检查计时方向，开发数据不冒充独立验证。", "",
             "|范围|方法|时延/native|更快题数|严格且重复稳定|Target调用|Draft调用|验证节点|", "|---|---|---:|---:|---|---:|---:|---:|"]
    for scope, values in result["scopes"].items():
        for method, s in values.items():
            lines.append(f"|{scope}|{method}|{s['latency_ratio']:.6f}|{s['faster']}/{s['requests']}|{s['safe']}|{s['target_calls']}|{s['draft_calls']}|{s['verify_nodes']}|")
    lines += ["", "准入扩测的候选及场景：`"+json.dumps(result["expand_candidates"], ensure_ascii=False)+"`", "",
              "全部节点构造/CPU与设备同步/自定义attention mask/分支选择/KV整理/特征提交均计入请求wall time。packed-linear隔离打包路径成本；tree32/48保留native链并加固定前四位置的Top-2单替换分支，每分支含4/8 token；repair4/8只重试一次真实恢复anchor之后的旧后缀。分支局部推进增加不能代替自然请求净收益。未复现完整BASTION或训练ReTrace，不据本轮否定这些方法。", ""]
    (ROOT/f"docs/ASD_CANDIDATE_{args.scope.upper()}_RESULTS.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(dict(overall=result["scopes"]["overall"], expand_candidates=result["expand_candidates"])))


if __name__ == "__main__":
    main()
