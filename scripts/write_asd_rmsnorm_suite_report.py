"""Natural-EOS quality and question-paired timing, retaining numerical drift."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from memorydraft.evaluation.asd_suite import BENCHMARKS, score
from memorydraft.evaluation.runtime_migration import read_rows
from scripts.diagnose_transport_online import write
from scripts.run_asd_compile_probe import trajectory_comparison
from scripts.run_asd_execution_probe import require_same, selected_records
from scripts.run_asd_rmsnorm_suite_probe import (LIMIT, METHODS, PROFILES, REPEATS, previous_rows,
                                                records_and_positions, timing_order)
from scripts.write_asd_compile_report import validate_ledger
from scripts.write_asd_suite_report import assessed, paired

ROOT = Path(__file__).resolve().parents[1]


def validate_row(row, records, stops):
    q = row["request_id"]
    if q not in records or row["benchmark"] != records[q]["benchmark"] or row["profile"] not in PROFILES:
        raise ValueError("Changed development identity")
    tokens = row["token_ids"]
    if row["max_new_tokens"] != LIMIT or not 0 < len(tokens) <= LIMIT:
        raise ValueError("Invalid natural length/limit")
    if ((row["finish_reason"] == "eos") != (tokens[-1] in stops)
            or any(t in stops for t in tokens[:-1]) or row["finish_reason"] not in {"eos", "length"}):
        raise ValueError("Invalid natural EOS")
    if row["finish_reason"] == "length" and len(tokens) != LIMIT:
        raise ValueError("Early natural truncation")
    for field in ("engine_latency_seconds", "policy_setup_seconds", "profile_switch_seconds", "latency_seconds"):
        v = row[field]
        if not math.isfinite(v) or v < 0 or (field in {"engine_latency_seconds", "latency_seconds"} and v == 0):
            raise ValueError("Invalid online timing")
    if row["latency_seconds"] != row["engine_latency_seconds"]+row["policy_setup_seconds"]+row["profile_switch_seconds"]:
        raise ValueError("Online cost excluded")
    delta = row["graphs_after"]-row["graphs_before"]
    if delta < 0 or (row["profile"] == "eager" and delta):
        raise ValueError("Invalid compile event accounting")
    if not 0 < row["peak_gpu_bytes"] <= 46*2**30:
        raise ValueError("Allocation budget exceeded")
    if row["method"] == "DFlash":
        if row["asd"]["spent"] or any(r.get("regret", 0) or r.get("relaxed_tokens", 0) for r in row["rounds"]):
            raise ValueError("Strict DFlash spent approximate budget")
    elif row["method"] != "AR":
        validate_ledger(row)
        counts = row["cpu_execution"]
        if set(counts) != {"fast_calls", "fallback_calls"} or any(not isinstance(v, int) or v < 0 for v in counts.values()) or sum(counts.values()) != len(row["rounds"]):
            raise ValueError("Missing CPU verifier calls")


def audit(rows, references, records, previous, stops):
    expected = {(q, m, p, i) for q in records for m in METHODS for p in PROFILES for i in range(REPEATS)}
    timed, refs = {}, {}
    for r in references:
        key = r["request_id"], r["profile"]
        if key in refs or key[0] not in records or r["method"] != "AR" or r["stage"] != "reference" or r["repeat"] != 0:
            raise ValueError("Invalid AR reference matrix")
        validate_row(r, records, stops)
        order = timing_order(key[0], -1, ("AR",))
        if not 0 <= r["slot"] < len(order) or order[r["slot"]] != ("AR", key[1]):
            raise ValueError("Changed AR pair order")
        refs[key] = r
    if set(refs) != {(q, p) for q in records for p in PROFILES}:
        raise ValueError("Incomplete AR reference matrix")
    for r in rows:
        key = r["request_id"], r["method"], r["profile"], r["repeat"]
        if key not in expected or key in timed or r["stage"] != "natural":
            raise ValueError("Invalid natural timed matrix")
        validate_row(r, records, stops)
        order = timing_order(key[0], key[3])
        if not 0 <= r["slot"] < len(order) or order[r["slot"]] != key[1:3]:
            raise ValueError("Changed frozen natural pair order")
        timed[key] = r
    if set(timed) != expected:
        raise ValueError("Incomplete natural timed matrix")
    scored = {}
    first = {}
    for q, record in records.items():
        for m in ("AR", *METHODS):
            for p in PROFILES:
                row = refs[q, p] if m == "AR" else timed[q, m, p, 0]
                first[q, m, p] = row
                baseline = previous[q, m] if previous is not None else (refs[q, "eager"] if m == "AR" else timed[q, m, "eager", 0])
                if previous is not None and row["profile"] == "eager":
                    require_same(row, previous[q, m])
                else:
                    trajectory_comparison(baseline, row)
                # Every method/profile sees the SAME request, not only its old arm.
                trajectory_comparison(dict(row, method="AR"), refs[q, "eager"])
                sk = q, row["text"]
                if sk not in scored:
                    scored[sk] = score(record, row["text"])
                if row["quality"] != scored[sk]:
                    raise ValueError("Changed natural quality scoring")
                if previous is not None and p == "eager" and row["quality"] != previous[q, m]["quality"]:
                    raise ValueError("Changed old natural quality")
                if m != "AR":
                    for i in range(REPEATS):
                        repeated = timed[q, m, p, i]
                        require_same(repeated, row)
                        if repeated["quality"] != row["quality"]:
                            raise ValueError("Changed repeated quality")
    pairs = [trajectory_comparison(first[q, m, "eager"], first[q, m, PROFILES[1]])
             for q in records for m in ("AR", *METHODS)]
    strict = {p: sum(first[q, "DFlash", p]["token_ids"] == first[q, "AR", p]["token_ids"] for q in records) for p in PROFILES}
    return dict(questions=len(records), timed=len(rows), references=len(references), quality_rows=len(first),
                eager_old_trajectory_checks=len(records)*5 if previous is not None else 0, unique_texts_rescored=len(scored),
                compiled_token_matches=sum(x["token_equal"] for x in pairs),
                compiled_ledger_matches=sum(x["ledger_equal"] for x in pairs), strict_matches_AR=strict,
                compile_events_during_formal=sum(r["graphs_after"]-r["graphs_before"] for r in rows),
                compile_events_during_AR=sum(r["graphs_after"]-r["graphs_before"] for r in references))


def stratified(cases, seed):
    grouped = defaultdict(list)
    for c in cases:
        grouped[c["benchmark"]].append(math.log(c["ratio"]))
    if set(grouped) != set(BENCHMARKS) or len({len(v) for v in grouped.values()}) != 1:
        raise ValueError("Expected balanced development strata")
    rng = random.Random(seed)
    samples = sorted(math.exp(statistics.mean(rng.choice(v) for v in grouped.values() for _ in v)) for _ in range(5000))
    return dict(n=len(cases), ratio=math.exp(statistics.mean(math.log(c["ratio"]) for c in cases)),
                ci95=[samples[125], samples[4874]], faster=sum(c["ratio"] < 1 for c in cases))


def summarize(rows, references, records, *, seed="rmsnorm-suite", development_subsets=True):
    timed = {(r["request_id"], r["method"], r["profile"], r["repeat"]): r for r in rows}
    first = {(r["request_id"], r["method"], r["profile"]): r for r in (*references, *[r for r in rows if r["repeat"] == 0])}
    cases = []
    old_ids = {r["id"] for r in selected_records()}
    for q, record in records.items():
        for m in METHODS:
            e, c = first[q, m, "eager"], first[q, m, PROFILES[1]]
            samples = {p: [timed[q, m, p, i]["latency_seconds"] for i in range(REPEATS)] for p in PROFILES}
            ratios = [c/e for e, c in zip(samples["eager"], samples[PROFILES[1]])]
            ratio = math.exp(statistics.mean(map(math.log, ratios)))
            length = len(c["token_ids"])/len(e["token_ids"])
            calls = (len(c["rounds"])+1)/(len(e["rounds"])+1)
            quality_change = int(c["quality"]["correct"])-int(e["quality"]["correct"]) if assessed(c) and assessed(e) else None
            cases.append(dict(request_id=q, benchmark=record["benchmark"], method=m,
                seen_in_small_compile=q in old_ids, ratio=ratio, paired_ratios=ratios, latency_samples=samples,
                comparison=trajectory_comparison(e, c), quality_eager=e["quality"], quality_compiled=c["quality"],
                quality_change=quality_change, eager_tokens=len(e["token_ids"]), compiled_tokens=len(c["token_ids"]),
                eager_rounds=len(e["rounds"]), compiled_rounds=len(c["rounds"]),
                eager_finish_reason=e["finish_reason"], compiled_finish_reason=c["finish_reason"],
                length_ratio=length, call_density_ratio=calls/length, amortized_cost_ratio=ratio/calls,
                max_spent_difference=abs((e["asd"] or {}).get("spent", 0)-(c["asd"] or {}).get("spent", 0))))
    overall, families, quality, subsets = {}, {}, {}, {}
    for m in METHODS:
        selected = [c for c in cases if c["method"] == m]
        overall[m] = stratified(selected, seed+":overall:"+m)
        same = [c["ratio"] for c in selected if c["comparison"]["token_equal"]]
        subsets[m] = {}
        if development_subsets:
            subsets[m].update(old12=stratified([c for c in selected if c["seen_in_small_compile"]], seed+":old12:"+m),
                              new36=stratified([c for c in selected if not c["seen_in_small_compile"]], seed+":new36:"+m))
        subsets[m]["same_tokens_descriptive"] = dict(n=len(same), ratio=math.exp(statistics.mean(map(math.log, same))) if same else None)
    for b in BENCHMARKS:
        families[b], quality[b] = {}, {}
        for m in ("AR", *METHODS):
            quality[b][m] = {}
            for p in PROFILES:
                selected = [r for (q, method, profile), r in first.items() if method == m and profile == p and r["benchmark"] == b]
                quality[b][m][p] = dict(tasks=len(selected), correct=sum(r["quality"]["correct"] for r in selected),
                    secondary_correct=sum(r["quality"].get("secondary_correct", r["quality"]["correct"]) for r in selected),
                    truncated=sum(r["finish_reason"] != "eos" for r in selected),
                    unassessed=[r["request_id"] for r in selected if not assessed(r)],
                    needs_review=[r["request_id"] for r in selected if r["quality"].get("needs_review")],
                    f1_mean=statistics.mean(r["quality"]["f1"] for r in selected) if b == "squad" else None)
            if m != "AR":
                selected = [c for c in cases if c["benchmark"] == b and c["method"] == m]
                families[b][m] = dict(**paired([c["ratio"] for c in selected], f"{seed}:family:{b}:{m}"),
                    token_matches=sum(c["comparison"]["token_equal"] for c in selected),
                    quality_improved=sum(c["quality_change"] == 1 for c in selected),
                    quality_lost=sum(c["quality_change"] == -1 for c in selected),
                    quality_unassessed=sum(c["quality_change"] is None for c in selected))
    ar = [dict(request_id=q, benchmark=r["benchmark"], comparison=trajectory_comparison(first[q, "AR", "eager"], first[q, "AR", PROFILES[1]]),
               quality_eager=first[q, "AR", "eager"]["quality"], quality_compiled=first[q, "AR", PROFILES[1]]["quality"]) for q, r in records.items()]
    note = "All primary natural cases retained. Paired repeats collapse per question; eight per family, 48 development questions. Old12/new36 are both development data. Same-token subset is descriptive. Amortized cost includes all online work, not pure Target kernels." if development_subsets else "All 48 validation questions retained; frozen candidate and metrics, paired repeats collapse per question. Same-token subset is descriptive. Amortized cost includes all online work, not pure Target kernels."
    return dict(overall=overall, families=families, quality=quality, cases=cases, subsets=subsets, AR=ar, note=note)


def main():
    base = ROOT/"results/asd-rmsnorm-suite"
    read = lambda name: [r for i in (0, 1) for r in read_rows(base/f"part-{i}/{name}.jsonl")]
    rows, references = read("rows"), read("references")
    records = {r["id"]: r for r in records_and_positions()[0]}
    info = [json.loads((base/f"part-{i}/provenance.json").read_text()) for i in (0, 1)]
    execution = json.loads((base/"execution-verification.json").read_text())
    if execution["status"] != "complete" or execution["files_verified_before"] != execution["files_verified_after"]:
        raise ValueError("Incomplete source verification")
    if any(p["status"] != "complete" or not p["original_unchanged"] or p["split"] != "development" for p in info):
        raise ValueError("Incomplete development execution")
    if info[0]["stop_token_ids"] != info[1]["stop_token_ids"]:
        raise ValueError("Changed stop tokens")
    result = dict(audit=audit(rows, references, records, previous_rows(), info[0]["stop_token_ids"]),
                  **summarize(rows, references, records),
                  preparation=[dict(shard=p["shard"], controller_setup_seconds=p["controller_setup_seconds"],
                                    warmup_wall_seconds=p["warmup_wall_seconds"], cache_bytes=p["cache_bytes"],
                                    compiler_after_warmup=p["compiler_after_warmup"], compiler=p["compiler"]) for p in info])
    write(base/"analysis.json", result)
    lines = ["# RMSNorm完整开发集自然EOS评估", "", "48开发题/六类各8；两profile同CPU接受/FP32原16块。1152配对计时（首重复同时评分）+96 AR质量对照，另20短warmup。上限4096，不强制EOS后续写。验证/保留不运行。", "",
             "|方法|编译/eager自然时延比|题级分层95% bootstrap|更快题|新增profile评估36题时延比|", "|---|---:|---|---:|---:|"]
    for m, v in result["overall"].items():
        lines.append(f"|{m}|{v['ratio']:.6f}|[{v['ci95'][0]:.6f},{v['ci95'][1]:.6f}]|{v['faster']}/48|{result['subsets'][m]['new36']['ratio']:.6f}|")
    lines += ["", "## 分任务时延与质量变化", "", "|任务|方法|时延比[95%]|token相同|主质量提升/下降/未评估|", "|---|---|---|---:|---|"]
    for b, methods in result["families"].items():
        for m, v in methods.items():
            lines.append(f"|{b}|{m}|{v['ratio']:.6f} [{v['ci95'][0]:.6f},{v['ci95'][1]:.6f}]|{v['token_matches']}/8|{v['quality_improved']}/{v['quality_lost']}/{v['quality_unassessed']}|")
    lines += ["", "## 完整答案质量", "", "MATH主评分为保守规范答案比较，secondary另含有限标量等价；代码为提供测试，unassessed不当作已评估失败；SQuAD另列F1。", "", "|任务|方法|eager/编译主正确|eager/编译secondary|eager/编译截断|eager/编译未评估|eager/编译F1|", "|---|---|---|---|---|---|---|"]
    for b, methods in result["quality"].items():
        for m, item in methods.items():
            a, c = item["eager"], item[PROFILES[1]]
            lines.append(f"|{b}|{m}|{a['correct']}/{c['correct']}（各8）|{a['secondary_correct']}/{c['secondary_correct']}|{a['truncated']}/{c['truncated']}|{len(a['unassessed'])}/{len(c['unassessed'])}|{a['f1_mean']}/{c['f1_mean']}|")
    lines += ["", "## 逐题时延、轨迹与计算量", "", "|题|方法|时延比|token相同|首次分歧|原/编译token|原/编译轮|长度比|调用密度比|摊销成本比|质量变化|", "|---|---|---:|---|---|---|---|---:|---:|---:|---|"]
    for c in result["cases"]:
        v = c["comparison"]
        lines.append(f"|{c['request_id']}|{c['method']}|{c['ratio']:.6f}|{v['token_equal']}|{v['first_token_difference']}|{c['eager_tokens']}/{c['compiled_tokens']}|{c['eager_rounds']}/{c['compiled_rounds']}|{c['length_ratio']:.6f}|{c['call_density_ratio']:.6f}|{c['amortized_cost_ratio']:.6f}|{c['quality_change']}|")
    lines += ["", "## 完整性与边界", "", json.dumps(result["audit"], ensure_ascii=False), "",
              "首重复保留全部线上编译成本；评分在一对两次生成之后、engine时钟外，相同题/文本只评一次，重复复用。所有重复必须重现自身token/轮次/账本。不同profile输出差异不删题；同token描述子集见analysis.json，不冒充独立验证。", "",
              "length×调用密度×摊销成本=时延比。摊销成本含prefill/Draft/CPU全部在线，不是纯Target内核时长。AR只有一次质量对照，不对其单次时延作加速统计。完整开发集不等于独立验证或质量非劣证明。", "",
              "编译outer包含backend，不能相加；warmup wall含推理，非纯编译时间。原始图事件/缓存/显存/数值轨迹见provenance与analysis.json。", "",
              "重建：`.venv\\Scripts\\python.exe -m scripts.write_asd_rmsnorm_suite_report`；协议ASD_RMSNORM_SUITE_PROTOCOL.md，解释ASD_RMSNORM_SUITE_FINDINGS.md。", ""]
    (ROOT/"docs/ASD_RMSNORM_SUITE_RESULTS.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(dict(audit=result["audit"], overall=result["overall"])))


if __name__ == "__main__":
    main()
