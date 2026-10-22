"""Frozen request-balanced finite-action oracle and leave-request-out feasibility."""
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from memorydraft.backends.allocation import ACTIONS, fit_stump, predict
from memorydraft.evaluation.runtime_migration import read_rows
from scripts.diagnose_transport_online import write
from scripts.run_asd_allocation_probe import REPEATS, REQUEST_ACTIONS

ROOT = Path(__file__).resolve().parents[1]


def audit_resources(infos, rows):
    for info in infos:
        compiler = info["compiler"]
        if (compiler["graph_count"] > 8 or compiler["backend_seconds"] > 300 or info["cache_bytes"] > 4*2**30
                or compiler["float32_matmul_precision"] != "highest" or compiler["allow_tf32"]
                or any(e["status"] != "complete" for e in compiler["events"])):
            raise ValueError("Allocation resource or precision budget failed")
    peak = max(r["peak_gpu_bytes"] for r in rows)
    if not 0 < peak <= 46*2**30:
        raise ValueError("Allocation memory budget failed")
    return dict(peak_gpu_bytes=peak, formal_compile_events=sum(
        r["counters"]["compile_graphs_after"]-r["counters"]["compile_graphs_before"] for r in rows))


def samples_from(rows, stage):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = row["request_id"], row.get("source_round", -1)
        grouped[key][row["action"]].append(row)
    actions = ACTIONS if stage == "branch" else REQUEST_ACTIONS
    samples = []
    for (q, index), arms in sorted(grouped.items()):
        if set(arms) != set(actions):
            raise ValueError("Incomplete allocation action matrix")
        for arm, entries in arms.items():
            if sorted(r["repeat"] for r in entries) != list(range(REPEATS)):
                raise ValueError("Missing allocation repeats")
            if any(r["token_ids"] != entries[0]["token_ids"] or r["rounds"] != entries[0]["rounds"] for r in entries):
                raise ValueError("Unstable allocation trajectory")
        first = arms[actions[0]][0]
        safe = all(r["token_ids"] == first["token_ids"] and r["prompt_sha256"] == first["prompt_sha256"]
                   and r["snapshot_id"] == first["snapshot_id"] for entries in arms.values() for r in entries)
        if stage == "request":
            safe &= all(r["token_match"] for entries in arms.values() for r in entries)
        field = "local_seconds" if stage == "branch" else "latency_seconds"
        times = {a: [r[field] for r in sorted(arms[a], key=lambda r: r["repeat"])] for a in actions}
        if any(t is None or not math.isfinite(t) or t <= 0 for ts in times.values() for t in ts):
            raise ValueError("Missing real action timing")
        base = statistics.mean(times[actions[0]])
        samples.append(dict(request_id=q, benchmark=first["benchmark"], source_round=index, safe=safe,
                            features=first.get("features", dict(generated=0, prompt_tokens=first["prompt_tokens"], recent_advance=1., last_advance=1.)),
                            costs={a: statistics.mean(ts)/base for a, ts in times.items()}, times=times))
    return samples


def space(samples, actions, models=None):
    if not samples:
        return None
    counts = {q: sum(s["request_id"] == q for s in samples) for q in {s["request_id"] for s in samples}}
    total = lambda fn: sum(fn(s)/counts[s["request_id"]] for s in samples)
    fixed = min(actions, key=lambda a: total(lambda s: s["costs"][a]))
    best = total(lambda s: s["costs"][fixed])
    oracle = total(lambda s: min(s["costs"].values()))/best
    held_oracle, held_fixed = 0., 0.
    for held in range(REPEATS):
        train = lambda s, a: statistics.mean(t for i, t in enumerate(s["times"][a]) if i != held)
        constant = min(actions, key=lambda a: total(lambda s: train(s, a)/statistics.mean(s["times"][actions[0]])))
        held_oracle += total(lambda s: s["times"][min(actions, key=lambda a: train(s, a))][held]/statistics.mean(s["times"][actions[0]]))
        held_fixed += total(lambda s: s["times"][constant][held]/statistics.mean(s["times"][actions[0]]))
    chosen = []
    if models is None:
        models = {q: fit_stump([s for s in samples if s["request_id"] != q], actions) for q in counts}
    for s in samples:
        chosen.append(dict(request_id=s["request_id"], source_round=s["source_round"], action=predict(models[s["request_id"]], s["features"])))
    predicted = total(lambda s: s["costs"][predict(models[s["request_id"]], s["features"])])/best
    return dict(requests=len(counts), states=len(samples), safe=all(s["safe"] for s in samples), best_fixed=fixed,
                fixed_costs={a: total(lambda s: s["costs"][a])/len(counts) for a in actions},
                optimistic_oracle_ratio=oracle, cross_repeat_oracle_ratio=held_oracle/held_fixed,
                leave_request_out_ratio=predicted, models=models, choices=chosen,
                final_model=fit_stump(samples, actions))


def analyze(requests, branches):
    result = {}
    for stage, rows, actions in (("request", requests, REQUEST_ACTIONS), ("branch", branches, ACTIONS)):
        samples = samples_from(rows, stage)
        overall = space(samples, actions)
        families = {b: space([s for s in samples if s["benchmark"] == b], actions, overall["models"]) for b in sorted({s["benchmark"] for s in samples})}
        scopes = {"overall": overall, **families}
        eligible = [name for name, v in scopes.items() if v is not None and v["safe"] and v["requests"] >= 2
                    and v["optimistic_oracle_ratio"] <= .95 and v["cross_repeat_oracle_ratio"] <= .97]
        # Family gates can identify a useful setting. Predictors are evaluated
        # across all requests so two samples never train their own family-only model.
        prediction = overall is not None and overall["safe"] and any(scopes[name]["leave_request_out_ratio"] <= .98 for name in eligible)
        result[stage] = dict(overall=overall, families=families, samples=samples,
                             oracle_eligible_scopes=eligible, prediction_gate=prediction)
    result["continue_selector"] = any(v["oracle_eligible_scopes"] and v["prediction_gate"] for v in result.values())
    return result


def main():
    base = ROOT/"results/asd-allocation"
    execution = json.loads((base/"execution-verification.json").read_text())
    if execution["status"] != "complete" or execution["files_verified_before"] != execution["files_verified_after"]:
        raise ValueError("Incomplete frozen-source execution")
    infos = [json.loads((base/f"part-{i}/provenance.json").read_text()) for i in (0, 1)]
    if any(p["status"] != "complete" or not p["original_unchanged"] or p["split"] != "development" for p in infos):
        raise ValueError("Incomplete or changed allocation probe")
    read = lambda name: [r for i in (0, 1) for r in read_rows(base/f"part-{i}/{name}.jsonl")]
    requests, branches, sources = read("requests"), read("branches"), read("sources")
    from scripts.run_asd_execution_probe import selected_records
    if len(sources) != 12 or len(requests) != 144 or {r["request_id"] for r in sources} != {r["id"] for r in selected_records()}:
        raise ValueError("Changed frozen development coverage")
    result = analyze(requests, branches)
    result["execution_audit"] = audit_resources(infos, [*requests, *branches, *sources])
    write(base/"analysis.json", result)
    lines = ["# 阶段3：有限真实计算动作检查", "", "开发题六类各2，原FP32/RMSNorm/16块。请求128上限、局部64新token均遵守EOS；各动作相同范围token一致时才可比较成本，不把截断前缀当完整自然答案。oracle是乐观有限动作参照，不是线上方法或全局上界。", "",
             "|范围|最优固定动作|事后oracle/固定|跨重复oracle/固定|留题预测/固定|安全|", "|---|---|---:|---:|---:|---|"]
    for stage in ("request", "branch"):
        for name, item in {"overall": result[stage]["overall"], **result[stage]["families"]}.items():
            if item:
                lines.append(f"|{stage}/{name}|{item['best_fixed']}|{item['optimistic_oracle_ratio']:.6f}|{item['cross_repeat_oracle_ratio']:.6f}|{item['leave_request_out_ratio']:.6f}|{item['safe']}|")
    lines += ["", "是否进入实际选择器评估："+str(result["continue_selector"]), "",
              "门槛及完整四动作代价、逐状态和留题模型见协议及analysis.json。家族小样本仅用于发现适用设置，不能冒充独立泛化。所有预填充/共同重放排除在局部clock之外，但当前起草/验证/KV提交/后续与同步包含；请求clock则含全部线上生成成本。", ""]
    (ROOT/"docs/ASD_ALLOCATION_RESULTS.md").write_bytes("\n".join(lines).encode())
    print(json.dumps({s: {k: v for k, v in result[s]["overall"].items() if k not in {"choices", "models"}} for s in ("request", "branch")}))
    print("continue_selector", result["continue_selector"])


if __name__ == "__main__":
    main()
