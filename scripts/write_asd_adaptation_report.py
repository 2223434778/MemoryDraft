"""Frozen paired cost/quality analysis for offline draft adaptation controls."""
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from memorydraft.evaluation.runtime_migration import read_rows
from scripts.diagnose_transport_online import write
from scripts.run_asd_adaptation_eval import POLICIES_USED, VARIANTS, records
from scripts.write_asd_allocation_report import audit_resources

ROOT = Path(__file__).resolve().parents[1]


def analyze(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r["request_id"]][r["stage"], r["variant"], r["policy"]].append(r)
    if not groups:
        raise ValueError("No adaptation evaluation")
    expected = {(s, v, p) for s in ("natural", "prefix") for v in VARIANTS for p in POLICIES_USED}
    samples = []
    for request_id, arms in sorted(groups.items()):
        if set(arms) != expected:
            raise ValueError("Incomplete adaptation matrix")
        costs = {}
        for (stage, variant, policy), entries in arms.items():
            entries.sort(key=lambda r: r["repeat"])
            if [r["repeat"] for r in entries] != list(range(3 if stage == "natural" else 2)):
                raise ValueError("Missing or duplicated adaptation repeats")
            times = [r["latency_seconds"] for r in entries]
            if any(not math.isfinite(t) or t <= 0 for t in times):
                raise ValueError("Invalid adaptation timing")
            first = entries[0]
            stable = all((r["token_ids"], r["rounds"], r["asd"]) == (first["token_ids"], first["rounds"], first["asd"]) for r in entries)
            safe = stable and all(r["same_input"] for r in entries)
            if policy == "DFlash":
                safe &= all(r["strict_token_match"] for r in entries)
            else:
                safe &= all(0 <= r["asd"]["spent"] <= 8+1e-9 and r["asd"]["budget"] == 8 for r in entries)
            if variant == "base" and stage == "natural":
                safe &= all(r["base_reference_match"] for r in entries)
            costs[stage, variant, policy] = dict(seconds=statistics.mean(times), times=times, tokens=len(first["token_ids"]),
                rounds=len(first["rounds"]), safe=bool(safe), finish_reason=first["finish_reason"], quality=first.get("quality"),
                spent=first["asd"]["spent"] if first["asd"] else 0.)
        sample = dict(request_id=request_id, benchmark=next(iter(arms.values()))[0]["benchmark"], arms={})
        for stage in ("natural", "prefix"):
            for policy in POLICIES_USED:
                base = costs[stage, "base", policy]
                ordinary = costs[stage, "ordinary", policy]
                common_128 = all(costs[stage, v, policy]["tokens"] == 128 for v in VARIANTS)
                for variant in VARIANTS:
                    item = costs[stage, variant, policy]
                    item.update(latency_ratio=item["seconds"]/base["seconds"],
                                tps_ratio=(item["tokens"]/item["seconds"])/(base["tokens"]/base["seconds"]),
                                ordinary_ratio=item["seconds"]/ordinary["seconds"],
                                paired_safe=item["safe"] and base["safe"] and ordinary["safe"],
                                repeat_ratios=[x/y for x, y in zip(item["times"], base["times"])],
                                both_reach_128=common_128)
                    if stage == "natural":
                        item.update(lost_correct=bool(base["quality"]["correct"] and not item["quality"]["correct"]),
                                    gained_correct=bool(not base["quality"]["correct"] and item["quality"]["correct"]),
                                    new_truncation=item["finish_reason"] != "eos" and base["finish_reason"] == "eos")
                    sample["arms"][f"{stage}/{variant}/{policy}"] = item
        samples.append(sample)
    scopes = {"overall": samples, **{b:[s for s in samples if s["benchmark"] == b] for b in sorted({s["benchmark"] for s in samples})}}
    summaries = {}
    for name, selected in scopes.items():
        summary = {}
        for stage in ("natural", "prefix"):
            for variant in VARIANTS:
                for policy in POLICIES_USED:
                    key = f"{stage}/{variant}/{policy}"
                    values = [s["arms"][key] for s in selected]
                    complete = [v for v in values if v["both_reach_128"]]
                    item = dict(requests=len(values), safe=all(v["paired_safe"] for v in values),
                                latency_ratio=statistics.mean(v["latency_ratio"] for v in values),
                                tps_ratio=statistics.mean(v["tps_ratio"] for v in values),
                                ordinary_ratio=statistics.mean(v["ordinary_ratio"] for v in values),
                                faster=sum(v["latency_ratio"] < 1 for v in values),
                                output_tokens=sum(v["tokens"] for v in values), target_calls=sum(v["rounds"] for v in values),
                                repeat_ratios=[statistics.mean(v["repeat_ratios"][i] for v in values) for i in range(len(values[0]["times"]))])
                    if stage == "natural":
                        item.update(correct=sum(v["quality"]["correct"] for v in values),
                                    lost_correct=sum(v["lost_correct"] for v in values), gained_correct=sum(v["gained_correct"] for v in values),
                                    new_truncations=sum(v["new_truncation"] for v in values),
                                    truncations=sum(v["finish_reason"] != "eos" for v in values))
                    else:
                        item.update(common_full_128_requests=len(complete),
                                    common_full_128_ratio=statistics.mean(v["latency_ratio"] for v in complete) if complete else None,
                                    common_full_128_vs_ordinary=statistics.mean(v["ordinary_ratio"] for v in complete) if complete else None)
                    summary[key] = item
        summaries[name] = summary
    return dict(scopes=summaries, samples=samples,
                all_safe=all(v["safe"] for s in samples for v in s["arms"].values()))


def main():
    base = ROOT/"results/asd-adaptation"
    execution = json.loads((base/"execution-verification.json").read_text(encoding="utf-8"))
    if execution["status"] != "complete" or execution["files_verified_before"] != execution["files_verified_after"]:
        raise ValueError("Incomplete adaptation execution")
    infos = [json.loads((base/f"eval-{i}/provenance.json").read_text(encoding="utf-8")) for i in (0, 1)]
    trainers = {a:json.loads((base/f"train-{a}/provenance.json").read_text(encoding="utf-8")) for a in ("ordinary", "asd-prefix")}
    collectors = [json.loads((base/f"collect-{i}/provenance.json").read_text(encoding="utf-8")) for i in (0, 1)]
    if any(p["status"] != "complete" or not p["original_unchanged"] for p in [*infos, *trainers.values(), *collectors]):
        raise ValueError("Incomplete adaptation phase or old evidence changed")
    rows = [r for i in (0, 1) for r in read_rows(base/f"eval-{i}/rows.jsonl")]
    if len(rows) != 540 or {r["request_id"] for r in rows} != {r["id"] for r in records()}:
        raise ValueError("Changed adaptation evaluation coverage")
    if (trainers["ordinary"]["train_order_keys"] != trainers["asd-prefix"]["train_order_keys"]
            or any(p["updates"] != 128 or p["accumulation"] != 2 for p in trainers.values())
            or any(not p["teacher_parameter_versions_unchanged"] or not p["draft_base_versions_unchanged"]
                   or p["peak_gpu_bytes"] > 46*2**30 or p["seconds"] > 900 for p in trainers.values())):
        raise ValueError("Unmatched training budget or frozen parameter violation")
    result = analyze(rows)
    result["resources"] = audit_resources(infos, rows)
    result["training"] = {a:{k:p[k] for k in ("train_states", "train_requests", "trainable_parameters", "seconds", "selected_step",
                            "initial_calibration", "selected_calibration_loss", "peak_gpu_bytes", "adapter_sha256")} for a, p in trainers.items()}
    result["collection_seconds"] = [p["seconds"] for p in collectors]
    write(base/"analysis.json", result)
    lines = ["# 阶段5：等预算离线Draft适配", "", "普通严格轨迹与ASD部署前缀两组rank8 LoRA，各128更新/累积2。32官方GSM训练题、8内部校准，与评估题隔离；teacher从实际前缀重新生成监督，不把ASD输出直接当标签。评估18开发题含GSM8，其余五类各2；完整自然三重复及128/EOS前缀两重复，共540计时。", "",
             "|范围|阶段/模型/策略|时延/base|TPS/base|时延/普通适配|更快题|安全|正确数/总数或共同128题数|", "|---|---|---:|---:|---:|---:|---|---|"]
    for scope, summary in result["scopes"].items():
        for key, v in summary.items():
            quality = f"{v['correct']}/{v['requests']}" if key.startswith("natural") else str(v["common_full_128_requests"])
            lines.append(f"|{scope}|{key}|{v['latency_ratio']:.6f}|{v['tps_ratio']:.6f}|{v['ordinary_ratio']:.6f}|{v['faster']}/{v['requests']}|{v['safe']}|{quality}|")
    lines += ["", "所有自然质量退化/恢复、截断、真实输出长度及共同达到128的配对结果见analysis.json。ASD路径可以改变，自然时延下降不能直接算计算提速；严格D输出一致时才可直接解释。适配器合并是固定模型准备成本，逐次单列，不混入在线服务切换宣传。", ""]
    (ROOT/"docs/ASD_ADAPTATION_RESULTS.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(dict(all_safe=result["all_safe"], training=result["training"], domain=result["scopes"]["gsm8k"])))


if __name__ == "__main__":
    main()
