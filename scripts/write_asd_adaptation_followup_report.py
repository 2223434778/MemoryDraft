"""Request-level contrasts, matched positions and independent GSM evaluation."""
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from memorydraft.evaluation.adaptation_followup import STUDY, VARIANTS, POLICIES, records
from memorydraft.evaluation.runtime_migration import read_rows
from scripts.diagnose_transport_online import write
from scripts.write_asd_allocation_report import audit_resources

ROOT = Path(__file__).resolve().parents[1]


def bootstrap(values):
    rng = random.Random("MemoryDraft-adaptation-followup-bootstrap-v1")
    means = sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(10000))
    return dict(mean=statistics.mean(values),low=means[249],high=means[9749],requests=len(values),replicates=10000)


def analyze(rows):
    grouped = defaultdict(lambda:defaultdict(list))
    for row in rows:
        grouped[row["request_id"]][row["stage"],row["variant"],row["policy"]].append(row)
    if not grouped:
        raise ValueError("Empty followup evaluation")
    expected = {(s,v,p) for s in ("natural","prefix") for v in VARIANTS for p in POLICIES}
    samples = []
    for request_id, arms in sorted(grouped.items()):
        if set(arms) != expected:
            raise ValueError("Incomplete four-model matrix")
        splits = {r["study_split"] for values in arms.values() for r in values}
        if len(splits) != 1 or not splits <= {"observed","independent"}:
            raise ValueError("Inconsistent exposure split")
        split = next(iter(splits))
        costs = {}
        for (stage,variant,policy),entries in arms.items():
            entries.sort(key=lambda r:r["repeat"])
            if [r["repeat"] for r in entries] != list(range(3 if stage == "natural" else 2)):
                raise ValueError("Invalid repeat coverage")
            times = [r["latency_seconds"] for r in entries]
            if any(not math.isfinite(t) or t <= 0 for t in times):
                raise ValueError("Invalid timing")
            first = entries[0]
            stable = all((r["token_ids"],r["rounds"],r["asd"],r["finish_reason"],r["adapter_sha256"]) ==
                         (first["token_ids"],first["rounds"],first["asd"],first["finish_reason"],first["adapter_sha256"]) for r in entries)
            safe = stable and all(r["same_input"] for r in entries)
            if policy == "DFlash":
                safe &= all(r["strict_token_match"] for r in entries)
            else:
                safe &= all(r["asd"]["budget"] == 8 and 0 <= r["asd"]["spent"] <= 8+1e-9 for r in entries)
            if split == "observed" and stage == "natural" and variant == "base":
                safe &= all(r["base_reference_match"] for r in entries)
            costs[stage,variant,policy] = dict(seconds=statistics.mean(times), times=times, tokens=len(first["token_ids"]),
                rounds=len(first["rounds"]),safe=bool(safe),finish_reason=first["finish_reason"],quality=first.get("quality"),
                spent=first["asd"]["spent"] if first["asd"] else 0.,relaxed=first["asd"]["relaxed_tokens"] if first["asd"] else 0)
        sample = dict(request_id=request_id,study_split=split,arms={})
        for stage in ("natural","prefix"):
            for policy in POLICIES:
                common = all(costs[stage,v,policy]["tokens"] == 128 for v in VARIANTS)
                base = costs[stage,"base",policy]
                ordinary = costs[stage,"ordinary",policy]
                position = costs[stage,"ordinary-position",policy]
                for variant in VARIANTS:
                    item = costs[stage,variant,policy]
                    item.update(latency_ratio=item["seconds"]/base["seconds"], ordinary_ratio=item["seconds"]/ordinary["seconds"],
                                position_ratio=item["seconds"]/position["seconds"],
                                tps_ratio=(item["tokens"]/item["seconds"])/(base["tokens"]/base["seconds"]),
                                common_full_128=common,paired_safe=all(costs[stage,v,policy]["safe"] for v in VARIANTS))
                    if stage == "natural":
                        item.update(lost_correct=bool(base["quality"]["correct"] and not item["quality"]["correct"]),
                                    gained_correct=bool(not base["quality"]["correct"] and item["quality"]["correct"]),
                                    lost_vs_position=bool(position["quality"]["correct"] and not item["quality"]["correct"]),
                                    new_truncation=base["finish_reason"] == "eos" and item["finish_reason"] != "eos")
                    sample["arms"][f"{stage}/{variant}/{policy}"] = item
        samples.append(sample)
    scopes = {}
    for split in ("observed","independent"):
        selected = [s for s in samples if s["study_split"] == split]
        if not selected:
            continue
        summary = {}
        for stage in ("natural","prefix"):
            for variant in VARIANTS:
                for policy in POLICIES:
                    key = f"{stage}/{variant}/{policy}"
                    values = [s["arms"][key] for s in selected]
                    item = dict(requests=len(values),safe=all(v["paired_safe"] for v in values),
                                latency_ratio=statistics.mean(v["latency_ratio"] for v in values),
                                ordinary_ratio=statistics.mean(v["ordinary_ratio"] for v in values),
                                position_ratio=statistics.mean(v["position_ratio"] for v in values),
                                tps_ratio=statistics.mean(v["tps_ratio"] for v in values),
                                output_tokens=sum(v["tokens"] for v in values),target_calls=sum(v["rounds"] for v in values),
                                relaxed_tokens=sum(v["relaxed"] for v in values),regret_spent=sum(v["spent"] for v in values),
                                faster=sum(v["latency_ratio"] < 1 for v in values),
                                faster_vs_position=sum(v["position_ratio"] < 1 for v in values))
                    common = values if stage == "natural" else [v for v in values if v["common_full_128"]]
                    item["contrast_requests"] = len(common)
                    item["bootstrap"] = {n:bootstrap([v[n] for v in common]) for n in ("latency_ratio","ordinary_ratio","position_ratio")} if common else None
                    if stage == "natural":
                        item.update(correct=sum(v["quality"]["correct"] for v in values),lost_correct=sum(v["lost_correct"] for v in values),
                                    gained_correct=sum(v["gained_correct"] for v in values),lost_vs_position=sum(v["lost_vs_position"] for v in values),
                                    truncations=sum(v["finish_reason"] != "eos" for v in values),new_truncations=sum(v["new_truncation"] for v in values))
                    else:
                        item.update(common_full_128_requests=len(common),
                                    common_full_128_ratio=statistics.mean(v["latency_ratio"] for v in common) if common else None,
                                    common_full_128_vs_position=statistics.mean(v["position_ratio"] for v in common) if common else None)
                    summary[key] = item
        scopes[split] = summary
    return dict(scopes=scopes,samples=samples,all_safe=all(v["safe"] for s in samples for v in s["arms"].values()))


def main():
    base = ROOT/f"results/{STUDY}"
    execution = json.loads((base/"execution-verification.json").read_text(encoding="utf-8"))
    infos = [json.loads((base/f"eval-{i}/provenance.json").read_text(encoding="utf-8")) for i in (0,1)]
    train = json.loads((base/"train/provenance.json").read_text(encoding="utf-8"))
    collect = [json.loads((base/f"collect-{i}/provenance.json").read_text(encoding="utf-8")) for i in (0,1)]
    if (execution["status"] != "complete" or execution["files_verified_before"] != execution["files_verified_after"]
            or not execution["previous_private_assets_unchanged"]
            or any(p["status"] != "complete" or not p["original_unchanged"] for p in [*infos,train,*collect])):
        raise ValueError("Incomplete execution or changed previous evidence")
    if (train["updates"] != 128 or train["accumulation"] != 2 or train["train_states"] != 64 or train["calibration_states"] != 32
            or not train["teacher_parameter_versions_unchanged"] or not train["draft_base_versions_unchanged"]):
        raise ValueError("Unmatched training budget")
    rows = [r for i in (0,1) for r in read_rows(base/f"eval-{i}/rows.jsonl")]
    expected = {r["id"]:r["study_split"] for r in records(ROOT)}
    if len(rows) != 960 or {r["request_id"]:r["study_split"] for r in rows} != expected:
        raise ValueError("Changed evaluation coverage")
    result = analyze(rows)
    result["resources"] = audit_resources(infos,rows)
    result["training"] = train
    natural = result["scopes"]["independent"]["natural/asd-prefix/ASD"]
    prefix = result["scopes"]["independent"]["prefix/asd-prefix/ASD"]
    checks = dict(all_safe=result["all_safe"], no_formal_compile=result["resources"]["formal_compile_events"] == 0,
                  no_new_quality_loss=natural["lost_correct"] == 0 and natural["lost_vs_position"] == 0,
                  no_new_truncation=natural["new_truncations"] == 0, natural_faster_vs_position=natural["position_ratio"] < 1,
                  common_prefix_faster_two_percent=(prefix["common_full_128_vs_position"] is not None and prefix["common_full_128_vs_position"] <= .98))
    result["content_increment_gate"] = dict(checks=checks, point_candidate=all(checks.values()),
        both_position_intervals_below_one=(prefix["bootstrap"] is not None and natural["bootstrap"]["position_ratio"]["high"] < 1
                                          and prefix["bootstrap"]["position_ratio"]["high"] < 1),
        interpretation="Single-seed bounded evidence; a passing point gate is not broad confirmation or deployment approval")
    refs = [r for i in (0,1) for r in read_rows(base/f"eval-{i}/references.jsonl")]
    if len(refs) != 24 or {r["request_id"] for r in refs} != set(expected):
        raise ValueError("Incomplete fresh AR references")
    result["ar_quality"] = {s:dict(requests=sum(r["study_split"] == s for r in refs),
        correct=sum(r["quality"]["correct"] for r in refs if r["study_split"] == s)) for s in ("observed","independent")}
    write(base/"analysis.json",result)
    lines = ["# 阶段5后续：同位置严格前缀与独立GSM评估","",
             "仅新训ordinary-position，保持原ASD适配器不变、匹配其64训练状态的生成位置；相同32题、初始化、更新、内部校准和选模规则。旧8题单列，新16题为项目未使用的官方test题；不是上游未见数据或多seed确认。","",
             "|范围|阶段/模型/策略|时延/base|时延/普通|时延/同位置|TPS/base|正确或共同128题数|安全|",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for scope,summary in result["scopes"].items():
        for key,v in summary.items():
            quality = f"{v['correct']}/{v['requests']}" if key.startswith("natural") else str(v["common_full_128_requests"])
            lines.append(f"|{scope}|{key}|{v['latency_ratio']:.6f}|{v['ordinary_ratio']:.6f}|{v['position_ratio']:.6f}|{v['tps_ratio']:.6f}|{quality}|{v['safe']}|")
    lines += ["","共同128的实际比值及预先指定的请求级bootstrap、每题质量/截断/token/调用数见analysis.json。完整生成与共同128集合不同，不直接相减解释因果。新的严格源不必落在真实Draft轮边界，匹配的是Target确认前缀位置，不宣称这是线上可用的额外动作。",""]
    (ROOT/"docs/ASD_ADAPTATION_FOLLOWUP_RESULTS.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(dict(all_safe=result["all_safe"],ar_quality=result["ar_quality"],
                         independent={k:v for k,v in result["scopes"]["independent"].items() if k.endswith("/ASD")})))


if __name__ == "__main__":
    main()
