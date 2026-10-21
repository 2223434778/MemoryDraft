"""Independent validation with preregistered engineering admission criteria."""
import json
from pathlib import Path

from memorydraft.evaluation.runtime_migration import read_rows
from scripts.diagnose_transport_online import write
from scripts import write_asd_rmsnorm_suite_report as common
from scripts.run_asd_rmsnorm_suite_probe import METHODS, PROFILES
from scripts.run_asd_rmsnorm_validation_probe import CANDIDATE_FILES, records_and_positions

ROOT = Path(__file__).resolve().parents[1]


def admission(summary, audit, rows, references, info, execution):
    first = {(r["request_id"], r["method"], r["profile"]): r for r in
             (*references, *[r for r in rows if r["repeat"] == 0])}
    pairs = [(a, first[q, m, PROFILES[1]]) for (q, m, p), a in first.items() if p == "eager"]
    identity = all(a["token_ids"] == b["token_ids"] and a["text"] == b["text"] and a["finish_reason"] == b["finish_reason"]
                   for a, b in pairs)
    quality = all(int(b["quality"]["correct"]) >= int(a["quality"]["correct"])
        and int(b["quality"].get("secondary_correct", b["quality"]["correct"])) >= int(a["quality"].get("secondary_correct", a["quality"]["correct"]))
        and b["quality"].get("f1", 0) >= a["quality"].get("f1", 0)
        and not (common.assessed(a) and not common.assessed(b))
        and not (a["finish_reason"] == "eos" and b["finish_reason"] != "eos") for a, b in pairs)
    resource = all(p["compiler"]["graph_count"] <= 8 and p["compiler"]["backend_seconds"] <= 300
                   and p["cache_bytes"] <= 4*2**30 and p["compiler"]["float32_matmul_precision"] == "highest"
                   and not p["compiler"]["allow_tf32"] for p in info)
    source = (execution["status"] == "complete" and execution["files_verified_before"] == execution["files_verified_after"]
        and all(p["status"] == "complete" and p["original_unchanged"] and p["split"] == "validation"
                and p["initial_isolation"]["protected_generation_matches"] == 0
                and p["initial_isolation"]["candidate"]["files"] == CANDIDATE_FILES for p in info))
    checks = dict(all_240_full_outputs_identical=len(pairs) == 240 and identity,
                  no_observed_quality_or_completion_regression=quality,
                  strict_DFlash_matches_AR_in_both_profiles=all(v == 48 for v in audit["strict_matches_AR"].values()),
                  all_four_overall_ci_upper_below_one=all(v["ci95"][1] < 1 for v in summary["overall"].values()),
                  no_family_point_regression=all(v["ratio"] < 1 for methods in summary["families"].values() for v in methods.values()),
                  resource_bounds=resource, frozen_source_and_initial_isolation=source)
    return dict(eligible_for_explicit_profile=all(checks.values()), checks=checks,
                scope="Engineering gate for an explicit numerical profile, not a universal lossless or statistical quality-noninferiority proof. Reserve remains ungenerated; failures are retained, not retuned on validation.")


def main():
    base = ROOT/"results/asd-rmsnorm-validation"
    read = lambda name: [r for i in (0, 1) for r in read_rows(base/f"part-{i}/{name}.jsonl")]
    rows, references = read("rows"), read("references")
    records = {r["id"]: r for r in records_and_positions()[0]}
    info = [json.loads((base/f"part-{i}/provenance.json").read_text()) for i in (0, 1)]
    execution = json.loads((base/"execution-verification.json").read_text())
    if execution["status"] != "complete" or execution["files_verified_before"] != execution["files_verified_after"]:
        raise ValueError("Incomplete source verification")
    if any(p["status"] != "complete" or p["split"] != "validation" or not p["original_unchanged"] for p in info):
        raise ValueError("Incomplete validation execution")
    if info[0]["stop_token_ids"] != info[1]["stop_token_ids"]:
        raise ValueError("Changed stop token contract")
    audit = common.audit(rows, references, records, None, info[0]["stop_token_ids"])
    summary = common.summarize(rows, references, records, seed="rmsnorm-validation", development_subsets=False)
    result = dict(audit=audit, **summary, admission=admission(summary, audit, rows, references, info, execution),
                  preparation=[dict(shard=p["shard"], controller_setup_seconds=p["controller_setup_seconds"],
                                    warmup_wall_seconds=p["warmup_wall_seconds"], compiler=p["compiler"],
                                    cache_bytes=p["cache_bytes"], initial_isolation=p["initial_isolation"]) for p in info])
    write(base/"analysis.json", result)
    lines = ["# RMSNorm冻结候选独立验证", "", "48 validation题，六类各8；1152配对计时+96 AR完整输出对照，另20短warmup。上限4096/自然EOS，reserve96未生成。首轮同时评分，不额外加算生成；未按验证结果调参。", "",
             "|方法|编译/eager自然时延比|题级分层95% bootstrap|更快题|", "|---|---:|---|---:|"]
    for m, v in summary["overall"].items():
        lines.append(f"|{m}|{v['ratio']:.6f}|[{v['ci95'][0]:.6f},{v['ci95'][1]:.6f}]|{v['faster']}/48|")
    lines += ["", "## 预登记工程判定", "", "|条件|通过|", "|---|---|"]
    for k, v in result["admission"]["checks"].items():
        lines.append(f"|{k}|{v}|")
    lines += ["", "显式profile接入条件全部满足："+str(result["admission"]["eligible_for_explicit_profile"]), "",
              "## 分任务与质量", "", "|任务|方法|时延比[95%]|token相同|主质量提升/下降/未评估|", "|---|---|---|---:|---|"]
    for b, methods in summary["families"].items():
        for m, v in methods.items():
            lines.append(f"|{b}|{m}|{v['ratio']:.6f} [{v['ci95'][0]:.6f},{v['ci95'][1]:.6f}]|{v['token_matches']}/8|{v['quality_improved']}/{v['quality_lost']}/{v['quality_unassessed']}|")
    lines += ["", "|任务|方法|eager/编译主正确|eager/编译secondary|eager/编译截断|未评估|SQuAD F1|", "|---|---|---|---|---|---|---|"]
    for b, methods in summary["quality"].items():
        for m, item in methods.items():
            a, c = item["eager"], item[PROFILES[1]]
            lines.append(f"|{b}|{m}|{a['correct']}/{c['correct']}（各8）|{a['secondary_correct']}/{c['secondary_correct']}|{a['truncated']}/{c['truncated']}|{len(a['unassessed'])}/{len(c['unassessed'])}|{a['f1_mean']}/{c['f1_mean']}|")
    lines += ["", "## 完整性与范围", "", json.dumps(audit, ensure_ascii=False), "",
              "验证题没有旧生成参考；两profile使用本轮匹配eager对照及相同Target输入。每profile三次重复精确重现；所有数值分歧/错答/截断保留。逐题首次分歧、长度/调用密度/摊销成本和质量详见analysis.json及原始gzip JSONL。", "",
              "MATH规范答案主评分与有限标量等价分列，代码仅提供测试，SQuAD EM/F1，JSON严格合同；未评估与待复核不可冒充已确认失败。AR仅一次输出/质量对照，不统计AR速度。", "",
              "判定是本轮显式执行选项的工程准入，不是任意输入无损或统计质量非劣证明。未通过不删除样本、不在validation调参；reserve保持未生成。", "",
              "重建：`.venv\\Scripts\\python.exe -m scripts.write_asd_rmsnorm_validation_report`；协议ASD_RMSNORM_VALIDATION_PROTOCOL.md。", ""]
    (ROOT/"docs/ASD_RMSNORM_VALIDATION_RESULTS.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(dict(audit=audit, overall=summary["overall"], admission=result["admission"])))


if __name__ == "__main__":
    main()
