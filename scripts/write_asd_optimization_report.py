"""Audit and summarize the frozen optimization matrix without tuning it."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from memorydraft.evaluation.runtime_migration import read_rows
from scripts.diagnose_transport_online import write
from scripts.run_asd_optimization_pipeline import select_transfer

ROOT = Path(__file__).resolve().parents[1]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["experiment_profile"], row["arm"]].append(row)
    result = {}
    for (profile, arm), items in groups.items():
        natural = [r for r in items if r["max_new_tokens"] == 2048]
        prefix = [r for r in items if r["max_new_tokens"] == 128]
        n_seconds = sum(r["latency_seconds"] for r in natural)
        p_seconds = sum(r["latency_seconds"] for r in prefix)
        result[profile + "/" + arm] = dict(tasks=len(natural), natural_seconds=n_seconds,
                natural_tokens=sum(len(r["token_ids"]) for r in natural),
                correct=sum(r["quality"]["correct"] for r in natural),
                truncated=sum(r["finish_reason"] != "eos" for r in natural),
                prefix_seconds=p_seconds, prefix_complete=all(len(r["token_ids"]) == 128 for r in prefix),
                prefix_tokens=sum(len(r["token_ids"]) for r in prefix),
                prefix_rounds=sum(len(r["rounds"]) for r in prefix),
                prefix_tps=sum(len(r["token_ids"]) for r in prefix) / p_seconds if p_seconds else None,
                short_decisions=sum(k == 7 for r in prefix for k in r["selected_lengths"]),
                selected_decisions=sum(len(r["selected_lengths"]) for r in prefix))
    return result


def draft_summary(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = row["request_id"], row["position"]
        if row["arm"] in groups[key]:
            raise ValueError("Duplicate draft diagnostic")
        groups[key][row["arm"]] = row
    pairs = list(groups.values())
    if any(set(p) != {"proposal", "recovery"} or p["proposal"]["common_sha256"] != p["recovery"]["common_sha256"] for p in pairs):
        raise ValueError("Unpaired draft diagnostic")
    common = [p for p in pairs if all(r["new_tokens"] == 128 and r["first_round"] for r in p.values())]
    first = {arm: sum(p[arm]["first_round"]["accepted"] for p in common) for arm in ("proposal", "recovery")}
    rounds = {arm: sum(p[arm]["rounds"] for p in common) for arm in first}
    ratio = first["proposal"] / first["recovery"] if first["recovery"] else None
    density_ratio = rounds["proposal"] / rounds["recovery"] if rounds["recovery"] else None
    enough = len(common) >= 8 and len({p["proposal"]["request_id"] for p in common}) >= 3
    return dict(pairs=len(pairs), equal128_pairs=len(common), first_accepted=first, total_rounds=rounds,
                first_accepted_ratio=ratio, round_density_ratio=density_ratio,
                training_gate=bool(enough and ratio is not None and ratio <= .8 and density_ratio >= 1.1))


def audit_part(path):
    from memorydraft.evaluation.asd_budget import score
    from scripts.run_asd_optimization_probe import records

    data = records()
    info = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
    if info["status"] != "complete" or not info["original_unchanged"]:
        raise ValueError("Incomplete or changed execution: " + str(path))
    rows = read_rows(path / "rows.jsonl")
    expected = len(info["task_ids"]) * len(info["arms"]) * 3
    if len(rows) != expected:
        raise ValueError("Missing matrix rows")
    seen = {}
    for r in rows:
        key = r["request_id"], r["arm"], r["max_new_tokens"], r["repeat"]
        if key in seen:
            raise ValueError("Duplicate matrix row")
        seen[key] = r
        if (not 0 < len(r["token_ids"]) <= r["max_new_tokens"] or
                (r["finish_reason"] == "eos") != (r["token_ids"][-1] in info["stop_token_ids"]) or
                r["asd"]["spent"] > r["asd"]["budget"] + 1e-9 or r["latency_seconds"] <= 0):
            raise ValueError("Invalid generation/ledger/timing")
        if r["quality"] != score(data[r["request_id"]], r["text"]):
            raise ValueError("Changed final-answer scoring")
        if r["max_new_tokens"] == 128 and r["repeat"] == 1:
            first = seen.get((r["request_id"], r["arm"], 128, 0))
            if first is None:
                first = next(s for s in rows if (s["request_id"], s["arm"], s["max_new_tokens"], s["repeat"]) == (r["request_id"], r["arm"], 128, 0))
            if (first["token_ids"], first["asd"], first["rounds"]) != (r["token_ids"], r["asd"], r["rounds"]):
                raise ValueError("Nondeterministic timed repeats")
        if r["arm"].startswith("D-") and any(x.get("regret", 0) for x in r["rounds"]):
            raise ValueError("Strict arm spent approximate regret")
    expected_keys = {(q, arm, limit, rep) for q in info["task_ids"] for arm in info["arms"]
                     for limit, rep in ((2048, 0), (128, 0), (128, 1))}
    if set(seen) != expected_keys:
        raise ValueError("Changed frozen matrix identities")
    for q in info["task_ids"]:
        for r in [x for x in rows if x["request_id"] == q]:
            identity = r["prompt_sha256"], r["snapshot_id"]
            if identity != (rows[next(i for i,x in enumerate(rows) if x["request_id"] == q)]["prompt_sha256"],
                            rows[next(i for i,x in enumerate(rows) if x["request_id"] == q)]["snapshot_id"]):
                raise ValueError("Changed Target prompt/history")
        for old, fast in (("D-old", "D-fast"), ("U-old", "U-fast")):
            if old in info["arms"]:
                for limit, rep in ((2048, 0), (128, 0), (128, 1)):
                    a, b = seen[q, old, limit, rep], seen[q, fast, limit, rep]
                    if (a["token_ids"], a["asd"]) != (b["token_ids"], b["asd"]):
                        raise ValueError("Execution optimization changed output")
    ar = {r["request_id"]: r for r in read_rows(path / "ar.jsonl")}
    numeric = []
    for q in info["task_ids"]:
        for arm in info["arms"]:
            if not arm.startswith("D-"):
                continue
            baseline = seen[q, arm, 128, 0]["token_ids"]
            reference = ar[q]["token_ids"]
            first = next((i for i,(a,b) in enumerate(zip(baseline, reference)) if a != b), min(len(baseline), len(reference)))
            numeric.append(dict(request_id=q, arm=arm, exact=baseline == reference, common_prefix=first,
                                ar_tokens=len(reference), d_tokens=len(baseline)))
    return [{**r, "experiment_profile": info["profile"]} for r in rows], numeric


def main():
    base = ROOT / "results/asd-optimization"
    verification = json.loads((base / "execution-verification.json").read_text(encoding="utf-8"))
    if verification["status"] != "complete" or verification["files_verified_before"] != verification["files_verified_after"]:
        raise ValueError("Incomplete source verification")
    all_rows, numeric, pairs = defaultdict(list), {}, []
    controls = dict(ar128=0, profilers=0)
    for path in sorted(base.iterdir()):
        if not path.is_dir() or not path.name.startswith(("development-", "transfer-")):
            continue
        rows, comparison = audit_part(path)
        controls["ar128"] += len(read_rows(path / "ar.jsonl"))
        controls["profilers"] += int((path / "profile.json").exists())
        stage = "development" if path.name.startswith("development-") else "transfer"
        all_rows[stage].extend(rows)
        numeric[path.name] = comparison
        if (path / "draft-pairs.jsonl.gz").exists():
            pairs.extend(read_rows(path / "draft-pairs.jsonl"))
    extension_info = None
    extension_base = base / "extension"
    if extension_base.exists():
        extension_info = json.loads((extension_base / "execution-verification.json").read_text(encoding="utf-8"))
        if extension_info["status"] != "complete" or extension_info["files_verified_before"] != extension_info["files_verified_after"]:
            raise ValueError("Incomplete extension source verification")
        for path in sorted(extension_base.iterdir()):
            if path.is_dir() and path.name.startswith(("attribution-", "transfer-")):
                rows, comparison = audit_part(path)
                controls["ar128"] += len(read_rows(path / "ar.jsonl"))
                controls["profilers"] += int((path / "profile.json").exists())
                stage = "extension-attribution" if path.name.startswith("attribution-") else "extension-transfer"
                all_rows[stage].extend(rows)
                numeric["extension/" + path.name] = comparison
        from scripts.run_asd_optimization_extension import choose_lengths
        rebuilt = choose_lengths([r for r in all_rows["extension-attribution"] if r["experiment_profile"] == "bf16-sdpa"])
        if rebuilt != json.loads((extension_base / "length-selection.json").read_text(encoding="utf-8")):
            raise ValueError("Extension length selection differs from frozen rule")
    choice = select_transfer(all_rows["development"])
    if choice != json.loads((base / "transfer-selection.json").read_text(encoding="utf-8")):
        raise ValueError("Transfer selection differs from frozen rule")
    old = {}
    for part in ("math-a", "mixed-b"):
        for r in read_rows(ROOT / "results/asd-confidence" / part / "mechanism/rows.jsonl"):
            old[r["request_id"], r["method"]] = r
    reproduced = 0
    for row in all_rows["development"]:
        if row["experiment_profile"] == "fp32-eager" and row["max_new_tokens"] == 2048 and row["arm"] in {"D-old", "U-old"}:
            reference = old[row["request_id"], "DFlash" if row["arm"] == "D-old" else "ASD-Utility"]
            for field in ("token_ids", "prompt_sha256", "snapshot_id"):
                if row[field] != reference[field]:
                    raise ValueError("Previous baseline changed: " + field)
            if row["arm"] == "U-old" and row["asd"] != reference["asd"]:
                raise ValueError("Previous Utility ledger changed")
            reproduced += 1
    if reproduced != 8:
        raise ValueError("Missing old baseline reproductions")
    summary = {stage: summarize(rows) for stage, rows in all_rows.items()}
    draft = draft_summary(pairs)
    analysis = dict(source_commit=verification["commit"], extension_commit=extension_info["commit"] if extension_info else None,
                    rows={k:len(v) for k,v in all_rows.items()},
                    summary=summary, numeric_compatibility=numeric, draft=draft, selection=choice,
                    previous_baselines_reproduced=reproduced, controls={**controls, "draft_branches": len(pairs)})
    write(base / "analysis.json", analysis)
    lines = ["# ASD 剩余优化空间排查", "", "全部为已观察任务的探索性排查；自然计时每题每臂1次，128前缀2次。错误/截断全部保留，不推断显著性或质量非劣。", "",
             "严格profile与近似策略分开；改变dtype/attention不是ASD独有收益。生产默认仍FP32/eager、严格验证。", "",
             "追加归因在看过首轮BF16正信号后单独冻结，不冒充原预注册，也不将重复旧题合并成独立样本。", ""]
    lines += ["## 主要判断", ""]
    for stage in ("development", "transfer"):
        table = summary[stage]
        old = table["fp32-eager/D-fast"]
        mixed = table["bf16-sdpa/D-fast"]
        asd = table["bf16-sdpa/ASD-fast"]
        lines.append(f"- {stage}：BF16/SDPA严格D的等128秒较FP32/eager变化{(mixed['prefix_seconds']/old['prefix_seconds']-1)*100:+.2f}%；同BF16下基础ASD较严格D变化{(asd['prefix_seconds']/mixed['prefix_seconds']-1)*100:+.2f}%。这些是固定前128token的探索性数据，不能推断完整长请求或质量非劣。")
    if not any(g["passed"] for g in choice["length_gates"].values()):
        lines.append("- FP32下全部6个长度变体未通过原速度门槛；不进入参数扫描。增量EOS逐ID/账本保持，但没有可确认收益，保留显式选项。")
    if extension_info:
        table = summary["extension-attribution"]
        for arm in ("D-fast", "ASD-fast", "U-fast"):
            eager, sdpa = table["bf16-eager/" + arm], table["bf16-sdpa/" + arm]
            lines.append(f"- BF16追加归因{arm}：SDPA/eager等128秒比{sdpa['prefix_seconds']/eager['prefix_seconds']:.4f}；不同attention仍需单列数值轨迹，不把该比值当ASD独有增量。")
        lines.append("- BF16全部6个长度复验均未通过门槛：短验证约慢15%–19%，短Draft约慢25%–29%，两种动态长度约慢1%；停止这些变体，不重新选阈值。逐臂比值见extension/length-selection.json。")
        lines.append("- BF16/eager自身也明显快于原FP32/eager，支持精度变化是执行收益的重要来源；SDPA相对BF16/eager的作用随轨迹变化，不能解释为统一内核加速百分比。追加数据用于归因，未用于重新挑选迁移配置。")
    lines.append("- 主开发/迁移BF16-SDPA严格D与同配置AR在6/8题的128前缀一致；math500:7位置118、gsm8k:25位置9出现数值路径分歧。该组合不能作为逐token无损配置晋升默认，当前生产合同保持不变。")
    lines.append("- 本轮决定：保留基础ASD/简单Utility研究选项与执行配置证据，停止长度/预算调度变体及ASD前缀适配训练投入。没有穷尽其他模型、大并发引擎或全部训练方案，不能据此宣称ASD已无全局优化空间。")
    lines += [""]
    for stage, table in summary.items():
        lines += ["## " + stage, "", "|profile/臂|自然秒|token|正确|截断|128秒（两次）|128轮|等128完整|短验证次数|", "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
        for name, m in sorted(table.items()):
            lines.append(f"|{name}|{m['natural_seconds']:.6f}|{m['natural_tokens']}|{m['correct']}/{m['tasks']}|{m['truncated']}|{m['prefix_seconds']:.6f}|{m['prefix_rounds']}|{m['prefix_complete']}|{m['short_decisions']}|")
        lines.append("")
    lines += ["## 数值兼容性", "", "AR和严格DFlash在同一profile的实际128前缀比较；一题可能检查多个严格长度臂。不一致保留，不能声称逐ID无损。", "", "|运行分片|一致检查/总检查|首分歧（零起始）|", "|---|---|---|"]
    for part, items in numeric.items():
        failures = ", ".join(f"{r['request_id']}/{r['arm']}:{r['common_prefix']}" for r in items if not r["exact"]) or "无"
        lines.append(f"|{part}|{sum(r['exact'] for r in items)}/{len(items)}|{failures}|")
    lines += ["", "## Draft前缀适应诊断", "", f"{draft['pairs']}对单token分支，其中双方达128的{draft['equal128_pairs']}对。第一块严格接受总数proposal/recovery={draft['first_accepted']}，后续128总轮={draft['total_rounds']}。",
              f"第一块接受比={draft['first_accepted_ratio']}，每token轮比={draft['round_density_ratio']}；冻结训练必要性门槛通过={draft['training_gate']}。",
              "", "分支分别重prefill、之后严格续写；两个实际前缀可有不同难度。这是部署分布问题的必要性检查，不是训练效果或在线速度证据。", "",
              "## 范围", "", "验证截短不退还完整Draft计算；缩短Draft块需真实重跑，不能用长Draft事后裁剪冒充。长度调度只读过去推进与账本。动态截短后门控作用于实际块，因此自然轨迹可改变，必须分列质量与长度。",
              "", "profiler独立带插桩，排除主计时。迁移选择由运行前冻结规则重建；仅当开发有至少5%等128优势且质量/截断通过才追加对应臂，没有事后扫描阈值。",
              "", f"审计主生成{sum(analysis['rows'].values())}次，另{len(pairs)}条Draft分支、{controls['ar128']}条AR128和{controls['profilers']}次独立profiler；短warmup不计入主结果。8条原FP32完整D/Utility基线逐ID/输入/账本重现，旧结果原始SHA核验见postrun-audit.json。",
              "", "重建：`.venv\\Scripts\\python.exe -m scripts.write_asd_optimization_report`。协议`docs/ASD_OPTIMIZATION_PROTOCOL.md`及`docs/ASD_OPTIMIZATION_EXTENSION_PROTOCOL.md`，原始证据`results/asd-optimization/`。", ""]
    (ROOT / "docs/ASD_OPTIMIZATION_RESULTS.md").write_bytes("\n".join(lines).encode("utf-8"))
    print(json.dumps(dict(rows=analysis["rows"], selection=choice, draft=draft), ensure_ascii=False))


if __name__ == "__main__":
    main()
