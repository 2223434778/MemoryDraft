"""Audit compact full-plan evidence and rebuild a report without model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from memorydraft.evaluation.matrix_probe import METHODS, SCORED, audit, inputs, observed_rows, protected, timing_gate
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.evaluation.splice_probe import input_digest, prefix_acceptance
from scripts.write_splice_report import audit_calls, preserve_diagnostic


ROOT = Path(__file__).resolve().parents[1]


def check_provenance(info, manifest, runner):
    files = {n.removeprefix("memorydraft/"): s for n, s in manifest["files"].items()
             if n.startswith("memorydraft/") and n.endswith((".py", ".json", ".txt"))}
    if (info["status"] != "complete" or not info["original_unchanged"] or info["original_sha256"] != protected(ROOT)
            or info["runner_sha256"] != manifest["files"][runner]
            or info["protocol_sha256"] != manifest["files"]["docs/OT_MATRIX_PROTOCOL.md"]
            or info["source"]["files"] != files
            or info["source"]["sha256"] != hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()):
        raise ValueError("Changed matrix execution provenance")


def audit_offline(source, manifest):
    tasks, events = inputs(ROOT)
    keyed = {t.request.id: t for t in tasks}
    info = json.loads((source / "offline/provenance.json").read_text())
    check_provenance(info, manifest, "scripts/probe_matrix_offline.py")
    if info["inputs_sha256"] != input_digest(tasks, events):
        raise ValueError("Changed offline inputs")
    recorded = observed_rows(ROOT)
    drafts = {}
    for row in recorded:
        for call in row["calls"]:
            drafts.setdefault((row["request_id"], call["generated"]), []).append(call["draft_tokens"])
    frames = read_rows(source / "offline/frames.jsonl")
    if len({f["state_sha256"] for f in frames}) != len(frames) or info["frames"] != len(frames):
        raise ValueError("Missing or duplicate offline state")
    for frame in frames:
        task = keyed[frame["request_id"]]
        remaining = task.reference["token_ids"][frame["generated"]:]
        pool = frame["candidates"]
        if (frame["snapshot_id"] != task.snapshot_id or frame["project_id"] != task.project_id or not 1 <= len(pool) <= 8
                or not any(prefix_acceptance(d, remaining) == frame["draft_accepted"] and
                           all(c["tokens"][:c["prefix_length"]] == d[:c["prefix_length"]] for c in pool)
                           for d in drafts[frame["request_id"], frame["generated"]])):
            raise ValueError("Changed common state, pool or Draft reference")
        for item in pool:
            if item["oracle_accepted"] != prefix_acceptance(item["tokens"], remaining):
                raise ValueError("Changed offline candidate acceptance")
            for method, assessment in item["assessments"].items():
                features = assessment["features"]
                if features and (set(features) != {"support", "concentration", "continuity", "dropped", "draft_weakest"}
                                 or not all(math.isfinite(v) and 0 <= v <= 1 for v in features.values())):
                    raise ValueError("Invalid full-plan features")
                if method == "Splice-PlanOT" and features:
                    expected = (features["support"] >= .5 and features["concentration"] >= .5
                                and features["continuity"] >= .5 and features["dropped"] <= .5)
                    reliability = .4 * features["support"] + .3 * features["concentration"] + .3 * features["continuity"]
                    if assessment["admissible"] != expected or (expected and not math.isclose(assessment["score"],
                            .5 * item["assessments"]["Splice-OT"]["score"] + .5 * reliability, abs_tol=1e-12)):
                        raise ValueError("Changed fixed structural rule")
        for method in SCORED:
            ranked = []
            for i, item in enumerate(pool):
                score = item["cheap"] if method == "Splice-Edit" else item["assessments"][method]["score"]
                if score is not None and score >= .65:
                    ranked.append((score, -item["prefix_length"], -item["position"], i))
            selected = max(ranked)[-1] if ranked else None
            expected = dict(selected=selected, accepted=pool[selected]["oracle_accepted"] if selected is not None
                            else frame["draft_accepted"], fallback=selected is None)
            if frame["decisions"][method] != expected:
                raise ValueError("Changed common-pool decision")
    replayed = sum(len(r["calls"]) for r in recorded)
    changed = {m: sum(f["decisions"][m]["selected"] != f["decisions"]["Splice-OT"]["selected"] for f in frames)
               for m in SCORED[-3:]}
    summary = dict(frames=len(frames), replayed_calls=replayed, changed=changed, gpu_gate=any(changed.values()),
                   methods={m: dict(accepted=sum(f["decisions"][m]["accepted"] for f in frames),
                                    fallback=sum(f["decisions"][m]["fallback"] for f in frames),
                                    worse_than_draft=sum(f["decisions"][m]["accepted"] < f["draft_accepted"] for f in frames))
                            for m in SCORED})
    if (summary != json.loads((source / "offline/summary.json").read_text())
            or info["replayed_calls"] != replayed or info["gpu_gate"] != summary["gpu_gate"]):
        raise ValueError("Changed offline summary or gate")
    return info, frames, summary


def load_gpu(source, stage, manifest, offline):
    tasks, events = inputs(ROOT, subset=True)
    info = json.loads((source / stage / "provenance.json").read_text())
    check_provenance(info, manifest, "scripts/run_matrix_probe.py")
    if (info["stage"] != stage or info["instrumented"] != (stage == "mechanism") or info["methods"] != list(METHODS)
            or info["inputs_sha256"] != input_digest(tasks, events) or info["source"] != offline["source"]
            or info["offline_summary_sha256"] != file_hash(source / "offline/summary.json")):
        raise ValueError("Changed matrix GPU scope")
    rows = read_rows(source / stage / "rows.jsonl")
    summary = audit(rows, tasks, 1 if stage == "mechanism" else 2, instrumented=stage == "mechanism")
    if len(rows) != info["rows"] or summary != json.loads((source / stage / "summary.json").read_text()):
        raise ValueError("Changed GPU summary")
    if stage == "mechanism":
        audit_calls(rows, tasks)
        if info["timing_gate"] != timing_gate(rows):
            raise ValueError("Changed matrix timing gate")
    return info, rows, summary


def build(source):
    manifest = json.loads((source / "execution-source.json").read_text())
    offline, frames, summary = audit_offline(source, manifest)
    quality = {}
    for method in SCORED:
        good, bad, missed = 0, 0, 0
        for frame in frames:
            selected = frame["decisions"][method]["selected"]
            pool = frame["candidates"]
            if selected is None:
                missed += int(any(c["oracle_accepted"] > c["prefix_length"] for c in pool))
            else:
                reaches = pool[selected]["oracle_accepted"] > pool[selected]["prefix_length"]
                good += int(reaches)
                bad += int(not reaches)
        quality[method] = dict(reaches_history=good, no_history_accepted=bad, abstained_available_history=missed)
    diagnostic = dict(offline=summary, common_state_quality=quality)
    lines = ["# OT完整矩阵：三个小改动的开发结果", "", "2026-10-19。全部任务和Target答案此前已观察；本轮为机制开发检查，不是独立holdout。", "",
             "A=末端加权；B=基于初始匹配流的相邻行局部顺序代价修正/第二次求解；C=锚点之前的支持度、集中度、连续性、dustbin质量排序/放弃。All=A+B+C。",
             "固定公式/多数支持阈值见OT_MATRIX_PROTOCOL.md，没有接受概率校准或按答案调参；候选长度仍固定上限32，本轮不实现预计收益/动态长度。", "",
             f"旧记录{summary['replayed_calls']}次实际调用全部重现；得到{len(frames)}个去重可评分状态、{sum(len(f['candidates']) for f in frames)}个候选。参考答案只在评分后离线计算连续接受。", "",
             "| 方法 | 同状态连续接受总和 | 回退Draft状态 | 低于原Draft的状态 |", "|---|---:|---:|---:|"]
    for method, values in summary["methods"].items():
        lines.append(f"| {method} | {values['accepted']} | {values['fallback']} | {values['worse_than_draft']} |")
    lines += ["", "接续可靠性诊断：至少接受一个历史尾部token才算到达历史；误放弃指池中存在可到达历史的候选但回退Draft。以下仍是同状态事后描述。", "",
              "| 方法 | 选中并到达历史 | 选中但未接受历史 | 放弃池内可用接续 |", "|---|---:|---:|---:|"]
    for method, values in quality.items():
        lines.append(f"| {method} | {values['reaches_history']} | {values['no_history_accepted']} | {values['abstained_available_history']} |")
    lines += ["", "这些状态来自多条已观察路径，不是独立请求；总和不代表在线新增token或端到端加速。B/C/All相对原OT选择变化数：" + str(summary["changed"]) + "。", "",
              "| 请求/位置 | 原Draft | " + " | ".join(SCORED) + " |", "|---|---:|" + "---:|" * len(SCORED)]
    for frame in frames:
        lines.append(f"| {frame['request_id']}/{frame['generated']} | {frame['draft_accepted']} | " +
                     " | ".join(str(frame["decisions"][m]["accepted"]) + (" (D)" if frame["decisions"][m]["fallback"] else "") for m in SCORED) + " |")
    if summary["gpu_gate"]:
        info, rows, mechanism = load_gpu(source, "mechanism", manifest, offline)
        diagnostic.update(mechanism=mechanism, timing_gate=info["timing_gate"])
        lines += ["", f"GPU机制回放{len(rows)}/{len(rows)}输出token、Target输入、合法快照与原AR严格一致。", "",
                  "| 方法 | 总轮 | 实际Draft前向 | 对齐提议 | 历史尾部接受 | 矩阵拒绝候选 | 新求解次数 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            c = mechanism["all"][method]["counters"]
            forwards = c["draft_rounds"] if method == "Suffix-R" else c["draft_forward_rounds"]
            lines.append(f"| {method} | {c['rounds']} | {forwards} | {c['alignment_proposals']} | {c['splice_history_accepted_tokens']} | {c['matrix_rejections']} | {c['matrix_solves']} |")
        lines += ["", "插桩时延不作速度证据。计时门槛：" + ("通过。" if info["timing_gate"]["passed"] else "未通过，停止GPU扩展。")]
        if info["timing_gate"]["passed"]:
            timing_info, timed, timings = load_gpu(source, "timing", manifest, offline)
            diagnostic["timing"] = timings
            lines += ["", f"未插桩交错计时{len(timed)}/{len(timed)}严格一致。每请求两重复，第二次求解、特征提取及放弃成本均计入。", "",
                      "| 方法 | 总秒 | 相对Guided-Edit延迟 | 相对原Splice-OT延迟 |", "|---|---:|---:|---:|"]
            for method in METHODS:
                seconds = timings["all"][method]["seconds"]
                lines.append(f"| {method} | {seconds:.6f} | {(seconds/timings['all']['Guided-Edit']['seconds']-1)*100:+.2f}% | {(seconds/timings['all']['Splice-OT']['seconds']-1)*100:+.2f}% |")
            lines += ["", "| 请求 | " + " | ".join(METHODS) + " |", "|---|" + "---:|" * len(METHODS)]
            for request in dict.fromkeys(r["request_id"] for r in rows):
                lines.append("| " + request + " | " + " | ".join(f"{sum(r['latency_seconds'] for r in timed if r['request_id']==request and r['method']==m):.6f}" for m in METHODS) + " |")
            base = {(r["request_id"], r["method"]): r for r in rows}
            keys = ("rounds", "draft_rounds", "reuse_rounds", "reuse_accepted_tokens", "alignment_calls",
                    "alignment_candidates", "alignment_proposals", "alignment_accepted_tokens", "draft_forward_rounds",
                    "matrix_rejections", "matrix_solves")
            same = all(all(r["counters"].get(k, 0) == base[r["request_id"], r["method"]]["counters"].get(k, 0) for k in keys) for r in timed)
            diagnostic["timing_counter_paths_match_mechanism"] = same
            lines += ["", f"未插桩与机制回放计数路径一致：{same}。"]
    examples = []
    lines += ["", "具体错误定位（候选接受量仅作事后诊断）：", "",
              "| 请求/位置 | 候选 | 保留Draft | 连续接受 | 支持度 | 连续性 | C允许 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for request, position in (("confirm-schema-0:0", 130), ("payment-runbook:1", 15), ("shipment-contract:1", 22)):
        frame = next(f for f in frames if f["request_id"] == request and f["generated"] == position)
        for i, item in enumerate(frame["candidates"][:3]):
            assessment = item["assessments"]["Splice-PlanOT"]
            f = assessment["features"]
            examples.append(dict(request_id=request, generated=position, candidate=i, prefix=item["prefix_length"],
                                 accepted=item["oracle_accepted"], features=f, allowed=assessment["admissible"]))
            lines.append(f"| {request}/{position} | {i} | {item['prefix_length']} | {item['oracle_accepted']} | {f['support']:.3f} | {f['continuity']:.3f} | {assessment['admissible']} |")
    diagnostic["illustrative_candidates"] = examples
    lines += ["", "局限与判断：", "",
              "- B在14个共同状态的选择与原OT完全相同，GPU总轮也同为67；第二次求解增加了成本。",
              "- C将选中但未接受历史的状态4→1，同时误放弃两处池内可用接续；旧JSON可接受32的候选因连续性0.469<0.5被拒。GPU总轮67→69，说明更保守没有换来整体提速。",
              "- runbook错误候选保留14个Draft但仅接受7，连续性却为0.704；正确的32-token候选连续性0.397而被拒。既有固定阈值误拒，也有对齐质量与Target接受错位；只降低阈值不保证选对。",
              "- 组合相对原OT总时延约低0.16%，但仍比已测TailOT高约2.07%。C在JSON负例减少部分无效验证开销，仍未胜过便宜Edit；不能据局部减少浪费宣称整体或OT独有收益。",
              "- 相邻token在历史中的有序匹配是上下文对齐证据，不能证明尚未确认的Draft前缀或后续原文会被Target接受。本批没有校准概率或新项目泛化证据。",
              "- 保留全部正负结果，不在本批上继续换阈值宣称确认。默认suffix和独立infra面试版保持原样。", "",
              f"冻结执行提交`{manifest['commit']}`，package SHA `{offline['source']['sha256']}`。", "",
              "```bash", "python -m scripts.write_matrix_report", "```", ""]
    return "\n".join(lines), diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/ot-matrix")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/OT_MATRIX_RESULTS.md")
    args = parser.parse_args()
    report, diagnostic = build(args.source)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    preserve_diagnostic(args.source / "diagnostic.json", diagnostic)


if __name__ == "__main__":
    main()
