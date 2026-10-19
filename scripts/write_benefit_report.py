"""Rebuild cost-routing evidence and its report without model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from memorydraft.evaluation.benefit_probe import METHODS, audit, calibration, inputs, policy_data, protected, timing_gate
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.evaluation.splice_probe import input_digest
from memorydraft.retrieval.benefit import BenefitPolicy
from scripts.probe_benefit_offline import decisions
from scripts.write_splice_report import audit_calls, preserve_diagnostic


ROOT = Path(__file__).resolve().parents[1]


def check_provenance(info, manifest, runner):
    files = {n.removeprefix("memorydraft/"): h for n, h in manifest["files"].items()
             if n.startswith("memorydraft/") and n.endswith((".py", ".json", ".txt"))}
    if (info["status"] != "complete" or not info["original_unchanged"] or info["original_sha256"] != protected(ROOT)
            or info["runner_sha256"] != manifest["files"][runner] or info["source"]["files"] != files
            or info["source"]["sha256"] != hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()):
        raise ValueError("Changed benefit execution provenance")
    if "protocol_sha256" in info and info["protocol_sha256"] != manifest["files"]["docs/OT_BENEFIT_PROTOCOL.md"]:
        raise ValueError("Changed frozen benefit protocol")


def audit_costs(source):
    costs = json.loads((source / "cost/costs.json").read_text())
    rows = json.loads((source / "cost/measurements.json").read_text())
    if len(rows) != 43 or any(len(r["seconds"]) != 7 or not all(math.isfinite(x) and x > 0 for x in r["seconds"]) for r in rows):
        raise ValueError("Missing cost measurements")
    for p in costs["profiles"]:
        for length, seconds in p["verify"].items():
            actual = next(r for r in rows if r["label"] == "target_verify" and r["context"] == p["context"] and r["length"] == int(length))
            if seconds != statistics.median(actual["seconds"]):
                raise ValueError("Changed Target cost")
        actual = next(r for r in rows if r["label"] == "draft_forward" and r["context"] == p["context"])
        if p["draft"] != statistics.median(actual["seconds"]):
            raise ValueError("Changed Draft cost")
    if (costs["lookup"] != statistics.median(statistics.median(r["seconds"]) for r in rows if r["label"] == "shortlist_and_policy")
            or costs["ot_candidate"] != statistics.median(statistics.median(r["seconds"]) / r["candidates"] for r in rows if r["label"] == "tail_ot_pool")):
        raise ValueError("Changed retrieval cost")
    return costs


def audit_decisions(rows, costs, tasks):
    keyed = {t.request.id: t for t in tasks}
    samples = calibration(ROOT)
    for row in rows:
        if row["method"] not in METHODS[-3:]:
            if row["decisions"]:
                raise ValueError("Policy trace on a baseline")
            continue
        policy = BenefitPolicy(**policy_data(samples, costs, row["project_id"]))
        calls = {c["generated"]: c for c in row["calls"]}
        for d in row["decisions"]:
            generated = d["generated"]
            c = calls[generated]
            expected = policy.predict_features(d["features"], len(c["draft_tokens"]),
                        min(32, keyed[row["request_id"]].request.max_new_tokens - generated),
                        row["prompt_tokens"] + generated - 1,
                        allow_ot=row["method"] != "Budget-Edit", adaptive=row["method"] != "Gate-OT")
            if d != dict(generated=generated, **expected) or row["project_id"] in d["projects"]:
                raise ValueError("Changed pre-OT benefit decision")
            if c["proposal"] and (len(c["proposal"]["tokens"]) > d["choice"]["length"]
                                  or c["proposal"]["lookup"] != "benefit-" + d["choice"]["route"] + "-splice"):
                raise ValueError("Proposal disagrees with the gate")
        if (len(row["decisions"]) != row["counters"]["benefit_decisions"]
                or sum(d["choice"]["route"] == "ot" for d in row["decisions"]) != row["counters"]["benefit_ot_calls"]):
            raise ValueError("Missing policy/OT accounting")


def build(source):
    manifest = json.loads((source / "execution-source.json").read_text())
    for stage, runner in (("cost", "profile_benefit_costs"), ("offline", "probe_benefit_offline"),
                          ("mechanism", "run_benefit_probe"), ("timing", "run_benefit_probe")):
        info = json.loads((source / stage / "provenance.json").read_text())
        check_provenance(info, manifest, f"scripts/{runner}.py")
        if stage != "cost" and info["costs_sha256"] != file_hash(source / "cost/costs.json"):
            raise ValueError("Changed measured cost binding")
    costs = audit_costs(source)
    offline = read_rows(source / "offline/rows.jsonl")
    if offline != decisions(ROOT, costs):
        raise ValueError("Changed project-excluded offline predictions")
    tasks, events = inputs(ROOT, subset=True)
    stages = {}
    for stage in ("mechanism", "timing"):
        info = json.loads((source / stage / "provenance.json").read_text())
        rows = read_rows(source / stage / "rows.jsonl")
        summary = audit(rows, tasks, 1 if stage == "mechanism" else 2, instrumented=stage == "mechanism")
        if (len(rows) != info["rows"] or info["inputs_sha256"] != input_digest(tasks, events)
                or info["methods"] != list(METHODS) or info["samples"] != calibration(ROOT)
                or summary != json.loads((source / stage / "summary.json").read_text())):
            raise ValueError("Changed development rows, calibration or summary")
        stages[stage] = (info, rows, summary)
    mechanism, timing = stages["mechanism"], stages["timing"]
    if mechanism[0]["timing_gate"] != timing_gate(mechanism[1]):
        raise ValueError("Changed mechanism gate")
    audit_calls([{**r, "method": "Splice-" + r["method"]} if r["method"] in METHODS[-3:] else r for r in mechanism[1]], tasks)
    audit_decisions(mechanism[1], costs, tasks)
    expected = {(r["request_id"], r["method"]): r["counters"] for r in mechanism[1]}
    for r in timing[1]:
        if any(v != r["counters"][k] for k, v in expected[r["request_id"], r["method"]].items() if k != "alignment_seconds"):
            raise ValueError("Timing changed the checked decoding path")
    all_times = timing[2]["all"]
    diagnostic = dict(costs=costs, offline=json.loads((source / "offline/summary.json").read_text()),
                      mechanism=mechanism[2], timing=timing[2], paired_paths=len(timing[1]),
                      comparisons={m: {base: all_times[m]["seconds"] / all_times[base]["seconds"] - 1
                                      for base in ("Suffix-R", "Guided-Edit", "Splice-Edit", "Splice-TailOT")}
                                   for m in METHODS[-3:]})
    lines = ["# Draft内部拼接：预计收益调用OT与动态长度", "",
             "2026-10-19。保留内部Draft/历史拼接；本轮不使用完整矩阵可靠性。14个旧状态用于经验校准，每次排除当前整个项目，最近3个其他项目预测推进量。全部任务答案此前已观察，是开发检查，不是独立holdout。", "",
             "动作是原Draft、Edit拼接、TailOT拼接；长度16/24/32。先扣除实测OT成本、比较预计推进/周期成本，超过最佳廉价动作2%才调用OT。单候选不调用。Gate-OT固定32；Budget-Edit禁止OT；Budget-OT同时路由与选长度。详见OT_BENEFIT_PROTOCOL.md。", "",
             "## 成本与预测", "", "以下为独立核函数微基准的7次中位数，单位毫秒，不是端到端延迟。", "",
             "| 上下文 | Draft（pending16） | Verify15 | Verify16 | Verify24 | Verify32 |", "|---|---:|---:|---:|---:|---:|"]
    for p in costs["profiles"]:
        lines.append(f"| {p['context']} | {p['draft']*1000:.3f} | " + " | ".join(f"{p['verify'][str(n)]*1000:.3f}" for n in (15, 16, 24, 32)) + " |")
    o = diagnostic["offline"]
    lines += ["", f"共享shortlist/policy约{costs['lookup']*1000:.3f}ms；每候选TailOT约{costs['ot_candidate']*1000:.3f}ms。模型加载与这轮校准不在每请求计时内，在线检索/策略/Draft/Target等均在。", "",
              f"14个项目排除的旧状态：OT调用{o['ot_calls']}，缩短拼接{o['short']}，保留原Draft{o['draft']}。预测接受总和{o['predicted_accepted']:.2f}、实际{o['actual_accepted']}，平均绝对误差{o['mean_absolute_error']:.2f} token。这些是共同状态的预测检查，不能当在线净增益。", "",
              "## 严格机制和端到端计时", "", f"56/56机制与112/112交错计时输出token、Target输入和合法快照匹配AR；{len(timing[1])}/{len(timing[1])}计时计数路径与机制一致。", "",
              "| 方法 | 总轮 | 实际Draft前向 | 历史尾部接受 | 预计收益决策 | OT调用/候选 | 缩短提议 | 总计时秒 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        c = mechanism[2]["all"][m]["counters"]
        lines.append(f"| {m} | {c['rounds']} | {c['draft_rounds'] if m == 'Suffix-R' else c['draft_forward_rounds']} | {c['splice_history_accepted_tokens']} | {c['benefit_decisions']} | {c['benefit_ot_calls']}/{c['benefit_ot_candidates']} | {c['benefit_short']} | {all_times[m]['seconds']:.6f} |")
    lines += ["", "历史尾部接受量与保留Draft前缀分开统计；不是相对其他路径的净增量。下表为各请求两次端到端计时和，秒。", "",
              "| 请求 | " + " | ".join(METHODS) + " |", "|---|" + "---:|" * len(METHODS)]
    for task in tasks:
        lines.append("| " + task.request.id + " | " + " | ".join(f"{sum(r['latency_seconds'] for r in timing[1] if r['request_id'] == task.request.id and r['method'] == m):.6f}" for m in METHODS) + " |")
    lines += ["", "## 判断边界", ""]
    for m, ratios in diagnostic["comparisons"].items():
        lines.append(f"- {m}相对Suffix {ratios['Suffix-R']:+.2%}，Guided-Edit {ratios['Guided-Edit']:+.2%}，Splice-Edit {ratios['Splice-Edit']:+.2%}，固定TailOT {ratios['Splice-TailOT']:+.2%}（负值是延迟下降）。")
    lines += ["- 校准只有14个状态；排除当前项目后，预测可能偏离实际。当前模型只看廉价候选统计，没有足够证据证明能识别保留Draft前缀的错误。", "- 局部rate近似未显式建模后续精确suffix命中、Draft pending状态、不同span实际长度；缩短块可能新增未来轮数。不能将预计收益当成加速保证。", "- 不在本批上继续更换邻居数/阈值挑最优；默认suffix与独立infra面试版保持原样。", "",
              f"冻结执行提交`{manifest['commit']}`，package SHA `{mechanism[0]['source']['sha256']}`。", "", "```bash", "python -m scripts.write_benefit_report", "```", ""]
    return "\n".join(lines), diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/ot-benefit")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/OT_BENEFIT_RESULTS.md")
    args = parser.parse_args()
    report, diagnostic = build(args.source)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    preserve_diagnostic(args.source / "diagnostic.json", diagnostic)


if __name__ == "__main__":
    main()
