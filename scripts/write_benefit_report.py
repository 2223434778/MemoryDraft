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


def equivalent(a, b):
    # Python 3.12+ changed float sum(). Permit only rounding noise in derived
    # statistics; discrete routes, lengths, token IDs and evidence bytes stay exact.
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if type(a) is float and type(b) is float:
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return type(a) is type(b) and a == b


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
            if not equivalent(d, dict(generated=generated, **expected)) or row["project_id"] in d["projects"]:
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
        if info["costs_sha256"] != file_hash(source / "cost/costs.json"):
            raise ValueError("Changed measured cost binding")
    costs = audit_costs(source)
    offline = read_rows(source / "offline/rows.jsonl")
    if not equivalent(offline, decisions(ROOT, costs)):
        raise ValueError("Changed project-excluded offline predictions")
    offline_summary = dict(states=len(offline), ot_calls=sum(r["choice"]["route"] == "ot" for r in offline),
                          short=sum(r["choice"]["length"] < 32 and r["choice"]["route"] != "draft" for r in offline),
                          draft=sum(r["choice"]["route"] == "draft" for r in offline),
                          actual_accepted=sum(r["actual_accepted"] for r in offline),
                          predicted_accepted=sum(r["predicted_accepted"] for r in offline),
                          mean_absolute_error=sum(abs(r["actual_accepted"] - r["predicted_accepted"]) for r in offline) / len(offline))
    stored_offline = json.loads((source / "offline/summary.json").read_text())
    if not equivalent(offline_summary, stored_offline):
        raise ValueError("Changed offline prediction summary")
    offline_summary = stored_offline
    tasks, events = inputs(ROOT, subset=True)
    stages = {}
    for stage in ("mechanism", "timing"):
        info = json.loads((source / stage / "provenance.json").read_text())
        rows = read_rows(source / stage / "rows.jsonl")
        summary = audit(rows, tasks, 1 if stage == "mechanism" else 2, instrumented=stage == "mechanism")
        stored = json.loads((source / stage / "summary.json").read_text())
        if (len(rows) != info["rows"] or info["inputs_sha256"] != input_digest(tasks, events)
                or info["methods"] != list(METHODS) or info["samples"] != calibration(ROOT)
                or not equivalent(summary, stored)):
            raise ValueError("Changed development rows, calibration or summary")
        stages[stage] = (info, rows, stored)
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
    diagnostic = dict(costs=costs, offline=offline_summary,
                      mechanism=mechanism[2], timing=timing[2], paired_paths=len(timing[1]),
                      comparisons={m: {base: all_times[m]["seconds"] / all_times[base]["seconds"] - 1
                                      for base in ("Suffix-R", "Guided-Edit", "Splice-Edit", "Splice-TailOT")}
                                   for m in METHODS[-3:]})
    lines = ["# Draft内部拼接：预计收益调用OT与动态长度", "",
             "2026-10-19。保留内部Draft/历史拼接；本轮不使用完整矩阵可靠性。14个旧状态用于经验校准，每次排除当前整个项目，最近3个其他项目预测推进量。全部任务答案此前已观察，是开发检查，不是独立holdout。", "",
             "动作是原Draft、Edit拼接、TailOT拼接；长度16/24/32。先扣除实测OT成本、比较预计推进/周期成本，超过最佳廉价动作2%才调用OT。单候选不调用。Gate-OT固定32；Budget-Edit禁止OT；Budget-OT同时路由与选长度。详见OT_BENEFIT_PROTOCOL.md。", "",
             "首版250ab31的56次逐请求严格检查均通过，但最终汇总将拼接归因检查错误套到Guided-Edit而失败，未进入计时。仅修复审计范围后，以新冻结版本重跑正式全部阶段；失败现场保存在failed-attempt1，不混入下面成功计时。", "",
             "## 成本与预测", "", "以下为独立核函数微基准的7次中位数，单位毫秒，不是端到端延迟。", "",
             "| 上下文 | Draft（pending16） | Verify15 | Verify16 | Verify24 | Verify32 |", "|---|---:|---:|---:|---:|---:|"]
    for p in costs["profiles"]:
        lines.append(f"| {p['context']} | {p['draft']*1000:.3f} | " + " | ".join(f"{p['verify'][str(n)]*1000:.3f}" for n in (15, 16, 24, 32)) + " |")
    o = diagnostic["offline"]
    lines += ["", f"共享shortlist/policy约{costs['lookup']*1000:.3f}ms；每候选TailOT约{costs['ot_candidate']*1000:.3f}ms。模型加载与这轮校准不在每请求计时内，在线检索/策略/Draft/Target等均在。", "",
              f"对14个旧状态逐一排除其所属项目后：OT调用{o['ot_calls']}，缩短拼接{o['short']}，保留原Draft{o['draft']}。预测接受总和{o['predicted_accepted']:.2f}、实际{o['actual_accepted']}，平均绝对误差{o['mean_absolute_error']:.2f} token。这些是共同状态的预测检查，不能当在线净增益。", "",
              "| 旧请求/位置 | 决策 | 预计接受 | 实际接受 | Edit | TailOT |", "|---|---|---:|---:|---:|---:|"]
    for r in offline:
        lines.append(f"| {r['request_id']}/{r['generated']} | {r['choice']['route']}/{r['choice']['length']} | {r['predicted_accepted']:.2f} | {r['actual_accepted']} | {r['edit_accepted']} | {r['ot_accepted']} |")
    lines += ["",
              "## 严格机制和端到端计时", "", f"56/56机制与112/112交错计时输出token、Target输入和合法快照匹配AR；{len(timing[1])}/{len(timing[1])}计时计数路径与机制一致。", "",
              "| 方法 | 总轮 | 实际Draft前向 | 历史尾部接受 | 预计收益决策 | OT调用/候选 | 缩短提议 | 总计时秒 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        c = mechanism[2]["all"][m]["counters"]
        ot_calls = c["alignment_calls"] if m == "Splice-TailOT" else c["benefit_ot_calls"]
        ot_candidates = c["alignment_candidates"] if m == "Splice-TailOT" else c["benefit_ot_candidates"]
        lines.append(f"| {m} | {c['rounds']} | {c['draft_rounds'] if m == 'Suffix-R' else c['draft_forward_rounds']} | {c['splice_history_accepted_tokens']} | {c['benefit_decisions']} | {ot_calls}/{ot_candidates} | {c['benefit_short']} | {all_times[m]['seconds']:.6f} |")
    lines += ["", "历史尾部接受量与保留Draft前缀分开统计；不是相对其他路径的净增量。下表为各请求两次端到端计时和，秒。", "",
              "| 请求 | " + " | ".join(METHODS) + " |", "|---|" + "---:|" * len(METHODS)]
    for task in tasks:
        lines.append("| " + task.request.id + " | " + " | ".join(f"{sum(r['latency_seconds'] for r in timing[1] if r['request_id'] == task.request.id and r['method'] == m):.6f}" for m in METHODS) + " |")
    lines += ["", "## 判断边界", ""]
    for m, ratios in diagnostic["comparisons"].items():
        lines.append(f"- {m}相对Suffix {ratios['Suffix-R']:+.2%}，Guided-Edit {ratios['Guided-Edit']:+.2%}，Splice-Edit {ratios['Splice-Edit']:+.2%}，固定TailOT {ratios['Splice-TailOT']:+.2%}（负值是延迟下降）。")
    lines += ["- 本轮未显示新收益策略的净优势。OT候选求解31→7，Target总验证轮65→67，实际Draft前向仍31。按微基准量级，少24次OT约省54ms，而两轮Target验证约120–180ms；这是成本量级解释，不是独立在线分解实测。", "- 排除当前项目后的预测在shipment-contract/22预计接受30.56、实际6，选择OT而Edit可以接受32；在payment-runbook/15选Edit接受7，错过TailOT的32。便宜特征未准确识别OT的条件增益。", "- 动态长度代码已实现，但本批始终选择32，Gate-OT与Budget-OT解码路径相同；两者极小计时差不能归因为长度优化。预测偏乐观，使较长块的预计推进增量始终覆盖额外验证成本；未证实动态长度对其他场景无效。", "- 校准只有14个状态；排除当前项目后，预测可能偏离实际。当前模型只看廉价候选统计，没有足够证据证明能识别保留Draft前缀的错误。", "- 局部rate近似未显式建模后续精确suffix命中、Draft pending状态、不同span实际长度；缩短块可能新增未来轮数。不能将预计收益当成加速保证。", "- 下一步应先检查有独立校准数据的廉价Draft前缀风险/接受生存预测，达到区分度后再比较收益路由；当前不建议继续扩大这套14状态邻居策略的场景或阈值搜索。内部拼接与其SQL局部收益保留。", "- 不在本批上继续更换邻居数/阈值挑最优；默认suffix与独立infra面试版保持原样。", "",
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
