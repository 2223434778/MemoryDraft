"""Frozen task-held-out compute prediction and two-action oracle audit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from memorydraft.evaluation.asd_budget import load_tasks, score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from scripts.diagnose_transport_online import write
from scripts.run_asd_compute_probe import FEATURES, ROOT, TASK_IDS, WINDOW, selected_states, sources


def require(ok, message):
    if not ok:
        raise ValueError(message)


def strict_future_ok(row, source_block):
    for i, r in enumerate(row["rounds"][source_block + 1:], source_block + 1):
        expected = r["strict_accepted"]
        if row["finish_reason"] == "eos" and i == len(row["rounds"]) - 1:
            expected = min(expected, len(row["token_ids"]) - r["generated_before"])
        if r["accepted"] != expected or r["regret"] != 0:
            return False
    return row["strict_future_regret"] == 0


def predict(entries):
    """No same-question labels or preprocessing; fixed ridge, including an intercept."""
    predictions = {}
    for q in sorted({e["request_id"] for e in entries}):
        train = [e for e in entries if e["request_id"] != q and e["complete"]]
        heldout = [e for e in entries if e["request_id"] == q]
        if len(train) < 3:
            for e in heldout:
                predictions[q, e["position"]] = dict(action=0, delta=None, fallback=True)
            continue
        x = np.array([e["features"] for e in train], dtype=float)
        y = np.array([e["rounds"][1] - e["rounds"][0] for e in train], dtype=float)
        mean, scale = x.mean(0), x.std(0)
        scale = np.where(scale > 1e-12, scale, 1.)
        standardized = (x - mean) / scale
        target_mean = y.mean()
        weights = np.linalg.solve(standardized.T @ standardized + np.eye(len(FEATURES)),
                                  standardized.T @ (y - target_mean))
        for e in heldout:
            delta = float(((np.array(e["features"]) - mean) / scale) @ weights + target_mean)
            predictions[q, e["position"]] = dict(action=0 if delta >= 0 else 1, delta=delta, fallback=False)
    return predictions


def summarize(entries):
    predictions = predict(entries)
    eligible = [e for e in entries if e["complete"] and all(e["correct"])]
    selectors = dict(Utility=lambda e: 0, Strict=lambda e: 1,
                     Run=lambda e: int(e["regret"] / e["run"] > .25),
                     Ridge=lambda e: predictions[e["request_id"], e["position"]]["action"],
                     Oracle=lambda e: int(e["rounds"][1] < e["rounds"][0]))
    totals = {}
    for name, select in selectors.items():
        totals[name] = dict(rounds=sum(e["rounds"][select(e)] for e in eligible),
                            seconds=sum(e["seconds"][select(e)] for e in eligible),
                            natural_correct=sum(e["correct"][select(e)] for e in entries),
                            reject_choices=sum(select(e) for e in entries),
                            natural_truncated=sum(e["finish"][select(e)] != "eos" for e in entries))
    best_fixed = min(totals["Utility"]["rounds"], totals["Strict"]["rounds"])
    oracle_gain = 1 - totals["Oracle"]["rounds"] / best_fixed if best_fixed else 0.
    enough = len(eligible) >= 8 and len({e["request_id"] for e in eligible}) >= 3
    oracle_pass = enough and oracle_gain >= .05
    predictor_pass = (oracle_pass and totals["Ridge"]["rounds"] < min(best_fixed, totals["Run"]["rounds"])
                      and totals["Ridge"]["natural_correct"] >= totals["Utility"]["natural_correct"]
                      and not totals["Ridge"]["natural_truncated"])
    labeled = [e for e in entries if e["complete"]]
    errors = [abs(predictions[e["request_id"], e["position"]]["delta"] - (e["rounds"][1] - e["rounds"][0]))
              for e in labeled if predictions[e["request_id"], e["position"]]["delta"] is not None]
    return dict(states=len(entries), eligible=len(eligible), eligible_questions=len({e["request_id"] for e in eligible}),
                totals=totals, oracle_gain_over_best_fixed=oracle_gain, enough=enough,
                oracle_pass=oracle_pass, predictor_pass=predictor_pass,
                mae_round_difference=float(np.mean(errors)) if errors else None,
                fallback_states=sum(v["fallback"] for v in predictions.values()),
                predictions=[dict(request_id=q, position=p, **v) for (q, p), v in predictions.items()])


def analyze(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "ASD_COMPUTE_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    verification = json.loads((directory / "execution-verification.json").read_text(encoding="utf-8"))
    require(verification["status"] == "complete" and verification["commit"] == manifest["commit"], "Execution incomplete")
    require(verification["files_verified_before"] == verification["files_verified_after"] == len(manifest["files"]),
            "Execution source coverage")
    refs = sources()
    records = {r["id"]: r for r in load_tasks(ROOT / "data/research/asd-budget/records.json")}
    entries, count = [], 0
    for part, questions in (("math-a", TASK_IDS[:2]), ("mixed-b", TASK_IDS[2:])):
        info = json.loads((directory / part / "provenance.json").read_text(encoding="utf-8"))
        expected = {q: selected_states(refs[q]) for q in questions}
        require(info["status"] == "complete" and info["original_unchanged"] and info["selections"] == expected,
                "Frozen selection or worker status")
        require(info["window"] == WINDOW, "Fixed horizon")
        for field, name in (("runner_sha256", "scripts/run_asd_compute_probe.py"),
                            ("protocol_sha256", "docs/ASD_COMPUTE_PROTOCOL.md")):
            require(info[field] == manifest["files"][name], "Execution hash")
        # These previously published bytes are stable across Windows/Linux.
        for name, digest in manifest["files"].items():
            if name.startswith("results/asd-confidence/"):
                require(file_hash(ROOT / name) == digest, "Old source evidence changed")
        baselines = read_rows(directory / part / "baselines.jsonl")
        require(len(baselines) == info["baseline_rows"] == len(questions), "Baseline coverage")
        for row, q in zip(baselines, questions):
            ref = refs[q]
            require(row["request_id"] == q and row["token_ids"] == ref["token_ids"] and row["asd"] == ref["asd"],
                    "Baseline reproduction")
            require((row["prompt_sha256"], row["snapshot_id"]) == (ref["prompt_sha256"], ref["snapshot_id"]), "Baseline input")
        rows = read_rows(directory / part / "branches.jsonl")
        require(len(rows) == info["branch_rows"] == 48, "Branch coverage")
        indexed = {(r["request_id"], r["state"]["position"], r["horizon"], r["arm"], r["repeat"]): r for r in rows}
        keys = {(q, s["position"], h, a, rep) for q, states in expected.items() for s in states
                for h, reps in (("natural", [0]), ("prefix", [0, 1])) for rep in reps for a in ("accept", "reject")}
        require(len(indexed) == len(rows) and set(indexed) == keys, "Duplicate/missing branches")
        count += len(rows) + len(baselines)
        for q, states in expected.items():
            for s in states:
                ref, b, boundary = refs[q], s["block"], s["boundary"]
                event = ref["events"][b]
                chosen = []
                for arm in ("accept", "reject"):
                    group = [indexed[q, s["position"], h, arm, rep]
                             for h, rep in (("natural", 0), ("prefix", 0), ("prefix", 1))]
                    for row in group:
                        a = event["accepted"] if arm == "accept" else s["index"]
                        charge = event["regret"] if arm == "accept" else 0.
                        require(row["state"] == s and row["action"]["accepted"] == a, "Frozen action")
                        require(row["action"]["strict_accepted"] == event["strict_accepted"]
                                and row["action"]["source_block"] == b and row["action"]["proposed"] == len(event["proposal_ids"]),
                                "Source action metadata")
                        require(abs(row["action"]["charge"] - charge) < 1e-8
                                and abs(row["action"]["spent"] - s["spent_before_block"] - charge) < 1e-8
                                and abs(row["action"]["original_spent"] - s["spent_before_block"]) < 1e-8,
                                "Preserved action ledger")
                        tokens = row["token_ids"]
                        require(tokens[:boundary] == ref["token_ids"][:boundary]
                                and tokens[boundary:boundary + a] == event["proposal_ids"][:a], "Common and cached prefix")
                        if arm == "reject":
                            require(tokens[s["position"]] == s["recovery"], "Recovery token")
                        limit = 2048 if row["horizon"] == "natural" else boundary + WINDOW
                        require(row["max_new_tokens"] == limit and row["new_tokens"] == len(tokens) - boundary
                                and boundary < len(tokens) <= limit, "Horizon bounds")
                        eos = [i for i, t in enumerate(tokens) if t in info["stop_token_ids"]]
                        require(not eos or eos == [len(tokens) - 1], "Output after EOS")
                        require(row["finish_reason"] == ("eos" if eos else "length"), "EOS reason")
                        require(row["quality"] == score(records[q], row["text"]), "Unchanged scoring")
                        rds = row["rounds"]
                        require(row["future_rounds"] == len(rds) - b - 1 and rds[b]["accepted"] == a, "Future round count")
                        require(all(r["accepted"] == ref["rounds"][i]["accepted"] for i, r in enumerate(rds[:b])), "Pre-fork actions")
                        require(strict_future_ok(row, b), "Strict future")
                        require(math.isfinite(row["post_decision_seconds"]) and row["post_decision_seconds"] > 0, "Compute timing")
                    natural, p0, p1 = group
                    require(p0["token_ids"] == p1["token_ids"] and p0["action"] == p1["action"]
                            and p0["future_rounds"] == p1["future_rounds"], "Repeat trajectories")
                    require(p0["token_ids"] == natural["token_ids"][:len(p0["token_ids"])], "Strict capped prefix")
                    chosen.append((natural, p0, p1))
                entries.append(dict(request_id=q, position=s["position"], block=b,
                                    features=[s[name] for name in FEATURES], regret=s["regret"], run=s["run"],
                                    local_gain=s["local_gain"], complete=all(g[1]["new_tokens"] == WINDOW for g in chosen),
                                    rounds=[g[1]["future_rounds"] for g in chosen],
                                    seconds=[sum(r["post_decision_seconds"] for r in g[1:]) / 2 for g in chosen],
                                    seconds_repeats=[[r["post_decision_seconds"] for r in g[1:]] for g in chosen],
                                    correct=[g[0]["quality"]["correct"] for g in chosen],
                                    finish=[g[0]["finish_reason"] for g in chosen],
                                    natural_tokens=[len(g[0]["token_ids"]) for g in chosen]))
    return dict(commit=manifest["commit"], generations=count,
                original_baseline_correct={q: refs[q]["quality"]["correct"] for q in TASK_IDS},
                entries=entries, summary=summarize(entries))


def render(result):
    s = result["summary"]
    lines = ["# 实际计算收益可行性检查", "", f"4道旧开发题、16个不同源块，{result['generations']}次生成。"
             "原Utility动作A与严格当前动作B在同一验证块、保留实际KV后分叉，之后都严格续写。", "",
             f"主要共同质量/等128token集：{s['eligible']}状态，覆盖{s['eligible_questions']}题。"
             "这是事后可比子集，质量信息线上不可得；所有状态质量仍在下表保留。", "",
             "|题/位置|本轮多推进|未来轮A/B|自然最终正确A/B|自然长度A/B|主要集|",
             "|---|---:|---|---|---|---|"]
    for e in result["entries"]:
        pair = lambda x: "/".join(str(v) for v in x)
        lines.append(f"|{e['request_id']}/{e['position']}|{e['local_gain']}|{pair(e['rounds'])}|"
                     f"{pair(e['correct'])}|{pair(e['natural_tokens'])}|{e['complete'] and all(e['correct'])}|")
    lines.extend(["", "|选择器|主要集未来轮|主要集局部秒|全部16状态所选自然答案正确|所选自然截断|",
                  "|---|---:|---:|---:|---:|"])
    for name, t in s["totals"].items():
        lines.append(f"|{name}|{t['rounds']}|{t['seconds']:.6f}|{t['natural_correct']}/16|{t['natural_truncated']}|")
    lines.extend(["", f"有限动作oracle相对最好固定动作的未来轮节省：{s['oracle_gain_over_best_fixed']*100:.2f}%。"
                  f"预定样本覆盖达标={s['enough']}；oracle >=5%门槛通过={s['oracle_pass']}；"
                  f"固定特征留题预测器门槛通过={s['predictor_pass']}。",
                  f"留题未来轮差MAE={s['mae_round_difference']}，fallback状态={s['fallback_states']}。", "",
                  f"原完整Utility基线正确{sum(result['original_baseline_correct'].values())}/4；"
                  f"本轮分叉后严格续写的32条自然分支正确{sum(sum(e['correct']) for e in result['entries'])}/32。"
                  "表中Utility仅表示当前动作A，之后都严格续写，不能解释为原完整Utility已经纠错。"
                  "Run列是首例外的局部二选一规则，不是完整ASD-Run在线基线。", "",
                  "预测使用当前动作多推进、原始regret差、Run、此前4轮平均推进；不使用置信度。"
                  "固定岭回归alpha=1，各题的训练/标准化只用其他题；训练仅按等128筛选，不用正确答案筛选。"
                  "这是4道已观察题的可行性诊断，没有调参或独立新题验证。", "",
                  "计时从已付共同Target验证、当前选择器和源核查之后开始，计入缓存提交及严格后续，"
                  "排除共同prefill/重放和预测开销。Oracle按轮数选择，秒数不是按最小时延再次挑选。"
                  "两次局部测量不支持线上端到端加速或显著性结论。", "",
                  "各状态是独立二选一，不能将oracle收益直接相加解释为完整近似策略收益；"
                  "自然长度、EOS及最终答案须同时评估，最终答案正确也不保证整段推理一致。", "",
                  ("阶段决定：主要样本覆盖足够，但oracle未达预定5%门槛。停止继续这条128token、二选一的未来计算预测路线；"
                   "不扫描岭回归参数、不增加复杂预测头、不扩大同类开发题。现有Utility作为简单实验选项保留。"
                   if s["enough"] and not s["oracle_pass"] else
                   "阶段决定：证据边界依照上述冻结覆盖/收益/预测器门槛，不据开发题直接部署预测器。"), "",
                  f"冻结执行commit `{result['commit']}`。原始证据results/asd-compute；协议ASD_COMPUTE_PROTOCOL.md。",
                  "重建：`python -m scripts.write_asd_compute_report`。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/asd-compute")
    args = parser.parse_args()
    result = analyze(args.source)
    write(args.source / "analysis.json", result)
    (ROOT / "docs/ASD_COMPUTE_RESULTS.md").write_text(render(result), encoding="utf-8", newline="\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
