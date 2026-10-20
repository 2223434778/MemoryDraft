"""Audit and summarize the frozen Utility causal development diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from memorydraft.evaluation.asd_budget import load_tasks, score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from scripts.diagnose_transport_online import write
from scripts.run_asd_utility_causal_probe import CONFIG, ROOT, TASK_IDS, selected_states, sources, validate_repair


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_policy(row, state, source, stops):
    """Scalar recomputation from published mismatches/gaps, independent of Torch selector."""
    spent, relaxed, shortened, foregone = 0., 0, 0, 0
    require(len(row["events"]) == len(row["rounds"]) == row["asd"]["blocks"], "Policy block count")
    for block, (e, rd) in enumerate(zip(row["events"], row["rounds"])):
        k = len(e["mismatch"])
        require(k and all(len(e[name]) == k for name in ("proposal_ids", "target_ids", "regrets")), "Event alignment")
        mismatch = [a != b for a, b in zip(e["proposal_ids"], e["target_ids"])]
        require(mismatch == e["mismatch"], "Mismatch identity")
        require(e["proposal_eos"] == [i + 1 for i, t in enumerate(e["proposal_ids"]) if t in stops], "Proposal EOS")
        require(all(math.isfinite(r) and r >= 0 for r in e["regrets"]), "Finite regret")
        require(all(m or abs(r) < 1e-8 for m, r in zip(mismatch, e["regrets"])), "Matching token regret")
        costs = [r if m else 0. for r, m in zip(e["regrets"], mismatch)]
        cumulative = [0.]
        for cost in costs:
            cumulative.append(cumulative[-1] + cost)
        strict = next((i for i, m in enumerate(mismatch) if m), k)
        eos = min(e["proposal_eos"], default=k + 1)
        ceiling, count = 0, 0
        for i, m in enumerate(mismatch):
            count += m
            if ((m and costs[i] / (k - i) > CONFIG.gate) or count > CONFIG.cap
                    or cumulative[i + 1] > max(0., CONFIG.budget - spent)):
                break
            ceiling = i + 1
        ceiling = min(ceiling, eos)
        baseline = min(strict, eos)
        choices = list(range(baseline, ceiling + 1))
        best = max(choices, key=lambda n: (min(n + 1, eos) - min(baseline + 1, eos)
                                          - CONFIG.regret_price * cumulative[n], -cumulative[n], n))
        if state is not None and block == state["block"]:
            require(best == source["events"][block]["accepted"], "Original repair choice")
            best = state["index"]
            require(e.get("repair") == row["intervention"], "Repair event metadata")
        require(e["accepted"] == rd["accepted"] == best and e["feasible_accepted"] == ceiling, "Utility selection")
        require(e["strict_accepted"] == strict, "Strict length")
        charge = cumulative[best]
        spent += charge
        relaxed += sum(mismatch[:best])
        shortened += best < ceiling
        foregone += ceiling - best
        require(abs(e["regret"] - charge) < 1e-8 and abs(e["spent"] - spent) < 1e-8, "Persistent ledger")
        require(spent <= CONFIG.budget + 1e-9, "Request budget")
        p = rd["generated_before"]
        require(e["generated_before"] == p and rd["advance"] == best + 1, "Round progress")
        require(row["token_ids"][p:p + best] == e["proposal_ids"][:best], "Committed proposal prefix")
    require(abs(row["asd"]["spent"] - spent) < 1e-8 and row["asd"]["relaxed_tokens"] == relaxed,
            "Summary ledger")
    require(row["asd"]["shortened_blocks"] == shortened and row["asd"]["foregone_proposal_tokens"] == foregone,
            "Summary shortening")
    return len(row["events"])


def analyze(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "ASD_UTILITY_CAUSAL_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    execution = json.loads((directory / "execution-verification.json").read_text(encoding="utf-8"))
    require(execution["status"] == "complete" and execution["commit"] == manifest["commit"], "Execution failed")
    require(execution["files_verified_before"] == execution["files_verified_after"] == len(manifest["files"]),
            "Execution source coverage")
    references = sources()
    records = {r["id"]: r for r in load_tasks(ROOT / "data/research/asd-budget/records.json")}
    result = dict(commit=manifest["commit"], tasks={}, policy_rows=0, branch_rows=0, policy_blocks=0)
    for q, part in zip(TASK_IDS, ("gsm", "math")):
        source = references[q]
        states = selected_states(source)
        info = json.loads((directory / part / "provenance.json").read_text(encoding="utf-8"))
        require(info["status"] == "complete" and info["original_unchanged"], "Worker failed or old results changed")
        require(info["task_id"] == q and info["states"] == states and info["config"] == asdict(CONFIG), "Frozen selection/config")
        for field, name in (("runner_sha256", "scripts/run_asd_utility_causal_probe.py"),
                            ("protocol_sha256", "docs/ASD_UTILITY_CAUSAL_PROTOCOL.md"),
                            ("data_sha256", "data/research/asd-budget/records.json")):
            require(info[field] == manifest["files"][name], "Frozen source hash")
        for name, digest in manifest["files"].items():
            if name.startswith("results/asd-confidence/"):
                require(file_hash(ROOT / name) == digest, "Previous confidence evidence changed")
        policies = read_rows(directory / part / "policies.jsonl")
        branches = read_rows(directory / part / "branches.jsonl")
        require(len(policies) == info["policy_rows"] == 1 + len(states), "Policy coverage")
        require(len(branches) == info["branch_rows"] == 4 * len(states), "Branch coverage")
        baseline = policies[0]
        require(baseline["state"] is None and baseline["token_ids"] == source["token_ids"]
                and baseline["asd"] == source["asd"], "Baseline reproduction")
        for row in [*policies, *branches]:
            require(row["request_id"] == q and row["quality"] == score(records[q], row["text"]), "Task/quality")
            require(len(row["token_ids"]) <= 2048 and row["token_ids"], "Output bounds")
            eos_positions = [i for i, t in enumerate(row["token_ids"]) if t in info["stop_token_ids"]]
            require(not eos_positions or eos_positions == [len(row["token_ids"]) - 1], "Output beyond EOS")
            require(row["finish_reason"] == ("eos" if eos_positions else "length"), "Finish reason")
        for row, state in zip(policies, [None, *states]):
            require(row["state"] == state, "Policy state order")
            require((row["prompt_sha256"], row["snapshot_id"]) == (source["prompt_sha256"], source["snapshot_id"]),
                    "Fixed Target input")
            if state is not None:
                validate_repair(row["token_ids"], source["token_ids"], state)
                require(abs(row["intervention"]["spent"] - state["spent_before_token"]) < 1e-8, "Repair budget reset")
            result["policy_blocks"] += audit_policy(row, state, source, info["stop_token_ids"])
        indexed = {(r["state"]["position"], r["horizon"], r["arm"]): r for r in branches}
        require(len(indexed) == len(branches) and set(indexed) == {
            (s["position"], h, a) for s in states for h in ("natural", "prefix") for a in ("proposal", "recovery")},
                "Duplicate/missing branch")
        entries = []
        for state, repaired in zip(states, policies[1:]):
            p = state["position"]
            for horizon in ("natural", "prefix"):
                paired = [indexed[p, horizon, a] for a in ("proposal", "recovery")]
                require(paired[0]["common_prefix_sha256"] == paired[1]["common_prefix_sha256"], "Branch prefix hash")
                for row, arm in zip(paired, ("proposal", "recovery")):
                    require(row["state"] == state and row["forced_token"] == state["candidate" if arm == "proposal" else "recovery"],
                            "Forced token")
                    require(row["token_ids"][:p] == source["token_ids"][:p]
                            and row["token_ids"][p] == row["forced_token"], "Strict common prefix")
                    limit = 2048 - p - 1 if horizon == "natural" else min(128, 2048 - p - 1)
                    n = len(row["token_ids"]) - p - 1
                    require(row["continuation_limit"] == limit and row["continuation_tokens"] == n and n <= limit,
                            "Continuation bounds")
                    require(row["continuation_rounds"] == len(row["rounds"]), "Continuation rounds")
                    require(row["replay_argmax"] == state["recovery"]
                            and abs(row["replay_regret"] - state["regret"]) <= 1e-3, "Same-state Target check")
                    require(math.isfinite(row["target_entropy"]) and row["target_entropy"] >= 0
                            and row["recompute_seconds"] > 0, "Branch diagnostics")
            natural = [indexed[p, "natural", a] for a in ("proposal", "recovery")]
            prefix = [indexed[p, "prefix", a] for a in ("proposal", "recovery")]
            require(all(r["token_ids"] == natural[i]["token_ids"][:len(r["token_ids"])] for i, r in enumerate(prefix)),
                    "Strict prefix differs before cap")
            entries.append(dict(state=state, repaired_correct=repaired["quality"]["correct"],
                                repaired_tokens=len(repaired["token_ids"]), repaired_rounds=len(repaired["rounds"]),
                                repaired_spent=repaired["asd"]["spent"], repaired_finish=repaired["finish_reason"],
                                natural_correct=[r["quality"]["correct"] for r in natural],
                                natural_tokens=[r["continuation_tokens"] for r in natural],
                                natural_rounds=[r["continuation_rounds"] for r in natural],
                                natural_finish=[r["finish_reason"] for r in natural],
                                prefix_tokens=[r["continuation_tokens"] for r in prefix],
                                prefix_rounds=[r["continuation_rounds"] for r in prefix],
                                equal_128=all(r["continuation_tokens"] == 128 for r in prefix),
                                target_entropy=natural[0]["target_entropy"]))
        result["tasks"][q] = dict(baseline_correct=baseline["quality"]["correct"], baseline_tokens=len(baseline["token_ids"]),
                                  baseline_rounds=len(baseline["rounds"]), baseline_spent=baseline["asd"]["spent"], entries=entries)
        result["policy_rows"] += len(policies)
        result["branch_rows"] += len(branches)
    entries = [e for t in result["tasks"].values() for e in t["entries"]]
    equal = [e for e in entries if e["equal_128"]]
    result["summary"] = dict(
        strict_proposal_correct=sum(e["natural_correct"][0] for e in entries),
        strict_recovery_correct=sum(e["natural_correct"][1] for e in entries), states=len(entries),
        equal_128_pairs=len(equal),
        proposal_fewer_rounds=sum(e["prefix_rounds"][0] < e["prefix_rounds"][1] for e in equal),
        proposal_more_rounds=sum(e["prefix_rounds"][0] > e["prefix_rounds"][1] for e in equal),
        proposal_equal_rounds=sum(e["prefix_rounds"][0] == e["prefix_rounds"][1] for e in equal),
        strict_natural_truncated=sum(f != "eos" for e in entries for f in e["natural_finish"]))
    # Same-task quality contrasts only; a high-regret lone negative cannot establish incremental confidence value.
    contrasts = []
    for q, task in result["tasks"].items():
        for i, a in enumerate(task["entries"]):
            for b in task["entries"][i + 1:]:
                sa, sb = a["state"], b["state"]
                if (a["natural_correct"][0] != b["natural_correct"][0]
                        and abs(sa["regret"] - sb["regret"]) <= .25 and abs(sa["run"] - sb["run"]) <= 2):
                    contrasts.append(dict(request_id=q, positions=[sa["position"], sb["position"]]))
    result["matched_quality_contrasts"] = contrasts
    context_path = directory / "semantic-context.json"
    if context_path.exists():
        decoder = json.loads((directory / "context-decoder-provenance.json").read_text(encoding="utf-8"))
        require(decoder["output_sha256"] == file_hash(context_path)
                and decoder["script_sha256"] == file_hash(ROOT / "scripts/export_asd_utility_context.py"),
                "Decoded context source hash")
        require(decoder["target_revision"] == info["engine"]["target_revision"], "Decoded tokenizer revision")
        contexts = json.loads(context_path.read_text(encoding="utf-8"))
        indexed = {(r["request_id"], r["position"]): r for r in contexts}
        require(len(indexed) == len(entries) == len(contexts), "Decoded context coverage")
        for q, task in result["tasks"].items():
            for e in task["entries"]:
                s = e["state"]
                c = indexed[q, s["position"]]
                source = references[q]["events"][s["block"]]
                require(c["block_accepted"] == source["accepted"] and c["block_strict"] == source["strict_accepted"]
                        and c["run_gate_accepts"] == (s["regret"] / s["run"] <= CONFIG.gate), "Decoded context metadata")
                e["decoded_context"] = c
    return result


def render(result):
    lines = ["# Utility 错误轨迹与计算收益诊断", "",
             f"两个已观察开发题；完整策略{result['policy_rows']}次、严格分支{result['branch_rows']}次，"
             f"共{result['policy_rows'] + result['branch_rows']}生成。独立标量核对{result['policy_blocks']}个策略块。",
             "拒绝位置从运行前冻结的Utility源轨迹选择；包含块内第二个例外。不是新题泛化或在线加速实验。", ""]
    for q, task in result["tasks"].items():
        lines.extend([f"## {q}", "",
                      f"原Utility：正确={task['baseline_correct']}，{task['baseline_tokens']}token/"
                      f"{task['baseline_rounds']}轮，regret={task['baseline_spent']:.4f}。精确复现旧输出与账本。", "",
                      "同前缀严格分支列顺序均为proposal/recovery；位置为零起始输出token位置。Run包含当前例外。", "",
                      "|位置|regret|Run|p_top|p候选|排名|拒绝一次后正确|拒绝后token/轮|严格自然正确P/R|严格自然后续token P/R|128后续轮P/R|",
                      "|---|---:|---:|---:|---:|---:|---|---|---|---|---|"])
        for e in task["entries"]:
            s = e["state"]
            pairs = lambda values: "/".join(str(v) for v in values)
            lines.append(f"|{s['position']}|{s['regret']:.4f}|{s['run']}|{s['top_probability']:.4f}|"
                         f"{s['proposal_probability']:.4f}|{s['target_rank']}|{e['repaired_correct']}|"
                         f"{e['repaired_tokens']}/{e['repaired_rounds']}|{pairs(e['natural_correct'])}|"
                         f"{pairs(e['natural_tokens'])}|{pairs(e['prefix_rounds'])}"
                         f"{'（等128）' if e['equal_128'] else '（非等量）'}|")
        fixes = [e["state"]["position"] for e in task["entries"] if not task["baseline_correct"] and e["repaired_correct"]]
        damage = [e["state"]["position"] for e in task["entries"] if task["baseline_correct"] and not e["repaired_correct"]]
        labels = [e["state"]["position"] for e in task["entries"] if e["natural_correct"][0] != e["natural_correct"][1]]
        lines.extend(["", f"完整策略单次拒绝修复位置：{fixes}；损伤位置：{damage}。",
                      f"严格自然分支质量不同的位置：{labels}。", ""])
    summary = result["summary"]
    lines.extend(["## 本轮判断", "",
                  f"严格自然分支proposal正确{summary['strict_proposal_correct']}/{summary['states']}，"
                  f"recovery正确{summary['strict_recovery_correct']}/{summary['states']}；"
                  f"自然分支截断{summary['strict_natural_truncated']}条。",
                  f"双方均达到128 continuation token的{summary['equal_128_pairs']}对中，proposal后续轮数更少"
                  f"{summary['proposal_fewer_rounds']}对、更多{summary['proposal_more_rounds']}对、"
                  f"相同{summary['proposal_equal_rounds']}对。局部Run不能直接代表分叉后的未来验证效率；"
                  "这只是未来收益建模的诊断依据，不是新模型已有效的证据。",
                  f"同题、regret差<=.25且Run差<=2的proposal质量不同状态对：{result['matched_quality_contrasts']}。"
                  "未形成这样的对照时，不能宣称置信度有独立于现有变量的质量预测能力。", ""])
    gsm = result["tasks"]["gsm8k:21"]
    decisive = [e for e in gsm["entries"] if e["natural_correct"] == [False, True]]
    for e in decisive:
        s = e["state"]
        if "decoded_context" in e:
            c = e["decoded_context"]
            lines.extend([f"离线tokenizer解码显示位置{s['position']}的proposal为`{c['candidate_text'].strip()}`，"
                          f"recovery为`{c['recovery_text'].strip()}`；源文本此时已写出23 years ago，"
                          "proposal进入Final Answer，recovery开启计算块并在实际分支补出37−23=14。"
                          "这是给定前缀下继续计算与直接收尾的差别，不是数字token的简单替换。"
                          "语义上下文是GPU完成后的离线解码追加，不属于预先定义的在线特征。", ""])
        lines.extend([f"gsm8k:21位置{s['position']}有直接的严格分支质量差异；regret={s['regret']:.4f}，"
                      f"Run={s['run']}，r/Run={s['regret']/s['run']:.4f}。现有Run局部门控"
                      f"{'会拒绝' if s['regret']/s['run'] > CONFIG.gate else '会允许'}该状态；"
                      "这只是在同一源状态计算门控，不冒充完整Run重跑轨迹。",
                      f"该例外前已花{s['spent_before_token']:.4f}，剩余预算{CONFIG.budget-s['spent_before_token']:.4f}，"
                      f"足以支付当前{s['regret']:.4f}；源完整Utility最终支出{gsm['baseline_spent']:.4f}<8。"
                      "本例不能归因于预算耗尽，也不能单凭此例支持动态预算价格。", ""])
    lines.extend(["完整策略多处拒绝能修复同一错答，说明轨迹对早期选择敏感；不能把这些位置全部标为单token有害。"
                  "已找到的严格负分支仍可由既有高regret/Run描述，暂不新增置信度阈值或训练风险头。"
                  "若继续研究，先检验现成特征能否预测后续验证效率，并与Run/固定价格对照；本轮没有新的线上性能结论。", "",
                  "## 解释范围与后续决策", "",
                  "完整策略只改变源状态的一次接受长度，后续重新执行原Utility并保留先前支出；原块后续候选被丢弃，因此修复效果可含后续策略连锁变化。",
                  "严格分支保持原有前缀、强制一个token后重新prefill，测的是严格后续条件下的替换后果；它与完整Utility拒绝是不同干预。",
                  "正确性沿用原数据集最终答案评分，不保证整段推理一致。位置211恢复分支仍保留前缀中23 years ago的错误陈述，随后补算并输出14；该位置是已观察到的纠错机会，不是全部错误的唯一初始原因。",
                  "128 continuation只有双方都完整达到128才作等token轮数比较；文本仍可不同，开头prefill也不同。自然EOS长度与轮数分列。所有时延只有一次且含插桩或重prefill，不能当在线加速证据。",
                  "高p_top、概率集中或较低熵不是答案正确率。置信度是否提供regret/Run之外的独立信号，需要质量标签差异及可比状态；本轮不拟合权重、不自动扫阈值。",
                  "没有单次修复不等于例外无害，可能有组合效应或更早错误；出现单次修复也不能把该token在所有上下文中标为危险。",
                  "若已有特征不能区分收益/风险，保留简单Utility，不凭这一批开发状态上线更复杂策略。", "",
                  f"冻结执行commit `{result['commit']}`。运行前协议见ASD_UTILITY_CAUSAL_PROTOCOL.md；公开原始证据位于results/asd-utility-causal。",
                  "重建：`python -m scripts.write_asd_utility_causal_report`。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/asd-utility-causal")
    args = parser.parse_args()
    result = analyze(args.source)
    write(args.source / "analysis.json", result)
    (ROOT / "docs/ASD_UTILITY_CAUSAL_RESULTS.md").write_text(render(result), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in result.items() if k != "tasks"}))


if __name__ == "__main__":
    main()
