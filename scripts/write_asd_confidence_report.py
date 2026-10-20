"""Audit frozen confidence diagnostics and rebuild the development-only report."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from memorydraft.evaluation.asd_budget import load_tasks, score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from scripts.prepare_asd_attribution import scalar_decision
from scripts.run_asd_confidence_probe import ALPHA, POLICIES, ROOT, TASK_IDS, previous_references, select_states
from scripts.run_asd_budget_probe import compare_previous
from scripts.write_asd_attribution_report import decompose, totals

PARTS = {"math-a": ("math500:1", "math500:4"), "mixed-b": ("math500:7", "gsm8k:21")}


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)


def decision(event, config, spent, *, alpha=0):
    stop = min(event["proposal_eos"], default=None)
    ceiling = scalar_decision(event, replace(config, selection="longest"), spent, stop)
    if config.selection != "utility":
        return ceiling
    k = len(event["mismatch"])
    end = k + 1 if stop is None else stop
    baseline = min(ceiling["strict_accepted"], end)
    choices = []
    for length in range(baseline, ceiling["accepted"] + 1):
        priced = sum(r * (1 + alpha * (1 - 2 * mass)) for m, r, mass in
                     zip(event["mismatch"][:length], event["regrets"][:length], event["pair_mass"][:length]) if m)
        gain = min(length + 1, end) - min(baseline + 1, end)
        choices.append((gain - config.regret_price * priced, -priced, length))
    accepted = max(choices)[2]
    return {**ceiling, "accepted": accepted, "relaxed": sum(event["mismatch"][:accepted]),
            "regret": sum(r for m, r in zip(event["mismatch"][:accepted], event["regrets"][:accepted]) if m)}


def audit_events(row, eos_ids):
    method = row["method"]
    config = POLICIES[method]
    spent, exceptions = 0., 0
    if len(row["events"]) != len(row["rounds"]) or len(row["events"]) != row["counters"]["rounds"]:
        raise ValueError("Missing mechanism rounds")
    for event, rd in zip(row["events"], row["rounds"]):
        k = len(event["proposal_ids"])
        if not k or any(len(event[n]) != k for n in ("target_ids", "mismatch", "regrets", "values",
                        "pair_mass", "top_probability", "proposal_probability", "draft_probability", "target_rank")):
            raise ValueError("Misaligned confidence observations")
        if event["mismatch"] != [a != b for a, b in zip(event["proposal_ids"], event["target_ids"])]:
            raise ValueError("Token disagreement trace differs")
        if event["proposal_eos"] != [i + 1 for i, t in enumerate(event["proposal_ids"]) if t in eos_ids]:
            raise ValueError("Proposal EOS trace differs")
        for i, mismatch in enumerate(event["mismatch"]):
            top, proposal, mass = (event[n][i] for n in ("top_probability", "proposal_probability", "pair_mass"))
            draft = event["draft_probability"][i]
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (top, proposal, mass, draft)):
                raise ValueError("Invalid probability")
            if not math.isclose(mass, min(1., top + proposal if mismatch else top), abs_tol=2e-7):
                raise ValueError("Pair mass double-counted or changed")
            if min(top, proposal) > 1e-30 and not math.isclose(math.log(top / proposal), event["regrets"][i], abs_tol=2e-4):
                raise ValueError("Probability and regret disagree")
            if event["target_rank"][i] < 1 or (not mismatch and event["target_rank"][i] != 1):
                raise ValueError("Invalid candidate rank")
        d = decision(event, config, spent, alpha=ALPHA if method == "Confidence-Utility" else 0)
        if any(not close(d[key], event[key]) for key in d):
            raise ValueError("Independent confidence decision differs")
        expected_values = [k - i if config.value == "remaining" else
                           next((j - i for j in range(i + 1, k) if event["mismatch"][j]), k - i) for i in range(k)]
        if event["values"] != expected_values:
            raise ValueError("Suffix values changed")
        spent += d["regret"]
        exceptions += d["relaxed"]
        if not close(spent, event["spent"]) or spent > config.budget + 1e-8:
            raise ValueError("Raw regret ledger changed")
        before, accepted = event["generated_before"], d["accepted"]
        if before != rd["generated_before"] or accepted != rd["accepted"]:
            raise ValueError("Event/round disagreement")
        if row["token_ids"][before:before + accepted] != event["proposal_ids"][:accepted]:
            raise ValueError("Committed prefix differs from scored proposal")
    if not close(spent, row["asd"]["spent"]) or exceptions != row["asd"]["relaxed_tokens"]:
        raise ValueError("Final ledger differs")


def evidence_hash(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == expected:
        return
    legacy = {"results/research-audit/model_preflight.json", "results/research-audit/pilot_summary.json"}
    if path.relative_to(ROOT).as_posix() in legacy and hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() == expected:
        return
    raise ValueError("Protected evidence changed: " + str(path))


def analyze(source):
    source = Path(source)
    manifest = json.loads((source / "ASD_CONFIDENCE_SOURCE_MANIFEST.json").read_text())
    verification = json.loads((source / "execution-verification.json").read_text())
    if verification["status"] != "complete" or verification["files_verified_before"] != len(manifest["files"]) or verification["files_verified_after"] != len(manifest["files"]):
        raise ValueError("Execution verification incomplete")
    old = {n: h for n, h in manifest["files"].items() if n.startswith("results/") and not n.startswith("results/raw/")}
    for name, digest in old.items():
        evidence_hash(ROOT / name, digest)
    runtime_files = {n.removeprefix("memorydraft/"): h for n, h in manifest["files"].items()
                     if n.startswith("memorydraft/") and n.endswith((".py", ".json", ".txt"))}
    infos, mech, natural, prefixes, branches, selections = {}, [], [], [], [], {}
    for part, ids in PARTS.items():
        directory = source / part
        info = json.loads((directory / "provenance.json").read_text())
        if (info["status"] != "complete" or not info["original_unchanged"] or info["original_sha256"] != old
                or info["task_ids"] != list(ids) or info["alpha"] != ALPHA
                or info["policies"] != {m: asdict(c) for m, c in POLICIES.items()}
                or info["max_new_tokens"] != 2048 or info["prefix_tokens"] != 128):
            raise ValueError("Frozen configuration changed")
        if (info["source"]["files"] != runtime_files or info["source"]["sha256"] !=
                hashlib.sha256(json.dumps(runtime_files, sort_keys=True).encode()).hexdigest()):
            raise ValueError("Runtime source differs")
        for key, name in (("runner_sha256", "scripts/run_asd_confidence_probe.py"),
                          ("data_sha256", "data/research/asd-budget/records.json"),
                          ("protocol_sha256", "docs/ASD_CONFIDENCE_PROTOCOL.md")):
            if info[key] != manifest["files"][name] or file_hash(ROOT / name) != info[key]:
                raise ValueError("Frozen runner/data/protocol changed")
        for stage, destination, repeats in (("mechanism", mech, 1), ("natural", natural, 2), ("prefix", prefixes, 2)):
            rows = read_rows(directory / stage / "rows.jsonl")
            expected = {(q, m, r) for q in ids for m in POLICIES for r in range(repeats)}
            actual = [(r["request_id"], r["method"], r["repeat"]) for r in rows]
            if set(actual) != expected or len(actual) != len(expected) or info[stage + "_rows"] != len(rows):
                raise ValueError("Missing or duplicate generation")
            destination.extend(rows)
        br = read_rows(directory / "branches.jsonl")
        if len(br) != info["branch_rows"]:
            raise ValueError("Missing intervention rows")
        branches.extend(br)
        selections.update(json.loads((directory / "selections.json").read_text()))
        infos[part] = info
    records = {r["id"]: r for r in load_tasks(ROOT / "data/research/asd-budget/records.json")}
    mechanisms = {(r["request_id"], r["method"]): r for r in mech}
    previous = previous_references()
    generation = json.loads((source / "target-generation-config.json").read_text())
    eos_ids = set(generation["eos_token_id"])
    if not eos_ids or any(r["token_ids"][-1] not in eos_ids for r in mech if r["finish_reason"] == "eos"):
        raise ValueError("Target EOS configuration differs")
    for row in mech:
        audit_events(row, eos_ids)
        q = row["request_id"]
        base = mechanisms[q, "DFlash"]
        if (row["prompt_sha256"], row["snapshot_id"]) != (base["prompt_sha256"], base["snapshot_id"]):
            raise ValueError("Different Target input or memory")
        if row["quality"] != score(records[q], row["text"]):
            raise ValueError("Changed quality scoring")
        if row["method"] in ("DFlash", "ASD", "ASD-Run", "ASD-Utility"):
            old = compare_previous(SimpleNamespace(**row), previous[q, row["method"]])
            if row["previous_prefix"] != old:
                raise ValueError("Previous trajectory changed")
    for row in natural + prefixes:
        reference = mechanisms[row["request_id"], row["method"]]
        if (row["prompt_sha256"], row["snapshot_id"]) != (reference["prompt_sha256"], reference["snapshot_id"]):
            raise ValueError("Timed Target input changed")
        if row["latency_seconds"] <= 0 or row["quality"] != score(records[row["request_id"]], row["text"]):
            raise ValueError("Invalid timed result")
        if row in natural and (row["token_ids"] != reference["token_ids"] or row["asd"] != reference["asd"]
                               or row["counters"] != reference["counters"]):
            raise ValueError("Timed mechanism identity differs")
        if row in prefixes:
            if row["full_prefix"] != compare_previous(SimpleNamespace(**reference), row):
                raise ValueError("Short cap changed an earlier prefix")
            sibling = next(r for r in prefixes if r["request_id"] == row["request_id"] and r["method"] == row["method"]
                           and r["repeat"] != row["repeat"])
            if sibling["token_ids"] != row["token_ids"]:
                raise ValueError("Prefix repeats differ")
    interventions = []
    for q in TASK_IDS:
        source_row = mechanisms[q, "ASD"]
        expected = select_states(source_row)
        if selections[q] != expected:
            raise ValueError("Interventions selected using changed rule")
        for state in expected["states"]:
            pair = [r for r in branches if r["request_id"] == q and r["state"] == state]
            if len(pair) != 2 or {r["arm"] for r in pair} != {"proposal", "recovery"}:
                raise ValueError("Missing intervention pair")
            arms = {r["arm"]: r for r in pair}
            if len({r["common_prefix_sha256"] for r in pair}) != 1:
                raise ValueError("Intervention common prefixes differ")
            for row in pair:
                forced = state["candidate" if row["arm"] == "proposal" else "recovery"]
                position = state["position"]
                stable = row["replay_argmax"] == state["recovery"] and abs(row["replay_regret"] - state["regret"]) <= 1e-3
                if (row["token_ids"][:position] != source_row["token_ids"][:position]
                        or row["token_ids"][position] != forced or row["forced_token"] != forced
                        or row["same_state_stable"] != stable
                        or row["quality"] != score(records[q], row["text"])
                        or row["continuation_tokens"] != len(row["token_ids"]) - position - 1):
                    raise ValueError("Intervention replay/output differs")
            interventions.append(dict(request_id=q, matched=expected["matched"], state=state,
                                      stable=all(r["same_state_stable"] for r in pair),
                                      proposal_correct=arms["proposal"]["quality"]["correct"],
                                      recovery_correct=arms["recovery"]["quality"]["correct"],
                                      proposal_tokens=len(arms["proposal"]["token_ids"]),
                                      recovery_tokens=len(arms["recovery"]["token_ids"]),
                                      proposal_rounds=arms["proposal"]["continuation_rounds"],
                                      recovery_rounds=arms["recovery"]["continuation_rounds"],
                                      finish={k: v["finish_reason"] for k, v in arms.items()}))
    if len(branches) != 2 * len(interventions):
        raise ValueError("Unexpected intervention rows")
    tables, requests, shadow = {}, {}, {}
    for benchmark in ("math500", "gsm8k"):
        ids = [q for q in TASK_IDS if q.startswith(benchmark + ":")]
        chosen = [r for r in natural if r["request_id"] in ids]
        main = totals(chosen, POLICIES)
        equal_ids = [q for q in ids if all(len(r["token_ids"]) == 128 for r in prefixes if r["request_id"] == q)]
        prefix_table = totals([r for r in prefixes if r["request_id"] in equal_ids], POLICIES)
        for method, table in main.items():
            rows = [mechanisms[q, method] for q in ids]
            table.update(correct=sum(r["quality"]["correct"] for r in rows), requests=len(ids),
                         truncated=sum(r["finish_reason"] == "length" for r in rows),
                         regret=sum(r["asd"]["spent"] for r in rows),
                         exceptions=sum(r["asd"]["relaxed_tokens"] for r in rows))
        tables[benchmark] = dict(main=main, prefix=prefix_table, equal_prefix_ids=equal_ids,
                                 comparisons={m: decompose(main["ASD-Utility"], t) for m, t in main.items()})
    for q in TASK_IDS:
        requests[q] = {}
        for method in POLICIES:
            ref = mechanisms[q, method]
            rows = [r for r in natural if r["request_id"] == q and r["method"] == method]
            requests[q][method] = dict(seconds=sum(r["latency_seconds"] for r in rows), tokens=len(ref["token_ids"]),
                                      rounds=ref["counters"]["rounds"], quality=ref["quality"],
                                      finish_reason=ref["finish_reason"], regret=ref["asd"]["spent"],
                                      trajectory_equal_confidence=ref["token_ids"] == mechanisms[q, "Confidence-Utility"]["token_ids"])
        shadow[q] = {}
        for source_method in ("ASD", "ASD-Utility", "Confidence-Utility"):
            counts = {m: dict(equal=0, more=0, less=0) for m in POLICIES}
            for event in mechanisms[q, source_method]["events"]:
                spent = event["spent"] - event["regret"]
                baseline = decision(event, POLICIES["ASD-Utility"], spent)["accepted"]
                for method, config in POLICIES.items():
                    n = decision(event, config, spent, alpha=ALPHA if method == "Confidence-Utility" else 0)["accepted"]
                    counts[method]["equal" if n == baseline else "more" if n > baseline else "less"] += 1
            shadow[q][source_method] = counts
    return dict(execution_commit=manifest["commit"], runtime_sha256=next(iter(infos.values()))["source"]["sha256"],
                counts=dict(mechanism=len(mech), natural=len(natural), prefix=len(prefixes), branches=len(branches)),
                tables=tables, requests=requests, interventions=interventions, shadow=shadow)


def render(a):
    lines = ["# ASD 置信度小规模开发检查", "", "仅4道已观察开发题；不是独立泛化验证。alpha=.5、价格2冻结，固定价格1/3为对照；原ASD预算仍按未加权regret扣除。有效选择价格在1到3之间，固定对照不等于扫描所有中间价格。概率按原始logits的T=1归一化，仅作启发式特征，生成仍为greedy，没有进行正确率校准。", "",
             f"已审计机制{a['counts']['mechanism']}、完整自然计时{a['counts']['natural']}、128前缀计时{a['counts']['prefix']}、分支{a['counts']['branches']}次。全部质量/截断保留；计时输出与机制一致，旧四组保留之前未受短上限影响的前缀。", ""]
    for benchmark, table in a["tables"].items():
        main, prefix = table["main"], table["prefix"]
        lines += [f"## {benchmark}", "", "| 方法 | 完整秒（两重复） | token（两重复） | TPS | 每100token轮 | 正确 | 截断 | regret | 相对Utility时延 | 等128前缀秒 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m in POLICIES:
            t = main[m]
            lines.append(f"| {m} | {t['seconds']:.6f} | {t['tokens']} | {t['tps']:.3f} | {100*t['rounds_per_token']:.3f} | {t['correct']}/{t['requests']} | {t['truncated']} | {t['regret']:.3f} | {(t['seconds']/main['ASD-Utility']['seconds']-1)*100:+.2f}% | {prefix[m]['seconds']:.6f} |")
        c = table["comparisons"]["Confidence-Utility"]
        lines += ["", f"Confidence/Utility分解：长度比{c['length_ratio']:.6f}，轮密度比{c['round_density_ratio']:.6f}，每轮成本比{c['round_cost_ratio']:.6f}；三者乘积为时延比{c['latency_ratio']:.6f}。各项来自不同轨迹，不能当成独立因果贡献。",
                  f"等128token集合：`{', '.join(table['equal_prefix_ids'])}`；均遵守EOS，128上限可能改变最后一块决策，裁短块前的前缀已审计。", ""]
    lines += ["## 逐题与固定价格对照", "", "| 请求 | 方法 | 秒（两重复） | token（单次） | 验证轮（单次） | 正确 | 与置信度组完整token相同 |", "|---|---|---:|---:|---:|---|---|"]
    for q, methods in a["requests"].items():
        for m, r in methods.items():
            lines.append(f"| {q} | {m} | {r['seconds']:.6f} | {r['tokens']} | {r['rounds']} | {r['quality']['correct']} | {r['trajectory_equal_confidence']} |")
    lines += ["", "## 相同前缀的单次例外干预", "", "严格续写；每对只改变一个强制token，缓存重新计算。仅单例外后果，不等于近似策略完整rollout。matched表示源状态的regret差<=.25、run差<=2、pair_mass差>=.1；unmatched不能作为控制这些因素后的证据。", "",
              "| 请求/位置 | matched | pair_mass | regret | run | 同状态核查 | proposal正确 | recovery正确 | proposal/recovery总token | proposal/recovery后续轮 |", "|---|---|---:|---:|---:|---|---|---|---|---|"]
    for item in a["interventions"]:
        s = item["state"]
        lines.append(f"| {item['request_id']}/{s['position']} | {item['matched']} | {s['pair_mass']:.4f} | {s['regret']:.4f} | {s['run']} | {item['stable']} | {item['proposal_correct']} | {item['recovery_correct']} | {item['proposal_tokens']}/{item['recovery_tokens']} | {item['proposal_rounds']}/{item['recovery_rounds']} |")
    lines += ["", "## 相同源状态的离线选择", "", "原ASD、Utility及Confidence-Utility三个源的每块独立借用各自实际账本；没有推进替代策略状态，不能当作全请求节省。全部proposal/EOS已记录，因此可精确裁剪候选EOS。这是离线补充诊断，干预选样始终仅来自原ASD。", "",
              "| 请求 | 源轨迹 | 选择器 | 与Utility相同块 | 多接块 | 少接块 |", "|---|---|---|---:|---:|---:|"]
    for q, sources in a["shadow"].items():
        for source_method, methods in sources.items():
            for m, c in methods.items():
                lines.append(f"| {q} | {source_method} | {m} | {c['equal']} | {c['more']} | {c['less']} |")
    math_table = a["tables"]["math500"]
    gsm_table = a["tables"]["gsm8k"]
    base, conf = math_table["main"]["ASD-Utility"], math_table["main"]["Confidence-Utility"]
    prefix_change = (math_table["prefix"]["Confidence-Utility"]["seconds"] / math_table["prefix"]["ASD-Utility"]["seconds"] - 1) * 100
    lines += ["", "## 本轮判断", "",
              f"三道MATH的完整自然时延变化{(conf['seconds']/base['seconds']-1)*100:+.2f}%，输出长度变化{(conf['tokens']/base['tokens']-1)*100:+.2f}%，自然TPS变化{(conf['tps']/base['tps']-1)*100:+.2f}%，每token轮密度变化{(conf['rounds_per_token']/base['rounds_per_token']-1)*100:+.2f}%；等128token前缀时延变化{prefix_change:+.2f}%。总体没有新增吞吐优势。",
              "这不抹去局部正例：math500:7的置信度组71轮/765token，Utility为77轮/793token，轮密度确有改善；但math500:4变成83轮/576token，对照79轮/601token，方向相反。均为旧开发题，尚不构成场景判别器。",
              f"gsm8k:21置信度组修复了旧Utility错答，但固定价格1和3也答对。置信度组相对价格3时延变化{(gsm_table['main']['Confidence-Utility']['seconds']/gsm_table['main']['Utility-Price3']['seconds']-1)*100:+.2f}%；没有置信度独有的纠错或加速证据。",
              "八个单例外状态的16分支全部答对，无有害/无害质量标签差异，不能计算或宣称风险预测能力；这项干预未识别完整Utility轨迹的致错例外。只有math500:1/7满足预定regret/run配对，另两题为unmatched。",
              "具体指标局限：选中的原ASD状态math500:4/45，Target首选概率约0.9296，proposal约0.0683，二者和约0.9979，regret约2.611。pair_mass很高，但Target对两者的偏好明显不相同。以集中度一律降低价格可能削弱原regret惩罚；这说明该启发式定义有缺陷，并不证明这个源状态是置信度组退化的唯一原因。",
              "阶段决定：保留采集/干预/校验工具，停止扩大这条简单pair_mass动态定价规则；不在四题上反向调权重或继续扫alpha。尚不能据此否定所有置信度方法。若后续继续，应先在已知错误的Utility轨迹定位有害例外，再检验新增信号能否在控制regret和后缀长度后区分它们。"]
    lines += ["", "## 解释边界", "", "概率集中度是启发式，不是答案正确率；首选和proposal概率接近仍可对应相反数字/结论。原始regret、运行开销、答案质量和长度必须共同判断。单个开发正例、局部更长接受或更短回答均不能证明稳定吞吐增量。",
              "本轮不训练新风险模型、不根据结果扫描alpha。固定价格对照若能取得同等/更好行为，置信度独有价值仍不成立；不扩大该启发式到新题，除非机制和含全部开销的对照同时支持。",
              "", f"冻结执行commit `{a['execution_commit']}`；runtime `{a['runtime_sha256']}`。原发布证据与执行源码前后SHA不变。",
              "重建：`python -m scripts.write_asd_confidence_report`。原始公开题输出、概率、EOS、命令与测试在results/asd-confidence；权重不入库。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/asd-confidence")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/ASD_CONFIDENCE_RESULTS.md")
    args = parser.parse_args()
    a = analyze(args.source)
    (args.source / "analysis.json").write_bytes((json.dumps(a, ensure_ascii=False, indent=2) + "\n").encode())
    args.report.write_bytes(render(a).encode())
    print(json.dumps({b: t["comparisons"]["Confidence-Utility"] for b, t in a["tables"].items()}))


if __name__ == "__main__":
    main()
