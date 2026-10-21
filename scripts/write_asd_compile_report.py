"""Audit regional compilation without hiding numerical/profile differences."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_suite import BENCHMARKS
from memorydraft.evaluation.runtime_migration import read_rows
from scripts.diagnose_transport_online import write
from scripts.run_asd_compile_probe import (LIMIT, METHODS, PROFILES, REPEATS, REPLAY_REPEATS,
                                          previous_rows, timing_order, trajectory_comparison)
from scripts.run_asd_execution_probe import require_same, selected_records
from scripts.write_asd_cpu_report import estimate

ROOT = Path(__file__).resolve().parents[1]


def validate_ledger(row):
    summary = row["asd"]
    if row["method"] in {"AR", "DFlash"}:
        return
    spent = 0.
    relaxed = 0
    for rd in row["rounds"]:
        if not math.isfinite(rd["regret"]) or rd["regret"] < 0 or not 0 <= rd["relaxed_tokens"] <= 2:
            raise ValueError("Invalid compiler-profile regret or cap")
        spent += rd["regret"]
        relaxed += rd["relaxed_tokens"]
    if spent != summary["spent"] or relaxed != summary["relaxed_tokens"] or spent > 8+1e-9:
        raise ValueError("Compiler profile changed the request ledger contract")
    if summary["blocks"] != len(row["rounds"]):
        raise ValueError("Missing compiler-profile rounds")


def audit(rows, preflight, diagnostics, previous, *, profiles=PROFILES, replay_key="mlp_replays", expected_states=None):
    tasks = {r["id"] for r in selected_records()}
    expected_pre = {(q, m, p) for q in tasks for m in ("AR", *METHODS) for p in profiles}
    pre = {}
    for r in preflight:
        key = r["request_id"], r["method"], r["profile"]
        if key not in expected_pre or key in pre or r["stage"] != "preflight" or r["max_new_tokens"] != LIMIT:
            raise ValueError("Invalid compiler preflight identity")
        pre[key] = r
        if r["profile"] == "eager":
            require_same(r, previous[key[:2]])
        else:
            # Input identity is mandatory; numerical output agreement is measured.
            for k in ("request_id", "method", "prompt_sha256", "prompt_tokens", "snapshot_id"):
                if r[k] != previous[key[:2]][k]:
                    raise ValueError("Compiled profile changed input")
        validate_ledger(r)
    if set(pre) != expected_pre:
        raise ValueError("Incomplete compiler preflight")
    expected = {(q, m, p, i) for q in tasks for m in METHODS for p in profiles for i in range(REPEATS)}
    seen = set()
    timed_graphs = 0
    for r in rows:
        key = r["request_id"], r["method"], r["profile"], r["repeat"]
        if key not in expected or key in seen or r["stage"] != "timed":
            raise ValueError("Invalid compiler timed matrix")
        seen.add(key)
        require_same(r, pre[key[:3]])
        if timing_order(key[0], key[3], profiles=profiles)[r["slot"]] != key[1:3]:
            raise ValueError("Changed paired compiler order")
        if (not math.isfinite(r["latency_seconds"]) or r["latency_seconds"] <= 0
                or r["latency_seconds"] != r["engine_latency_seconds"]+r["policy_setup_seconds"]+r["profile_switch_seconds"]):
            raise ValueError("Compiler online cost excluded or invalid")
        difference = r["graphs_after"]-r["graphs_before"]
        if difference < 0 or (r["profile"] == "eager" and difference):
            raise ValueError("Invalid compiler event accounting")
        timed_graphs += difference
    if seen != expected:
        raise ValueError("Incomplete compiler timing matrix")
    seen = set()
    states = 0
    for r in diagnostics:
        q = r["request_id"]
        if q not in tasks or q in seen or r["stage"] != "diagnostic" or "latency_seconds" in r:
            raise ValueError("Invalid module diagnostic identity/timing leak")
        seen.add(q)
        require_same(r, pre[q, "DFlash", "eager"])
        keys = [(s["region"].split(".")[0], s["call"]) if expected_states is None else
                (s["region"].split(".")[0], s["region"].split(".")[-1], s["call"]) for s in r[replay_key]]
        if sorted(keys) != ([("draft", 0), ("target", 0), ("target", 1)] if expected_states is None else expected_states):
            raise ValueError("Missing actual prefill/decode region inputs")
        for state in r[replay_key]:
            for v in ("max_abs", "rms", "max_relative_floor_1e_6"):
                if not math.isfinite(state["numeric"][v]) or state["numeric"][v] < 0:
                    raise ValueError("Invalid same-input numerical diagnostic")
            for p in profiles:
                values = state["samples_seconds"][p]
                if len(values) != REPLAY_REPEATS or any(not math.isfinite(x) or x <= 0 for x in values):
                    raise ValueError("Invalid MLP resident samples")
                if state["median_seconds"][p] != statistics.median(values):
                    raise ValueError("Changed MLP median")
        states += len(keys)
    if seen != tasks:
        raise ValueError("Incomplete MLP diagnostics")
    comparisons = [trajectory_comparison(pre[q, m, "eager"], pre[q, m, profiles[1]])
                   for q in tasks for m in ("AR", *METHODS)]
    strict = {p: sum(pre[q, "DFlash", p]["token_ids"] == pre[q, "AR", p]["token_ids"] for q in tasks) for p in profiles}
    return dict(timed=len(rows), preflight=len(preflight), diagnostics=len(diagnostics), **{replay_key.replace("replays", "states"): states},
                eager_old_trajectory_checks=len(tasks)*5,
                compiled_token_matches=sum(c["token_equal"] for c in comparisons),
                compiled_ledger_matches=sum(c["ledger_equal"] for c in comparisons),
                strict_matches_AR_out_of_12=strict, backend_events_during_timing=timed_graphs)


def summarize(rows, preflight, diagnostics, *, profiles=PROFILES, replay_key="mlp_replays", seed="compile"):
    timed = {(r["request_id"], r["method"], r["profile"], r["repeat"]): r for r in rows}
    pre = {(r["request_id"], r["method"], r["profile"]): r for r in preflight}
    cases = []
    overall = {}
    for method in METHODS:
        logs = []
        for record in selected_records():
            q = record["id"]
            samples = {p: [timed[q, method, p, i]["latency_seconds"] for i in range(REPEATS)] for p in profiles}
            ratios = [c/e for e, c in zip(samples["eager"], samples[profiles[1]])]
            ratio = math.exp(statistics.mean(math.log(x) for x in ratios))
            logs.append((record["benchmark"], math.log(ratio)))
            e, c = pre[q, method, "eager"], pre[q, method, profiles[1]]
            calls = (len(c["rounds"])+1)/(len(e["rounds"])+1)
            length = len(c["token_ids"])/len(e["token_ids"])
            cases.append(dict(request_id=q, method=method, benchmark=record["benchmark"], ratio=ratio,
                              paired_ratios=ratios, latency_samples=samples, comparison=trajectory_comparison(e, c),
                              eager_tokens=len(e["token_ids"]), compiled_tokens=len(c["token_ids"]),
                              eager_rounds=len(e["rounds"]), compiled_rounds=len(c["rounds"]),
                              length_ratio=length, call_density_ratio=calls/length, amortized_cost_ratio=ratio/calls,
                              eager_peak_bytes=max(timed[q, method, "eager", i]["peak_gpu_bytes"] for i in range(REPEATS)),
                              compiled_peak_bytes=max(timed[q, method, profiles[1], i]["peak_gpu_bytes"] for i in range(REPEATS))))
        overall[method] = estimate(logs, seed+":"+method)
    families = {b: {m: math.exp(statistics.mean(math.log(c["ratio"]) for c in cases if c["benchmark"] == b and c["method"] == m))
                    for m in METHODS} for b in BENCHMARKS}
    micro = [dict(request_id=r["request_id"], **s) for r in diagnostics for s in r[replay_key]]
    return dict(cases=cases, overall=overall, families=families, **{replay_key: micro},
                note="Compiled/eager timing ratios; paired repeats collapse per question; stratified 12-question bootstrap. Numerical profiles are separate. Call cost includes all online work and is not pure Target kernel time.")


def main(*, study="COMPILE", region="MLP", profiles=PROFILES, replay_key="mlp_replays", expected_states=None, seed="compile"):
    folder = "asd-"+study.lower()
    base = ROOT/"results"/folder
    read = lambda name: [r for i in (0, 1) for r in read_rows(base/f"part-{i}/{name}.jsonl")]
    rows, preflight, diagnostics = read("rows"), read("preflight"), read("diagnostics")
    info = [json.loads((base/f"part-{i}/provenance.json").read_text()) for i in (0, 1)]
    result = dict(audit=audit(rows, preflight, diagnostics, previous_rows(), profiles=profiles,
                            replay_key=replay_key, expected_states=expected_states),
                  **summarize(rows, preflight, diagnostics, profiles=profiles, replay_key=replay_key, seed=seed),
                  preparation=[dict(shard=p["shard"], controller_setup_seconds=p["controller_setup_seconds"],
                                    cold_preparation_wall_seconds=p["cold_preparation_wall_seconds"],
                                    cache_bytes=p["cache_bytes_after_preparation"],
                                    compiler=p["compiler"]) for p in info])
    write(base/"analysis.json", result)
    lines = ["# 区域"+region+"编译小检查", "", "六类各前两道已观察开发题；四策略、相同CPU接受实现、FP32/原生16块。编译/eager是不同数值执行profile，保持输入和接受规则，数值差异另列。", "",
             "|方法|编译/eager时延比|题级分层95% bootstrap|", "|---|---:|---|"]
    for m, x in result["overall"].items():
        lines.append(f"|{m}|{x['ratio']:.5f}|[{x['ci95'][0]:.5f}, {x['ci95'][1]:.5f}]|")
    lines += ["", "## 分任务（每类2题点估计）", "", "|任务|DFlash|ASD|Run|Utility|", "|---|---:|---:|---:|---:|"]
    for b, cell in result["families"].items():
        lines.append("|"+b+"|"+"|".join(f"{cell[m]:.5f}" for m in METHODS)+"|")
    lines += ["", "## 轨迹、计算量和显存", "", "|题|方法|时延比|token相同|账本相同|原/编译轮数|输出长度比|调用密度比|摊销成本比|编译峰值GiB|", "|---|---|---:|---|---|---|---:|---:|---:|---:|"]
    for c in result["cases"]:
        a = c["comparison"]
        lines.append(f"|{c['request_id']}|{c['method']}|{c['ratio']:.5f}|{a['token_equal']}|{a['ledger_equal']}|{c['eager_rounds']}/{c['compiled_rounds']}|{c['length_ratio']:.4f}|{c['call_density_ratio']:.4f}|{c['amortized_cost_ratio']:.5f}|{c['compiled_peak_bytes']/2**30:.3f}|")
    lines += ["", "## 编译准备", "", "outer为PyTorch版本内_compile.compile_inner计时之和；backend为其内部子阶段，不相加。cold wall还包含预热和两个profile各题预检推理，不能全当编译时间。缓存从空独立目录开始，未启用CUDA Graph或max-autotune。", "",
              "|worker|区域数|最终图数|outer编译秒|backend秒|完整准备wall秒|缓存MiB|", "|---|---:|---:|---|---:|---:|---:|"]
    for p in result["preparation"]:
        c = p["compiler"]
        lines.append(f"|{p['shard']}|{len(c['regions'])}|{c['graph_count']}|{c['outer_compile_seconds']}|{c['backend_seconds']:.3f}|{p['cold_preparation_wall_seconds']:.3f}|{p['cache_bytes']/2**20:.3f}|")
    lines += ["", "## 完整性与边界", "", json.dumps(result["audit"], ensure_ascii=False), "",
              f"全部eager轨迹重现上一版本；同profile重复逐token/轮次/账本相同。跨profile差异完整保留，不放宽接受条件，不强制EOS后继续生成。12个独立DFlash诊断捕获{len(result[replay_key])}个真实{region}输入，同输入的原/编译输出误差与20次交错模块回放见analysis.json。未评测新的完整自然答案质量；不是无损通用加速结论。", "",
              f"重建：`.venv\\Scripts\\python.exe -m scripts.write_asd_{study.lower()}_report`；执行口径及资源预算见ASD_{study}_PROTOCOL.md，解释见ASD_{study}_FINDINGS.md。", ""]
    (ROOT/f"docs/ASD_{study}_RESULTS.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(result["audit"]))


if __name__ == "__main__":
    main()
