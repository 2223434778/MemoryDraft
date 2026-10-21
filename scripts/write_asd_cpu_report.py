"""Question-paired analysis of a single verifier execution change."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from memorydraft.evaluation.asd_suite import BENCHMARKS
from memorydraft.evaluation.runtime_migration import read_rows
from scripts.diagnose_transport_online import write
from scripts.run_asd_cpu_probe import ARMS, LIMIT, METHODS, REPEATS, REPLAY_REPEATS, timing_order
from scripts.run_asd_execution_probe import CAPTURE_ROUNDS, require_same, selected_records
from scripts.write_asd_block_suite_report import audit_events

ROOT = Path(__file__).resolve().parents[1]


def audit(rows, mechanism, previous, stops):
    tasks = {r["id"] for r in selected_records()}
    expected = {(q, m, impl, rep) for q in tasks for m, impl in ARMS for rep in range(REPEATS)}
    seen = set()
    for r in rows:
        key = r["request_id"], r["method"], r["implementation"], r["repeat"]
        if key not in expected or key in seen or r["stage"] != "timed" or r["max_new_tokens"] != LIMIT:
            raise ValueError("Invalid timed matrix/identity")
        seen.add(key)
        if not math.isfinite(r["latency_seconds"]) or r["latency_seconds"] <= 0:
            raise ValueError("Invalid timed latency")
        if (not math.isfinite(r["policy_setup_seconds"]) or r["policy_setup_seconds"] < 0
                or r["engine_latency_seconds"] <= 0
                or r["latency_seconds"] != r["engine_latency_seconds"]+r["policy_setup_seconds"]):
            raise ValueError("Policy setup excluded or invalid timing boundary")
        require_same(r, previous[r["request_id"], r["method"]])
        order = timing_order(r["request_id"], r["repeat"])
        if order[r["slot"]] != (r["method"], r["implementation"]):
            raise ValueError("Changed frozen paired order")
        if r["implementation"] == "cpu" and sum(r["cpu_execution"].values()) != len(r["rounds"]):
            raise ValueError("Missing CPU fast/fallback call accounting")
    if seen != expected:
        raise ValueError("Incomplete timed matrix")
    seen = set()
    blocks = 0
    states = 0
    for r in mechanism:
        key = r["request_id"], r["method"]
        if key in seen or key[0] not in tasks or key[1] not in METHODS or r["stage"] != "capture" or "latency_seconds" in r:
            raise ValueError("Invalid mechanism identity or timing leak")
        seen.add(key)
        require_same(r, previous[key])
        blocks += audit_events(r, stops)
        if sum(r["cpu_execution"].values()) != len(r["rounds"]):
            raise ValueError("Missing shadow block accounting")
        captures = r["resident_replays"]
        if [s["round"] for s in captures] != [i for i in CAPTURE_ROUNDS if i < len(r["rounds"])]:
            raise ValueError("Changed resident state selection")
        for s in captures:
            rd = r["rounds"][s["round"]]
            expected_decision = dict(accepted=rd["accepted"], strict_accepted=rd["strict_accepted"],
                                     relaxed=rd["relaxed_tokens"], regret=rd["regret"])
            if s["expected"] != expected_decision or s["spent_before"] != sum(x["regret"] for x in r["rounds"][:s["round"]]):
                raise ValueError("Changed resident decision or ledger")
            for impl in ("tensor", "cpu"):
                samples = s["samples_seconds"][impl]
                if len(samples) != REPLAY_REPEATS or any(not math.isfinite(x) or x <= 0 for x in samples):
                    raise ValueError("Invalid resident samples")
                if statistics.median(samples) != s["median_seconds"][impl]:
                    raise ValueError("Changed resident median")
        states += len(captures)
    if seen != {(q, m) for q in tasks for m in METHODS}:
        raise ValueError("Incomplete shadow matrix")
    return dict(timed=len(rows), capture=len(mechanism), shadow_blocks=blocks, resident_states=states,
                old_trajectory_checks=len(rows)+len(mechanism))


def estimate(logs, method):
    grouped = defaultdict(list)
    for benchmark, value in logs:
        grouped[benchmark].append(value)
    if set(grouped) != set(BENCHMARKS) or any(len(v) != 2 for v in grouped.values()):
        raise ValueError("Expected two questions in each fixed family")
    rng = random.Random("asd-cpu-bootstrap:"+method)
    samples = sorted(math.exp(statistics.mean(rng.choice(values) for values in grouped.values() for _ in range(2)))
                     for _ in range(5000))
    return dict(n=12, ratio=math.exp(statistics.mean(v for _, v in logs)), ci95=[samples[125], samples[4874]])


def summarize(rows, mechanism):
    timed = {(r["request_id"], r["method"], r["implementation"], r["repeat"]): r for r in rows}
    replays = {(r["request_id"], r["method"]): r for r in mechanism}
    cases = []
    overall = {}
    for method in METHODS:
        logs = []
        for record in selected_records():
            q = record["id"]
            original = [timed[q, method, "tensor", i]["latency_seconds"] for i in range(REPEATS)]
            cpu = [timed[q, method, "cpu", i]["latency_seconds"] for i in range(REPEATS)]
            ratios = [b/a for a, b in zip(original, cpu)]
            value = statistics.mean(math.log(x) for x in ratios)
            logs.append((record["benchmark"], value))
            states = replays[q, method]["resident_replays"]
            resident = {impl: statistics.median(s["median_seconds"][impl] for s in states) for impl in ("tensor", "cpu")}
            cases.append(dict(request_id=q, benchmark=record["benchmark"], method=method,
                              tokens=len(timed[q, method, "cpu", 0]["token_ids"]),
                              rounds=len(timed[q, method, "cpu", 0]["rounds"]),
                              paired_ratios=ratios, ratio=math.exp(value),
                              median_seconds=dict(tensor=statistics.median(original), cpu=statistics.median(cpu)),
                              resident_seconds=resident, resident_ratio=resident["cpu"]/resident["tensor"],
                              fallback_calls=sum(timed[q, method, "cpu", i]["cpu_execution"]["fallback_calls"] for i in range(REPEATS))))
        overall[method] = estimate(logs, method)
    families = {b: {m: math.exp(statistics.mean(math.log(c["ratio"]) for c in cases if c["benchmark"] == b and c["method"] == m))
                    for m in METHODS} for b in BENCHMARKS}
    return dict(cases=cases, overall=overall, families=families,
                note="Ratios CPU/tensor. Five adjacent pairs collapse within each question; bootstrap resamples questions within the six fixed families, not repeats/states. Observed development subset, not independent validation.")


def main():
    base = ROOT/"results/asd-cpu"
    rows = [r for i in (0, 1) for r in read_rows(base/f"part-{i}/rows.jsonl")]
    mechanism = [r for i in (0, 1) for r in read_rows(base/f"part-{i}/mechanism.jsonl")]
    previous = {(r["request_id"], r["method"]): r for i in (0, 1)
                for r in read_rows(ROOT/f"results/asd-execution/part-{i}/rows.jsonl") if r["repeat"] == 0}
    stops = json.loads((base/"part-0/provenance.json").read_text())["stop_token_ids"]
    result = dict(audit=audit(rows, mechanism, previous, stops), **summarize(rows, mechanism))
    write(base/"analysis.json", result)
    lines = ["# ASD短块CPU判定小检查", "", "六类各前两道已观察开发题；420次正式计时、36次独立影子检查。全部遵守EOS前128，无新完整质量评测。时延比CPU/原Tensor，小于1更快。", "",
             "|方法|12题配对几何比|分层题级95% bootstrap|", "|---|---:|---|" ]
    for m, v in result["overall"].items():
        lines.append(f"|{m}|{v['ratio']:.5f}|[{v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}]|")
    lines += ["", "## 分任务（每类2题，只列点估计）", "", "|任务|ASD|Run|Utility|", "|---|---:|---:|---:|"]
    for b, cell in result["families"].items():
        lines.append("|"+b+"|"+"|".join(f"{cell[m]:.5f}" for m in METHODS)+"|")
    lines += ["", "## 逐题与输入就绪回放", "", "|题|方法|token/轮|在线时延比|resident Tensor ms|resident CPU ms|回退调用（5次）|", "|---|---|---:|---:|---:|---:|---:|"]
    for c in result["cases"]:
        lines.append(f"|{c['request_id']}|{c['method']}|{c['tokens']}/{c['rounds']}|{c['ratio']:.5f}|{c['resident_seconds']['tensor']*1000:.4f}|{c['resident_seconds']['cpu']*1000:.4f}|{c['fallback_calls']}|")
    lines += ["", "## 审计", "", json.dumps(result["audit"], ensure_ascii=False), "",
              "所有在线及插桩轨迹/预算与旧记录一致；影子比较逐轮精确核对，原记录另用既有独立标量检查真实候选EOS。原生DFlash对照完整保留。回放/影子成本不进入正式时延；微基准收益不能代替在线收益。原12题属于开发集，本轮不推广为完整自然工作负载或所有GPU的结论。", "",
              "复现：`.venv\\Scripts\\python.exe -m scripts.write_asd_cpu_report`。解释与取舍见ASD_CPU_FINDINGS.md。", ""]
    (ROOT/"docs/ASD_CPU_RESULTS.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(result["audit"]))


if __name__ == "__main__":
    main()
