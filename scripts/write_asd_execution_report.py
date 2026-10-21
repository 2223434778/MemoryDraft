"""Audit and summarize execution diagnosis; no profiler latency is a speedup."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_suite import BENCHMARKS
from memorydraft.evaluation.runtime_migration import read_rows
from scripts.diagnose_transport_online import write
from scripts.run_asd_execution_probe import (CAPTURE_ROUNDS, LIMIT, METHODS, REPEATS, REPLAY_REPEATS,
                                            require_same, selected_records)

ROOT = Path(__file__).resolve().parents[1]


def audit(timed, profiles, captures, previous):
    tasks = {r["id"] for r in selected_records()}
    expected = {(q, m) for q in tasks for m in METHODS}
    if len(timed) != len(expected) * REPEATS or len(profiles) != len(expected) or len(captures) != len(tasks) * 3:
        raise ValueError("Incomplete execution matrix")
    seen = {}
    for row in timed:
        key = row["request_id"], row["method"]
        if key not in expected or row["stage"] != "timed" or row["max_new_tokens"] != LIMIT:
            raise ValueError("Unexpected timed identity")
        if not math.isfinite(row["latency_seconds"]) or row["latency_seconds"] <= 0:
            raise ValueError("Invalid timed latency")
        require_same(row, previous[key])
        repeats = seen.setdefault(key, set())
        if row["repeat"] in repeats:
            raise ValueError("Duplicate repeat")
        repeats.add(row["repeat"])
    if set(seen) != expected or any(v != set(range(REPEATS)) for v in seen.values()):
        raise ValueError("Missing timed arm/repeat")
    for rows, stage in ((profiles, "profile"), (captures, "capture")):
        keys = []
        for row in rows:
            key = row["request_id"], row["method"]
            keys.append(key)
            if row["stage"] != stage or "latency_seconds" in row:
                raise ValueError("Instrumented data mixed with timing")
            require_same(row, previous[key])
            if stage == "profile":
                p = row["profile"]
                for name in ("draft_forward", "target_verify", "accept_decision", "commit_kv", "publish_stop"):
                    if p["scopes"][name]["calls"] != len(row["rounds"]):
                        raise ValueError("Missing per-round scope")
                linked = sum(s["linked_device_us"] for s in p["scopes"].values())
                # CUPTI can retain device events with no frontend correlation.
                # Keep this residual explicitly, rather than inventing an owner.
                if linked > p["device_duration_sum_us"] + 1.:
                    raise ValueError("Device kernel attribution exceeds raw duration")
                if not 0 < p["linked_kernel_count"] <= p["device_event_count"]:
                    raise ValueError("Device kernel attribution exceeds raw event count")
                if not 0 < p["device_active_union_us"] <= p["device_duration_sum_us"] + 1.:
                    raise ValueError("Invalid active device union")
            else:
                states = row["resident_replays"]
                if [s["round"] for s in states] != [i for i in CAPTURE_ROUNDS if i < len(row["rounds"])]:
                    raise ValueError("Changed capture selection")
                for state in states:
                    rd = row["rounds"][state["round"]]
                    if state["expected"] != dict(accepted=rd["accepted"], strict_accepted=rd["strict_accepted"],
                                                  relaxed=rd["relaxed_tokens"], regret=rd["regret"]):
                        raise ValueError("Captured decision does not reproduce online round")
                    spent = sum(r["regret"] for r in row["rounds"][:state["round"]])
                    if state["spent_before"] != spent or state["proposals"] != rd["proposed"]:
                        raise ValueError("Captured budget/state mismatch")
                    for kind in ("policy", "strict_core", "sync_control"):
                        samples = state["samples_seconds"][kind]
                        if len(samples) != REPLAY_REPEATS or any(not math.isfinite(x) or x <= 0 for x in samples):
                            raise ValueError("Invalid replay samples")
                        if statistics.median(samples) != state["median_seconds"][kind]:
                            raise ValueError("Changed replay summary")
        required = expected if stage == "profile" else {k for k in expected if k[1] != "DFlash"}
        if len(set(keys)) != len(keys) or set(keys) != required:
            raise ValueError("Duplicate/missing instrumented identity")
    return dict(timed=len(timed), profile=len(profiles), capture=len(captures),
                replay_states=sum(len(r["resident_replays"]) for r in captures),
                baseline_trajectory_checks=len(timed)+len(profiles)+len(captures),
                profiles_with_unlinked_events=sum(r["profile"]["linked_kernel_count"] < r["profile"]["device_event_count"] for r in profiles))


def summarize(timed, profiles, captures):
    cells = {}
    for row in timed:
        key = row["request_id"], row["method"]
        cells.setdefault(key, []).append(row["latency_seconds"])
    measured = {k: statistics.median(v) for k, v in cells.items()}
    replay = {(r["request_id"], r["method"]): r for r in captures}
    cases = []
    for row in profiles:
        key = row["request_id"], row["method"]
        p = row["profile"]
        scopes = p["scopes"]
        total_gpu = p["device_duration_sum_us"]
        case = dict(request_id=key[0], method=key[1], benchmark=row["benchmark"],
                    tokens=len(row["token_ids"]), rounds=len(row["rounds"]),
                    median_latency_seconds=measured[key], latency_samples_seconds=cells[key],
                    profiler_dilation=row["instrumented_latency_seconds"]/measured[key],
                    device_active_fraction_of_span=p["device_active_union_us"]/p["device_span_us"],
                    unlinked_device_events=p["device_event_count"]-p["linked_kernel_count"],
                    unlinked_device_us=max(0., total_gpu-sum(v["linked_device_us"] for v in scopes.values())),
                    scopes={name: dict(**values, device_fraction=values["linked_device_us"]/total_gpu)
                            for name, values in scopes.items()})
        if key in replay:
            states = replay[key]["resident_replays"]
            medians = {kind: statistics.median(s["median_seconds"][kind] for s in states)
                       for kind in ("policy", "strict_core", "sync_control")}
            case.update(resident_median_seconds=medians, sampled_states=len(states),
                        resident_amortized_proxy_fraction=medians["policy"]*len(row["rounds"])/measured[key])
        cases.append(case)
    families = {}
    for b in BENCHMARKS:
        families[b] = {}
        for method in METHODS:
            selected = [c for c in cases if c["benchmark"] == b and c["method"] == method]
            ratios = [c["median_latency_seconds"]/measured[c["request_id"], "DFlash"] for c in selected]
            families[b][method] = dict(n=len(selected), latency_ratio_to_DFlash=math.exp(statistics.mean(math.log(x) for x in ratios)),
                                      mean_device_fractions={name: statistics.mean(c["scopes"][name]["device_fraction"] for c in selected)
                                                             for name in ("target_prefill", "draft_forward", "target_verify", "accept_decision", "commit_kv", "publish_stop")})
    return dict(cases=cases, families=families,
                note="Two already-observed development tasks per family. Profile, sync wait and GPU durations are overlapping clocks; resident proxy is not a predicted speedup or an upper bound.")


def main():
    base = ROOT / "results/asd-execution"
    rows = lambda name: [r for i in (0, 1) for r in read_rows(base / f"part-{i}/{name}.jsonl")]
    timed, profiles, captures = rows("rows"), rows("profiles"), rows("replays")
    previous = {(r["request_id"], r["method"]): r for i in (0, 1)
                for r in read_rows(ROOT / f"results/asd-block-suite/part-{i}/rows.jsonl")
                if r["stage"] == "prefix" and r["repeat"] == 0 and r["experimental_block_size"] == 16}
    result = dict(audit=audit(timed, profiles, captures, previous), **summarize(timed, profiles, captures))
    write(base / "analysis.json", result)
    lines = ["# ASD 原生16块执行成本诊断", "", "12个预定开发题；144次正式计时、48次独立profile、36次捕获。遵守EOS前128，不替代48题自然质量/时延结果。", "",
             "## 非插桩计时", "", "每类仅2题，表为题级中位时延配对几何比（小于1更快），不计算不可靠的小样本区间。", "",
             "|任务|ASD/DFlash|Run/DFlash|Utility/DFlash|", "|---|---:|---:|---:|"]
    for b, methods in result["families"].items():
        lines.append("|" + b + "|" + "|".join(f"{methods[m]['latency_ratio_to_DFlash']:.4f}" for m in METHODS[1:]) + "|")
    lines += ["", "## 接受判定与真实状态回放", "", "host范围含前序GPU等待；device为直接关联kernel之和，不将两者相加。回放输入已就绪，严格核心省去bonus位置，不是完整在线DFlash。proxy=所采状态策略耗时中位数×在线轮数/正式请求时延，仅用于量级诊断。", "",
              "|题|方法|轮|decision host ms/轮|decision device ms/轮|同步等待 ms/轮|resident策略 ms|resident严格核心 ms|resident proxy|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in result["cases"]:
        s = c["scopes"]["accept_decision"]; n = c["rounds"]
        resident = c.get("resident_median_seconds")
        tail = (f"{resident['policy']*1000:.3f}|{resident['strict_core']*1000:.3f}|{c['resident_amortized_proxy_fraction']:.2%}"
                if resident else "—|—|—")
        lines.append(f"|{c['request_id']}|{c['method']}|{n}|{s['host_elapsed_us']/n/1000:.3f}|{s['linked_device_us']/n/1000:.3f}|{s['sync_host_us']/n/1000:.3f}|{tail}|")
    lines += ["", "## GPU活动归属", "", "各题的直接kernel耗时占比先计算，再在同类别两题平均。不是请求时延的可相加分解，不是优化可兑现比例。", "",
              "|任务|方法|Target预填充|Draft含lm_head|Target验证|接受判定|KV提交|发布/停止|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for b, methods in result["families"].items():
        for method, cell in methods.items():
            lines.append("|"+b+"|"+method+"|"+"|".join(f"{v:.2%}" for v in cell["mean_device_fractions"].values())+"|")
    lines += ["", "## 完整性", "", json.dumps(result["audit"], ensure_ascii=False), "",
              "全部正式/插桩/捕获轨迹、prompt SHA、snapshot、预算和轮次与冻结前阶段16块前128记录一致。CUDA annotation排除，直接kernel链接的事件数/耗时不得超过原始device总数；没有CPU关联的设备事件另列unlinked_device，不强行归属。完整逐题数据、profiler膨胀比例、同步控制与回放样本见analysis.json及压缩原始记录。", "",
              "复现：`.venv\\Scripts\\python.exe -m scripts.write_asd_execution_report`。机制解释与后续取舍见ASD_EXECUTION_FINDINGS.md。", ""]
    (ROOT / "docs/ASD_EXECUTION_RESULTS.md").write_bytes("\n".join(lines).encode("utf-8"))
    print(json.dumps(result["audit"]))


if __name__ == "__main__":
    main()
