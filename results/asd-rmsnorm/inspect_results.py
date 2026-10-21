"""Postrun descriptive diagnostics; does not change the frozen paired estimates."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from memorydraft.evaluation.runtime_migration import read_rows
from scripts.diagnose_transport_online import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("results/asd-rmsnorm"))
    base = parser.parse_args().base
    analysis = json.loads((base/"analysis.json").read_text())
    micro = analysis["rmsnorm_replays"]
    grouped = defaultdict(list)
    for s in micro:
        key = s["region"].split(".")[0]+"."+s["region"].split(".")[-1]+"."+str(s["call"])
        grouped[key].append(s["median_seconds"]["rmsnorm-inductor"]/s["median_seconds"]["eager"])
    pre = {(r["request_id"], r["method"], r["profile"]): r for i in (0, 1)
           for r in read_rows(base/f"part-{i}/preflight.jsonl")}
    pairs = [(r, pre[q, m, "rmsnorm-inductor"]) for (q, m, p), r in pre.items() if p == "eager"]
    without_regret = lambda x: [{k: v for k, v in r.items() if k != "regret"} for r in x["rounds"]]
    workers = []
    all_rows = []
    for i in (0, 1):
        p = json.loads((base/f"part-{i}/provenance.json").read_text())
        c = p["compiler"]
        rows = list(read_rows(base/f"part-{i}/rows.jsonl"))
        all_rows.extend(rows)
        times = {(r["request_id"], r["method"], r["repeat"], r["profile"]): r["latency_seconds"] for r in rows}
        saving = statistics.mean(t-times[q, m, n, "rmsnorm-inductor"] for (q, m, n, prof), t in times.items() if prof == "eager")
        workers.append(dict(shard=i, inventory_by_owner=dict(Counter(x["region"].split(".")[0] for x in c["inventory"])),
                            cold_wall_seconds=p["cold_preparation_wall_seconds"], cache_bytes=p["cache_bytes_after_preparation"],
                            outer_compile_seconds=c["outer_compile_seconds"], controller_seconds=p["controller_setup_seconds"],
                            mean_paired_saving_seconds=saving,
                            conditional_amortization_requests=(c["outer_compile_seconds"]+p["controller_setup_seconds"])/saving if saving > 0 else None))
    result = dict(stage="postrun-descriptive-only", workers=workers,
        faster_questions={m: sum(c["ratio"] < 1 for c in analysis["cases"] if c["method"] == m) for m in analysis["overall"]},
        numeric=dict(states=len(micro), bitwise_equal=sum(s["numeric"]["bitwise_equal"] for s in micro),
                     max_abs=max(s["numeric"]["max_abs"] for s in micro), max_rms=max(s["numeric"]["rms"] for s in micro),
                     max_relative_floor_1e_6=max(s["numeric"]["max_relative_floor_1e_6"] for s in micro),
                     stride_differences=sum(s["original_stride"] != s["replay_stride"] for s in micro)),
        replay_by_role={k: dict(n=len(v), geometric_ratio=math.exp(statistics.mean(map(math.log, v)))) for k, v in grouped.items()},
        trajectories=dict(max_spent_difference=max(abs((a["asd"] or {}).get("spent", 0)-(b["asd"] or {}).get("spent", 0)) for a, b in pairs),
                          round_pairs=sum(len(a["rounds"]) for a, b in pairs), rounds_equal_without_regret=all(without_regret(a) == without_regret(b) for a, b in pairs)),
        compiled_switch_median_seconds=statistics.median(r["profile_switch_seconds"] for r in all_rows if r["profile"] != "eager"),
        peak_allocated_bytes={p: max(r["peak_gpu_bytes"] for r in all_rows if r["profile"] == p) for p in ("eager", "rmsnorm-inductor")},
        note="Same-input roles and amortization are descriptive, not additional independent requests or a production ROI guarantee. Outer time contains backend time; cold wall also contains inference.")
    write(base/"interpretation.json", result)
    print(json.dumps(dict(states=len(micro), workers=workers)))


if __name__ == "__main__":
    main()
