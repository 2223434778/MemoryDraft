"""Postrun descriptive facts; primary statistics remain in the frozen report."""
from collections import Counter
import json
from pathlib import Path
import statistics

from memorydraft.evaluation.runtime_migration import read_rows
from scripts.diagnose_transport_online import write
from scripts.run_asd_compile_probe import trajectory_comparison
from scripts.write_asd_suite_report import assessed

root = Path.cwd()
base = root/"results/asd-rmsnorm-validation"
analysis = json.loads((base/"analysis.json").read_text())
rows = [r for i in (0, 1) for r in read_rows(base/f"part-{i}/rows.jsonl")]
refs = [r for i in (0, 1) for r in read_rows(base/f"part-{i}/references.jsonl")]
first = {(r["request_id"], r["method"], r["profile"]): r for r in (*refs, *[r for r in rows if r["repeat"] == 0])}
pairs = [(r, first[q, m, "rmsnorm-inductor"]) for (q, m, p), r in first.items() if p == "eager"]
stripped = lambda r: [{k: v for k, v in rd.items() if k != "regret"} for rd in r["rounds"]]
changes, quality = [], []
for a, b in pairs:
    comparison = trajectory_comparison(a, b)
    if not comparison["token_equal"]:
        changes.append(dict(request_id=a["request_id"], method=a["method"], comparison=comparison,
            eager_tokens=len(a["token_ids"]), compiled_tokens=len(b["token_ids"]),
            eager_finish=a["finish_reason"], compiled_finish=b["finish_reason"],
            quality_eager=a["quality"], quality_compiled=b["quality"]))
    if a["quality"]["correct"] != b["quality"]["correct"] or a["quality"].get("secondary_correct") != b["quality"].get("secondary_correct") or a["quality"].get("f1") != b["quality"].get("f1"):
        quality.append(dict(request_id=a["request_id"], method=a["method"], quality_eager=a["quality"],
                            quality_compiled=b["quality"], both_assessed=assessed(a) and assessed(b)))
workers = []
for i in (0, 1):
    info = json.loads((base/f"part-{i}/provenance.json").read_text())
    c, warm = info["compiler"], info["compiler_after_warmup"]
    timed = list(read_rows(base/f"part-{i}/rows.jsonl"))
    ar = list(read_rows(base/f"part-{i}/references.jsonl"))
    extra = sum(r["graphs_after"]-r["graphs_before"] for r in (*timed, *ar))
    assert extra == c["graph_count"]-warm["graph_count"]
    times = {(r["request_id"], r["method"], r["repeat"], r["profile"]): r["latency_seconds"] for r in timed}
    saving = statistics.mean(t-times[q, m, n, "rmsnorm-inductor"] for (q, m, n, p), t in times.items() if p == "eager")
    unchanged = all(a["token_ids"] == b["token_ids"] for a, b in pairs if a["request_id"] in set(info["task_ids"]))
    workers.append(dict(shard=i, inventory_by_owner=dict(Counter(x["region"].split(".")[0] for x in c["inventory"])),
        warmup_wall_seconds=info["warmup_wall_seconds"], controller_seconds=info["controller_setup_seconds"],
        outer_compile_seconds=c["outer_compile_seconds"], backend_seconds=c["backend_seconds"],
        graphs_after_warmup=warm["graph_count"], final_graphs=c["graph_count"], online_compile_events=extra,
        cache_bytes=info["cache_bytes"], mean_paired_saving_seconds=saving,
        conditional_amortization_requests=(c["outer_compile_seconds"]+info["controller_setup_seconds"])/saving
            if extra == 0 and unchanged and saving > 0 else None))
result = dict(stage="postrun-descriptive-only", workers=workers, output_changes=changes, quality_changes=quality,
    cross_profile_pairs=len(pairs), text_matches=sum(a["text"] == b["text"] for a, b in pairs),
    finish_reason_matches=sum(a["finish_reason"] == b["finish_reason"] for a, b in pairs),
    ledger_matches=sum(a["asd"] == b["asd"] for a, b in pairs),
    rounds_equal_without_regret=sum(stripped(a) == stripped(b) for a, b in pairs),
    eager_rounds=sum(len(a["rounds"]) for a, b in pairs),
    max_spent_difference=max(abs((a["asd"] or {}).get("spent", 0)-(b["asd"] or {}).get("spent", 0)) for a, b in pairs),
    truncations=[dict(request_id=q, method=m, profile=p, quality=r["quality"]) for (q, m, p), r in first.items() if r["finish_reason"] != "eos"],
    peak_allocated_bytes={p: max(r["peak_gpu_bytes"] for r in (*rows, *refs) if r["profile"] == p) for p in ("eager", "rmsnorm-inductor")},
    compiled_switch_median_seconds=statistics.median(r["profile_switch_seconds"] for r in rows if r["profile"] != "eager"),
    cpu_calls={p: {k: sum(r.get("cpu_execution", {}).get(k, 0) for r in rows if r["profile"] == p)
                   for k in ("fast_calls", "fallback_calls")} for p in ("eager", "rmsnorm-inductor")},
    note="Descriptive, not new independent samples. Amortization is only estimated for unchanged outputs and zero online recompiles; outer includes backend, warmup wall includes inference. No production ROI or universal losslessness claim.")
write(base/"interpretation.json", result)
print(json.dumps(dict(workers=workers, output_changes=len(changes), quality_changes=len(quality),
                      ledger_matches=result["ledger_matches"], rounds_equal_without_regret=result["rounds_equal_without_regret"],
                      max_spent_difference=result["max_spent_difference"], peak=result["peak_allocated_bytes"], cpu_calls=result["cpu_calls"])))
