"""Postrun interpretation only; frozen primary statistics are never changed."""
from collections import Counter
import json
from pathlib import Path
import statistics

from memorydraft.evaluation.runtime_migration import read_rows

root = Path.cwd()
base = root/"results/asd-allocation"
analysis = json.loads((base/"analysis.json").read_text())
requests = [r for i in (0, 1) for r in read_rows(base/f"part-{i}/requests.jsonl")]
branches = [r for i in (0, 1) for r in read_rows(base/f"part-{i}/branches.jsonl")]
sources = [r for i in (0, 1) for r in read_rows(base/f"part-{i}/sources.jsonl")]
unavailable = [r for i in (0, 1) for r in read_rows(base/f"part-{i}/unavailable.jsonl")]
first = {(r["request_id"], r["action"]): r for r in requests if r["repeat"] == 0}
decomposition = []
for sample in analysis["request"]["samples"]:
    q = sample["request_id"]
    native_calls = 1+len(first[q, "native"]["rounds"])
    for action in ("half", "verify-half", "AR"):
        row = first[q, action]
        calls = len(row["token_ids"]) if action == "AR" else 1+len(row["rounds"])
        ratio = sample["costs"][action]
        decomposition.append(dict(request_id=q, action=action, time_ratio=ratio,
                                  target_call_ratio=calls/native_calls,
                                  amortized_time_per_call_ratio=ratio/(calls/native_calls)))
short = []
for s in analysis["branch"]["samples"]:
    if s["benchmark"] not in {"squad", "mbpp"}:
        continue
    q, n = s["request_id"], s["source_round"]
    arms = {}
    for action in s["costs"]:
        rows = [r for r in branches if r["request_id"] == q and r["source_round"] == n and r["action"] == action]
        row = rows[0]
        current = row["rounds"][n]
        arms[action] = dict(local_seconds_mean=statistics.mean(r["local_seconds"] for r in rows),
                            paired_seconds=[r["local_seconds"] for r in sorted(rows, key=lambda r: r["repeat"])],
                            actual_action=row["actual_action"], current_advance=current["advance"],
                            target_query_tokens=current["proposed"]+1, continuation_tokens=len(row["continuation"]),
                            future_rounds=len(row["rounds"])-n-1, draft_calls=sum(x["draft_forward"] for x in row["rounds"][n:]))
    short.append(dict(request_id=q, source_round=n, features=s["features"], oracle=min(s["costs"], key=s["costs"].get), arms=arms))
result = dict(stage="postrun-descriptive-only", generations=dict(sources=len(sources), requests=len(requests), branches=len(branches), warmup=10),
              primary_generations=len(sources)+len(requests)+len(branches), unavailable_states=unavailable,
              oracle_action_counts={stage: dict(Counter(min(s["costs"], key=s["costs"].get) for s in analysis[stage]["samples"])) for stage in ("request", "branch")},
              numerical_request_mismatches=sum(not r["token_match"] for r in requests),
              forced_native_target_actions=sum(r["action"] == "target" and r["actual_action"] != "target" for r in branches),
              request_decomposition=decomposition, interesting_local_states=short,
              note="Per-request factorization includes all request costs per Target call, not isolated kernel latency. Local state costs cannot be summed into a complete policy gain; source replay/prefill are excluded. EOS/short outputs remain matched across actions.")
(base/"interpretation.json").write_bytes((json.dumps(result, indent=2)+"\n").encode())
print(json.dumps(dict(counts=result["generations"], primary=result["primary_generations"], missing=len(unavailable),
    medians={action: {key: statistics.median(r[key] for r in decomposition if r["action"] == action)
                     for key in ("target_call_ratio", "amortized_time_per_call_ratio")}
             for action in ("half", "verify-half", "AR")}, short=short)))
