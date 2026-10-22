"""Posthoc descriptive mechanism counts; not a new selection rule or timer."""
from collections import Counter
import json
from pathlib import Path

from memorydraft.evaluation.runtime_migration import read_rows
from scripts.run_asd_candidate_probe import METHODS

root = Path.cwd()
base = root/"results/asd-candidate-pilot"
analysis = json.loads((base/"analysis.json").read_text())
rows = [r for i in (0, 1) for r in read_rows(base/f"part-{i}/rows.jsonl") if r["repeat"] == 0]
summary = {}
for method in METHODS:
    selected = [r for r in rows if r["candidate"] == method]
    rounds = [x for r in selected for x in r["rounds"]]
    repairs = [x for x in rounds if x["lookup"] == "suffix-repair"]
    tree = [x for x in rounds if "tree_nodes" in x]
    summary[method] = dict(requests=len(selected), output_tokens=sum(len(r["token_ids"]) for r in selected),
        finish_reasons=dict(Counter(r["finish_reason"] for r in selected)),
        target_calls=len(rounds), draft_calls=sum(x["draft_forward"] for x in rounds),
        verify_nodes=sum(x.get("tree_nodes", x["proposed"]+1) for x in rounds),
        raw_advance=sum(x["advance"] for x in rounds),
        advance_after_eos_or_cap=sum(max(0, 1+sum(x["advance"] for x in r["rounds"])-len(r["token_ids"])) for r in selected),
        tree_gain_histogram=dict(sorted(Counter(x.get("tree_gain", 0) for x in tree).items())),
        tree_native_accept_histogram=dict(sorted(Counter(x["native_accepted"] for x in tree).items())),
        branch_rounds=sum(any(i != j for i, j in enumerate(x["tree_path"])) for x in tree),
        repair_accept_histogram=dict(sorted(Counter(x["accepted"] for x in repairs).items())),
        repair_calls=len(repairs), repair_accepted=sum(x["accepted"] for x in repairs),
        all_quality_dicts_equal_old=all(r["quality"] == r["full_reference_quality"] for r in selected))
per_request = []
for sample in analysis["samples"]:
    native = sample["methods"]["native"]
    per_request.append(dict(request_id=sample["request_id"], benchmark=sample["benchmark"], methods={
        method: dict(latency_ratio=x["latency_ratio"], seconds=x["seconds"],
                     target_call_ratio=x["target_calls"]/native["target_calls"],
                     amortized_wall_per_call_ratio=(x["seconds"]/x["target_calls"])/(native["seconds"]/native["target_calls"]),
                     verify_nodes=x["verify_nodes"], tree_gain=x["tree_gain"],
                     repair_calls=x["repair_calls"], repair_zero=x["repair_zero"], repair_accepted=x["repair_accepted"])
        for method, x in sample["methods"].items()}))
result = dict(posthoc=True, methods=summary, per_request=per_request,
              notes=["Trace raw advance/tree_gain can include work beyond EOS or cap; output tokens are separately exact.",
                     "Wall/call includes prefill, drafting, decoding and every online cost; it is not isolated Target kernel latency.",
                     "No threshold, method or task was changed using these descriptive counts."])
(base/"interpretation.json").write_bytes((json.dumps(result, indent=2)+"\n").encode())
print(json.dumps(summary))
