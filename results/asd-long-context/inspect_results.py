"""Descriptive attribution only; never replaces the prespecified full matrix."""
import json
from pathlib import Path

base = Path(__file__).resolve().parent
analysis = json.loads((base/"analysis.json").read_text(encoding="utf-8"))
result = {}
for level,arms in analysis["summary"].items():
    asd = arms["ASD"]
    selected = [r for r in analysis["cases"] if str(r["level"]) == level and all(
        r["arms"][arm+"/prefix"]["tokens"] == 128 for arm in ("DFlash","ASD","position-ASD"))]
    result[level] = dict(ttft_fraction_of_mean_asd_wall=asd["mean_ttft_seconds"]/(asd["mean_ttft_seconds"]+asd["mean_post_first_seconds"]),
        common_prefix_rounds={arm:sum(r["arms"][arm+"/prefix"]["rounds"] for r in selected) for arm in ("ASD","position-ASD")},
        cold_formal_worker_first_requests=[])
for variant in ("base","position"):
    for shard in (0,1):
        directory = base/f"eval-{variant}-{shard}"
        cold = json.loads((directory/"warmup.jsonl").read_text(encoding="utf-8").splitlines()[0])
        result["512"]["cold_formal_worker_first_requests"].append(dict(variant=variant,shard=shard,
            method=cold["method"],seconds=cold["latency_seconds"],ttft_seconds=cold["ttft_seconds"],counters=cold["counters"]))
(base/"interpretation.json").write_bytes((json.dumps(result,indent=2)+"\n").encode())
print(json.dumps(result))
