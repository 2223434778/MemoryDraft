"""Cross-Python pure-statistics proof and explicitly posthoc request bootstrap."""
import json
from pathlib import Path
import random
import statistics

base = Path.cwd()/"results/asd-adaptation"
server = json.loads((base/"statistics-rebuild-server.json").read_text(encoding="utf-8"))
local = json.loads((base/"analysis.json").read_text(encoding="utf-8"))
differences = []
def compare(a, b, path=""):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            compare(a[k], b[k], path+"/"+k)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path+"/"+str(i))
    elif isinstance(a, float) and isinstance(b, float):
        if a != b:
            assert abs(a-b) < 1e-12, path
            differences.append(dict(path=path, server=a, local=b, absolute=abs(a-b)))
    else:
        assert a == b, path
compare(server, local)
result = dict(status="complete", all_nonfloating_fields_exact=True,
              maximum_float_difference=max((d["absolute"] for d in differences), default=0), differences=differences)
(base/"statistics-audit.json").write_bytes((json.dumps(result, indent=2)+"\n").encode())
bootstrap = {}
for policy in ("DFlash", "ASD"):
    for stage in ("natural", "prefix"):
        for versus, field in (("base", "latency_ratio"), ("ordinary", "ordinary_ratio")):
            values = [s["arms"][f"{stage}/asd-prefix/{policy}"][field] for s in local["samples"]
                      if s["benchmark"] == "gsm8k" and (stage == "natural" or s["arms"][f"{stage}/asd-prefix/{policy}"]["both_reach_128"])]
            rng = random.Random("adaptation-bootstrap:"+policy+stage+versus)
            draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
            bootstrap[f"{stage}/{policy}/vs-{versus}"] = dict(requests=len(values), ratio=statistics.mean(values),
                                                           percentile_95=[draws[249], draws[9749]])
summary = dict(posthoc=True, resampling_unit="request, after averaging its timing repeats", draws=10000,
               note="Descriptive bootstrap of observed development requests, not independent confirmation or a new gate.", gsm8k=bootstrap)
(base/"bootstrap.json").write_bytes((json.dumps(summary, indent=2)+"\n").encode())
print(json.dumps(dict(statistics_differences=len(differences), maximum=result["maximum_float_difference"], bootstrap=bootstrap)))
