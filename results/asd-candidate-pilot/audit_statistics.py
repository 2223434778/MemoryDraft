"""Compare frozen Python 3.11/server and 3.13/local pure statistics."""
import json
from pathlib import Path

base = Path.cwd()/"results/asd-candidate-pilot"
server = json.loads((base/"statistics-rebuild-server.json").read_text())
local = json.loads((base/"analysis.json").read_text())
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
assert server["expand_candidates"] == local["expand_candidates"]
result = dict(status="complete", all_nonfloating_fields_exact=True,
              maximum_float_difference=max((d["absolute"] for d in differences), default=0), differences=differences)
(base/"statistics-audit.json").write_bytes((json.dumps(result, indent=2)+"\n").encode())
infos = [json.loads((base/f"part-{i}/provenance.json").read_text()) for i in (0, 1)]
print(json.dumps(dict(differences=len(differences), maximum=result["maximum_float_difference"],
    resources=[dict(card=i, graphs=p["compiler"]["graph_count"], cache_bytes=p["cache_bytes"],
                    backend_seconds=p["compiler"]["backend_seconds"],
                    events=[{k: v for k, v in e.items() if k in {"status", "backend_seconds", "compile_seconds", "seconds"}} for e in p["compiler"]["events"]])
               for i, p in enumerate(infos)],
    quality=[dict(id=s["request_id"], quality=s["methods"]["native"]["quality"]) for s in local["samples"]])))
