"""Postrun CPU-only proof of private feature/adapter files and analysis revision."""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import datetime
import zipfile

import torch
from safetensors.torch import load_file

root = Path.cwd()
base = root/"results/raw/asd-adaptation"
sha = lambda b: hashlib.sha256(b).hexdigest()
execution = json.loads((base/"execution-verification.json").read_text())
assert execution["status"] == "complete"
examples = [json.loads(line) for i in (0, 1) for line in (base/f"collect-{i}/examples.jsonl").read_text().splitlines()]
paths = {}
for row in examples:
    p = root/row["feature_path"]
    assert p.resolve().is_relative_to(root/"artifacts")
    if row["feature_path"] not in paths:
        assert sha(p.read_bytes()) == row["feature_sha256"]
        item = torch.load(p, map_location="cpu", weights_only=True)
        assert item["context"].dtype == torch.float32 and item["context"].shape == (1, row["context_tokens"], 20480)
        assert item["labels"][0].tolist() == row["teacher_tokens"] and item["prefix_ids"][0].tolist() == row["prefix_ids"]
        assert item["anchor"].item() == row["prefix_ids"][-1]
        paths[row["feature_path"]] = p.stat().st_size
assert sum(paths.values()) == execution["feature_cache_bytes"] <= 16*2**30
adapters = {}
for arm in ("ordinary", "asd-prefix"):
    meta = json.loads((base/f"train-{arm}/provenance.json").read_text())
    p = root/meta["adapter_path"]
    assert sha(p.read_bytes()) == meta["adapter_sha256"]
    assert sha((p.parent/"final.safetensors").read_bytes()) == meta["final_sha256"]
    state = load_file(str(p))
    assert {n:list(t.shape) for n, t in state.items()} == meta["tensor_shapes"]
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in state.values())
    assert sum(t.numel() for t in state.values()) == meta["trainable_parameters"]
    adapters[arm] = dict(bytes=p.stat().st_size, sha256=meta["adapter_sha256"], tensors=len(state), numel=meta["trainable_parameters"])
raw = [json.loads(line) for i in (0, 1) for line in (base/f"eval-{i}/rows.jsonl").read_text().splitlines()]
from scripts.write_asd_adaptation_report import analyze as original_analyze
original = original_analyze(raw)
fixed_path = root/"artifacts/analysis-f8e6803.py"
spec = importlib.util.spec_from_file_location("fixed_adaptation_analysis", fixed_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
link = root/"results/asd-adaptation"
if not link.exists():
    link.symlink_to(base, target_is_directory=True)
assert link.resolve() == base
with (root/"artifacts/report-rebuild-server.log").open("w") as stream, contextlib.redirect_stdout(stream):
    module.main()
fixed = json.loads((base/"analysis.json").read_text())
changes = []
def compare(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            compare(a[k], b[k], path+"/"+k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path+"/"+str(i))
    elif a != b:
        assert any(k in path for k in ("both_reach_128", "common_full_128")), path
        changes.append(dict(path=path, execution_version=a, corrected_analysis=b))
compare(original, {k:fixed[k] for k in original})
correction = dict(execution_commit=execution["commit"], analysis_commit="f8e68035f6ae3f01e8e0fa531260a248e03fe7f8",
                  original_script_sha256=sha((root/"scripts/write_asd_adaptation_report.py").read_bytes()),
                  corrected_script_sha256=sha(fixed_path.read_bytes()), changes=changes)
(root/"artifacts/statistics-correction.json").write_text(json.dumps(correction, indent=2)+"\n")
(root/"artifacts/statistics-rebuild-server.json").write_text(json.dumps(fixed, indent=2)+"\n")
audit = dict(status="complete", unique_feature_files=len(paths), feature_bytes=sum(paths.values()), adapters=adapters,
             feature_checksums_and_shapes_verified=True, private_tensor_files_not_published=True)
(root/"artifacts/server-artifact-audit.json").write_text(json.dumps(audit, indent=2)+"\n")
z = root.parent/"asd-adaptation-results.zip"
with zipfile.ZipFile(z) as archive:
    assert archive.testzip() is None
status = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True)
assert all(int(s.split(',')[1]) < 100 for s in status.splitlines()), status
meta = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), execution=execution,
            zip_bytes=z.stat().st_size, zip_sha256=sha(z.read_bytes()), gpu_after=status,
            analysis_commit=correction["analysis_commit"], analysis_script_sha256=correction["corrected_script_sha256"])
(root/"completion.json").write_text(json.dumps(meta, indent=2)+"\n")
print(json.dumps(dict(completion=meta, artifact_audit=audit, analysis_changes=len(changes),
                     all_safe=fixed["all_safe"], domain={k:v for k,v in fixed["scopes"]["gsm8k"].items() if k.startswith("natural")})))
