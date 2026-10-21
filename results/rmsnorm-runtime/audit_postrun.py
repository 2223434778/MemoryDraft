"""Postrun verification of runtime source, profile compatibility and prior evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from scripts.run_rmsnorm_runtime_pipeline import check_results

root = Path.cwd()
base = root/"results/rmsnorm-runtime"
sha = lambda b: hashlib.sha256(b).hexdigest()
git = lambda *args: subprocess.check_output(["git", *args], cwd=root)
manifest = json.loads((base/"RMSNORM_RUNTIME_SOURCE_MANIFEST.json").read_text())
execution = json.loads((base/"execution-verification.json").read_text())
freeze = "52fe5be256a9982d4201c24a92b498a5e9cc0e2f"
assert manifest["commit"] == execution["commit"] == freeze
assert manifest["source_sha256"] == sha(json.dumps(manifest["files"], sort_keys=True).encode())
assert execution["status"] == "complete"
assert execution["files_verified_before"] == execution["files_verified_after"] == len(manifest["files"]) == 1453
entries = []
for entry in git("ls-tree", "-rz", freeze).split(b"\0"):
    if entry:
        meta, name = entry.split(b"\t", 1)
        entries.append((name.decode(), meta.split()[2].decode()))
raw = subprocess.check_output(["git", "cat-file", "--batch"], cwd=root,
                              input="".join(oid+"\n" for _, oid in entries).encode())
position, blobs = 0, {}
for name, oid in entries:
    end = raw.index(b"\n", position)
    header = raw[position:end].split()
    assert header[:2] == [oid.encode(), b"blob"]
    size = int(header[2])
    data = raw[end+1:end+1+size]
    position = end+size+2
    assert sha(data) == manifest["files"][name], name
    blobs[name] = data
core = ["memorydraft/backends/regional_compile.py", "memorydraft/runtime/engine.py", "memorydraft/runtime/config.py",
        "memorydraft/runtime/contracts.py", "memorydraft/runtime/benchmark.py", "configs/runtime-dflash-rmsnorm.json",
        "scripts/validate_infra.py", "scripts/run_rmsnorm_runtime_pipeline.py", "tests/test_runtime_rmsnorm.py"]
for name in core:
    assert (root/name).read_bytes().replace(b"\r\n", b"\n") == blobs[name], name
a = json.loads((base/"acceptance/acceptance.json").read_text())
assert a["engine"]["execution_profile"] == a["final_engine"]["execution_profile"] == "rmsnorm-inductor"
sources = {n.removeprefix("memorydraft/"): h for n, h in manifest["files"].items()
           if n.startswith("memorydraft/") and n.endswith((".py", ".json", ".txt"))}
assert a["source"]["files"] == sources
assert a["source"]["sha256"] == sha(json.dumps(sources, sort_keys=True).encode())
assert check_results(base, execution["checks"]["cache_bytes"]) == execution["checks"]
assert execution["validation_admission"] == json.loads(blobs["results/asd-rmsnorm-validation/analysis.json"])["admission"]
assert all(e["status"] == "complete" for e in a["compiler"]["events"])
prior = []
for name, commit in (("results/asd-suite-stage1/publication.json", "dcfc01d"),
                     ("results/asd-block-preflight/publication.json", "2a4ab50"),
                     ("results/asd-block-suite/publication.json", "d72f6b6"),
                     ("results/asd-execution/publication.json", "1e56ecb"),
                     ("results/asd-cpu/publication.json", "38ce3b3"),
                     ("results/asd-compile/publication.json", "4f3aa2a"),
                     ("results/asd-rmsnorm/publication.json", "841e0e5"),
                     ("results/asd-rmsnorm-suite/publication.json", "e958532"),
                     ("results/asd-rmsnorm-validation/publication.json", "9b8176e")):
    info = json.loads((root/name).read_text())
    for path, digest in info["files"].items():
        assert sha(git("show", commit+":"+path)) == digest, path
        if path.startswith("results/"):
            assert sha((root/path).read_bytes()) == digest, path
    prior.append(dict(path=name, commit=git("rev-parse", commit).decode().strip(), files=len(info["files"])))
refs = {ref: git("rev-parse", ref).decode().strip() for ref in ("main", "codex/infra-interview-v0.3", "infra-v0.3-interview^{}")}
assert set(refs.values()) == {"27e36199f8a65acdb9a9c42ccb744ddfa9a9402b"}
archives = []
for filename, prefix in (("rmsnorm-runtime-results.zip", base), ("rmsnorm-runtime-failed-tests-results.zip", base/"failed-tests")):
    path = root/"artifacts"/filename
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            assert (prefix/name).read_bytes() == z.read(name), name
    archives.append(dict(path=filename, bytes=path.stat().st_size, sha256=sha(path.read_bytes())))
tests = [s.attrib for s in ET.parse(base/"tests-server.xml").getroot()]
assert len(tests) == 1 and tests[0]["tests"] == "84" and tests[0]["failures"] == tests[0]["errors"] == "0"
failure = json.loads((base/"failed-tests/execution-verification.json").read_text())
assert failure["status"] == "failed" and not (base/"failed-tests/acceptance").exists()
rows = [json.loads(line) for line in (base/"acceptance/benchmark/rows.jsonl").read_text().splitlines()]
cli = json.loads((base/"cli.json").read_text())
result = dict(status="complete", source_commit=freeze, source_files=len(blobs), core_files=core,
              checks=execution["checks"], prior_publications=prior, infra_refs=refs, archives=archives,
              peak_allocated_bytes=max(r["peak_gpu_bytes"] for r in [*rows, *a["cold_start"], cli]),
              server_tests=tests, preserved_preflight_failure=True,
              scope="Engineering acceptance only; research performance remains the separate independent validation.")
(base/"postrun-audit.json").write_bytes((json.dumps(result, indent=2)+"\n").encode())
print(json.dumps(result))
