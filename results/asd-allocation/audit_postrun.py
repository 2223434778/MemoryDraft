"""Postrun proof of frozen source, actual KV frontiers and preserved prior evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from memorydraft.evaluation.runtime_migration import read_rows
from scripts.run_asd_allocation_probe import STATE_ROUNDS, WINDOW
from scripts.write_asd_allocation_report import audit_resources

root = Path.cwd()
base = root/"results/asd-allocation"
freeze = "cea9ebb92db4879144b79b0338cd1272b38b0a35"
sha = lambda b: hashlib.sha256(b).hexdigest()
git = lambda *args: subprocess.check_output(["git", *args], cwd=root)
manifest = json.loads((base/"ASD_ALLOCATION_SOURCE_MANIFEST.json").read_text())
execution = json.loads((base/"execution-verification.json").read_text())
assert manifest["commit"] == execution["commit"] == freeze
assert manifest["source_sha256"] == sha(json.dumps(manifest["files"], sort_keys=True).encode())
assert execution["status"] == "complete"
assert execution["files_verified_before"] == execution["files_verified_after"] == len(manifest["files"]) == 1491
entries = []
for entry in git("ls-tree", "-rz", freeze).split(b"\0"):
    if entry:
        meta, name = entry.split(b"\t", 1)
        entries.append((name.decode(), meta.split()[2].decode()))
raw = subprocess.check_output(["git", "cat-file", "--batch"], cwd=root,
                              input="".join(oid+"\n" for _, oid in entries).encode())
position = 0
for name, oid in entries:
    end = raw.index(b"\n", position)
    header = raw[position:end].split()
    assert header[:2] == [oid.encode(), b"blob"]
    size = int(header[2])
    data = raw[end+1:end+1+size]
    assert sha(data) == manifest["files"][name], name
    position = end+size+2
frozen = ["memorydraft/backends/dflash_reuse.py", "memorydraft/backends/allocation.py",
          "scripts/run_asd_allocation_probe.py", "scripts/run_asd_allocation_pipeline.py",
          "scripts/write_asd_allocation_report.py", "tests/test_allocation.py", "tests/test_asd_allocation_report.py",
          "docs/ASD_ALLOCATION_PROTOCOL.md"]
for n in frozen:
    assert (root/n).read_bytes().replace(b"\r\n", b"\n") == git("show", freeze+":"+n), n
protected, infos, counts, records = {}, [], [], []
for i in (0, 1):
    p = base/f"part-{i}"
    info = json.loads((p/"provenance.json").read_text())
    assert info["status"] == "complete" and info["original_unchanged"] and info["split"] == "development"
    assert info["counts"]["sources"] == 6 and info["counts"]["requests"] == 72
    for name, digest in info["original_sha256"].items():
        assert name not in protected or protected[name] == digest
        protected[name] = digest
    source = {r["request_id"]: r for r in read_rows(p/"sources.jsonl")}
    branches = list(read_rows(p/"branches.jsonl"))
    expected = {(q, n, a, repeat) for q, r in source.items() for n in STATE_ROUNDS if n < len(r["rounds"])
                for a in ("native", "half", "verify-half", "target") for repeat in range(3)}
    actual = {(r["request_id"], r["source_round"], r["action"], r["repeat"]) for r in branches}
    assert actual == expected and len(branches) == len(expected)
    for r in branches:
        original, n = source[r["request_id"]], r["source_round"]
        boundary = original["rounds"][n]["generated_before"]
        assert boundary == r["boundary"] and r["max_new_tokens"] == boundary+WINDOW
        assert r["token_ids"][:boundary] == original["token_ids"][:boundary]
        assert r["continuation"] == r["token_ids"][boundary:]
        assert 0 < len(r["continuation"]) <= WINDOW
        assert [{k: v for k, v in x.items() if k != "allocation"} for x in r["rounds"][:n]] == original["rounds"][:n]
        assert all(x["allocation"] == "native" for x in r["rounds"][n+1:])
        assert r["actual_action"] == r["rounds"][n]["allocation"]
    infos.append(info)
    records.extend([*source.values(), *branches, *read_rows(p/"requests.jsonl")])
    counts.append(info["counts"])
legacy = []
for name, digest in protected.items():
    raw = (root/name).read_bytes()
    if sha(raw) != digest:
        assert name in {"results/research-audit/model_preflight.json", "results/research-audit/pilot_summary.json"}
        assert sha(raw.replace(b"\r\n", b"\n")) == digest, name
        legacy.append(name)
prior = []
for name, commit in (("asd-suite-stage1", "dcfc01d"), ("asd-block-preflight", "2a4ab50"),
                     ("asd-block-suite", "d72f6b6"), ("asd-execution", "1e56ecb"), ("asd-cpu", "38ce3b3"),
                     ("asd-compile", "4f3aa2a"), ("asd-rmsnorm", "841e0e5"), ("asd-rmsnorm-suite", "e958532"),
                     ("asd-rmsnorm-validation", "9b8176e"), ("rmsnorm-runtime", "7991557")):
    publication = json.loads((root/f"results/{name}/publication.json").read_text())
    for n, digest in publication["files"].items():
        assert sha(git("show", commit+":"+n)) == digest, n
    prior.append(dict(name=name, commit=git("rev-parse", commit).decode().strip(), files=len(publication["files"])))
refs = {ref: git("rev-parse", ref).decode().strip() for ref in ("main", "codex/infra-interview-v0.3", "infra-v0.3-interview^{}")}
assert set(refs.values()) == {"27e36199f8a65acdb9a9c42ccb744ddfa9a9402b"}
path = root/"artifacts/asd-allocation-results.zip"
with zipfile.ZipFile(path) as z:
    for n in z.namelist():
        assert (base/n).read_bytes() == z.read(n), n
result = dict(status="complete", source_commit=freeze, source_files=len(entries), frozen_files=frozen,
              protected_results=len(protected), legacy_crlf_normalized=legacy, prior_publications=prior,
              infra_refs=refs, counts=counts, resources=audit_resources(infos, records),
              result_zip=dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes())),
              actual_KV_prefixes_and_native_future_verified=True)
(base/"postrun-audit.json").write_bytes((json.dumps(result, indent=2)+"\n").encode())
print(json.dumps(result))
