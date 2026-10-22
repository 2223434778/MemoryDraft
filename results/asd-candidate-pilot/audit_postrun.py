"""Readonly proof of canonical source, old evidence and candidate trace frontiers."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from memorydraft.evaluation.runtime_migration import read_rows
from scripts.write_asd_allocation_report import audit_resources
from scripts.write_asd_candidate_report import analyze

root = Path.cwd()
base = root/"results/asd-candidate-pilot"
freeze = "2949b9ed7dda32678ef0484194554cbc43365d74"
sha = lambda b: hashlib.sha256(b).hexdigest()
git = lambda *args: subprocess.check_output(["git", *args], cwd=root)
manifest = json.loads((base/"ASD_CANDIDATE_SOURCE_MANIFEST.json").read_text())
execution = json.loads((base/"execution-verification.json").read_text())
assert manifest["commit"] == execution["commit"] == freeze
assert manifest["source_sha256"] == sha(json.dumps(manifest["files"], sort_keys=True).encode())
assert execution["status"] == "complete"
assert execution["files_verified_before"] == execution["files_verified_after"] == len(manifest["files"]) == 1540
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
    assert sha(raw[end+1:end+1+size]) == manifest["files"][name], name
    position = end+size+2
frozen = ["memorydraft/backends/dflash_reuse.py", "memorydraft/backends/draft_tree.py",
          "memorydraft/backends/candidate_repair.py", "scripts/run_asd_candidate_probe.py",
          "scripts/run_asd_candidate_pipeline.py", "scripts/write_asd_candidate_report.py",
          "tests/test_draft_tree.py", "tests/test_candidate_repair.py", "tests/test_asd_candidate_report.py",
          "docs/ASD_CANDIDATE_PROTOCOL.md"]
for n in frozen:
    assert (root/n).read_bytes().replace(b"\r\n", b"\n") == git("show", freeze+":"+n), n
protected, infos, rows = {}, [], []
for i in (0, 1):
    p = base/f"part-{i}"
    info = json.loads((p/"provenance.json").read_text())
    assert info["status"] == "complete" and info["original_unchanged"] and info["split"] == "development"
    assert info["count"] == 108 and len(info["task_ids"]) == 6
    for name, digest in info["original_sha256"].items():
        assert name not in protected or protected[name] == digest
        protected[name] = digest
    infos.append(info)
    rows.extend(read_rows(p/"rows.jsonl"))
for row in rows:
    assert row["max_new_tokens"] == 4096 and row["scope"] == "pilot"
    assert row["execution_profile"] == "rmsnorm-inductor"
    for index, r in enumerate(row["rounds"]):
        frontier = row["prompt_tokens"]+r["generated_before"]-1
        assert r["draft_cache_before"]+r["pending_before"] == frontier
        assert r["target_cache_after"] == frontier+r["advance"]
        assert r["advance"] == r["accepted"]+1
        assert r["pending_after"] == r["advance"]+(0 if r["draft_forward"] else r["pending_before"])
        if "tree_path" in r:
            n = r["proposed"]+1
            parents = list(range(-1, n-1))
            length = 4 if row["candidate"] == "tree32" else 8
            if row["candidate"] != "packed-linear":
                for depth in range(1, min(4, n-1)+1):
                    parent = depth-1
                    for _ in range(min(length, n-depth)):
                        parents.append(parent)
                        parent = len(parents)-1
            path = r["tree_path"]
            assert len(parents) == r["tree_nodes"] and len(path) == r["advance"] and path[0] == 0
            assert all(parents[b] == a for a, b in zip(path, path[1:]))
            assert r["tree_gain"] == r["accepted"]-r["native_accepted"] >= 0
        if r["lookup"] == "suffix-repair":
            assert not r["draft_forward"] and 2 <= r["proposed"] <= int(row["candidate"][6:])
            if index+1 < len(row["rounds"]):
                assert row["rounds"][index+1]["draft_forward"]
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
                     ("asd-rmsnorm-validation", "9b8176e"), ("rmsnorm-runtime", "7991557"), ("asd-allocation", "1e95580")):
    publication = json.loads((root/f"results/{name}/publication.json").read_text())
    for n, digest in publication["files"].items():
        assert sha(git("show", commit+":"+n)) == digest, n
    prior.append(dict(name=name, commit=git("rev-parse", commit).decode().strip(), files=len(publication["files"])))
refs = {ref: git("rev-parse", ref).decode().strip() for ref in ("main", "codex/infra-interview-v0.3", "infra-v0.3-interview^{}")}
assert set(refs.values()) == {"27e36199f8a65acdb9a9c42ccb744ddfa9a9402b"}
path = root/"artifacts/asd-candidate-pilot-results.zip"
with zipfile.ZipFile(path) as z:
    for n in z.namelist():
        assert (base/n).read_bytes() == z.read(n), n
analysis = analyze(rows)
resources = audit_resources(infos, rows)
saved = json.loads((base/"analysis.json").read_text())
assert {k: v for k, v in saved.items() if k != "execution_audit"} == analysis
assert saved["execution_audit"] == resources
result = dict(status="complete", source_commit=freeze, source_files=len(entries), frozen_files=frozen,
              protected_results=len(protected), legacy_crlf_normalized=legacy, prior_publications=prior,
              infra_refs=refs, requests=len(rows), trace_frontiers_and_tree_paths_verified=True,
              all_candidate_outputs_match=all(analysis["globally_safe"].values()),
              resources=resources, result_zip=dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes())))
(base/"postrun-audit.json").write_bytes((json.dumps(result, indent=2)+"\n").encode())
print(json.dumps(result))
