"""No generation: source, prior evidence, paired inputs and publication audit."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from memorydraft.evaluation.long_context import load_data
from scripts.write_asd_long_context_report import analyze

root = next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
base = root/"results/asd-long-context"
sha = lambda data:hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_text(encoding="utf-8"))
def blobs(commit):
    entries = []
    for entry in filter(None,subprocess.check_output(["git","ls-tree","-rz",commit],cwd=root).split(b"\0")):
        meta,name = entry.split(b"\t",1)
        mode,kind,oid = meta.split()
        assert mode in (b"100644",b"100755") and kind == b"blob"
        entries.append((name.decode(),oid.decode()))
    raw = subprocess.check_output(["git","cat-file","--batch"],input="".join(oid+"\n" for _,oid in entries).encode(),cwd=root)
    offset,contents = 0,{}
    for name,oid in entries:
        end = raw.index(b"\n",offset);header = raw[offset:end].split()
        assert header[:2] == [oid.encode(),b"blob"]
        size = int(header[2]);contents[name] = raw[end+1:end+1+size];offset = end+size+2
    return contents
manifest = read(base/"ASD_LONG_CONTEXT_SOURCE_MANIFEST.json")
canonical = blobs(manifest["commit"])
assert {n:sha(data) for n,data in canonical.items()} == manifest["files"]
assert sha(json.dumps(manifest["files"],sort_keys=True).encode()) == manifest["source_sha256"]
legacy,protected,frozen = [],0,0
for n,data in canonical.items():
    if n.startswith("results/"):
        actual = (root/n).read_bytes()
        if actual != data:
            assert n in {"results/research-audit/model_preflight.json","results/research-audit/pilot_summary.json"}
            assert actual.replace(b"\r\n",b"\n") == data;legacy.append(n)
        protected += 1
    if (n.startswith("memorydraft/") or n.startswith("data/workloads/asd-long-context/")
            or n.startswith("scripts/") and "long_context" in n or n in {"tests/test_long_context.py","docs/ASD_LONG_CONTEXT_PROTOCOL.md"}):
        assert sha((root/n).read_bytes()) == manifest["files"][n],n
        frozen += 1
prior = blobs("1bcd40e1d6ef37756dbdd1e633039d991400bad0")
publication = json.loads(prior["results/asd-runtime/publication.json"])
assert all(sha(prior[n]) == h for n,h in publication["files"].items())
execution = read(base/"execution-verification.json")
assert execution["status"] == "complete" and execution["files_verified_before"] == execution["files_verified_after"] == len(canonical)
assert execution["original_adapter_unchanged"]
tasks,items,info = load_data(root)
levels = read(base/"admission.json")["levels"]
items = [r for r in items if r["level"] in levels]
rows,refs,workers = [],[],[]
for variant in ("base","position"):
    for shard in (0,1):
        directory = base/f"eval-{variant}-{shard}"
        meta = read(directory/"provenance.json");workers.append(meta)
        rows.extend(json.loads(line) for line in (directory/"rows.jsonl").read_text(encoding="utf-8").splitlines())
        if variant == "base":refs.extend(json.loads(line) for line in (directory/"references.jsonl").read_text(encoding="utf-8").splitlines())
refmap = {r["request_id"]:r for r in refs}
assert all(r["snapshot_id"] == refmap[r["request_id"]]["snapshot_id"] for r in rows)
recomputed = analyze(rows,refs,items,tasks)
analysis = read(base/"analysis.json")
assert all(analysis[k] == v for k,v in recomputed.items())
assert (base/"analysis.json").read_bytes() == (base/"analysis-server.json").read_bytes()
archive_path = root/"artifacts/asd-long-context-results.zip"
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    assert all((base/n).read_bytes() == archive.read(n) for n in archive.namelist())
    members = len(archive.namelist())
infra = {n:subprocess.check_output(["git","rev-parse",n],cwd=root).decode().strip() for n in ("main","codex/infra-interview-v0.3","infra-v0.3-interview^{}")}
assert set(infra.values()) == {"27e36199f8a65acdb9a9c42ccb744ddfa9a9402b"}
result = dict(status="complete",execution_commit=manifest["commit"],canonical_git_blobs=len(canonical),frozen_files=frozen,
    previous_result_files=protected,previous_publication_files=len(publication["files"]),legacy_crlf=legacy,
    matrix_rows=len(rows),ar_references=len(refs),all_strict_match_ar=recomputed["all_strict_match_ar"],formal_new_graphs=recomputed["formal_new_graphs"],
    admitted=levels,paired_snapshots=True,server_local_analysis_equal=True,infra_refs=infra,
    archive_bytes=archive_path.stat().st_size,archive_sha256=sha(archive_path.read_bytes()),archive_members=members)
(base/"postrun-audit.json").write_bytes((json.dumps(result,indent=2)+"\n").encode())
print(json.dumps(result))
