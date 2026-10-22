"""Task-local immutable source/result audit, with no token generation."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

root = next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
dest = root/"results/asd-runtime"
sha = lambda data:hashlib.sha256(data).hexdigest()
manifest = json.loads((dest/"ASD_RUNTIME_SOURCE_MANIFEST.json").read_text())
assert sha(json.dumps(manifest["files"],sort_keys=True).encode()) == manifest["source_sha256"]
def blobs(commit):
    entries = []
    for entry in filter(None,subprocess.check_output(["git","ls-tree","-rz",commit],cwd=root).split(b"\0")):
        metadata,name = entry.split(b"\t",1)
        mode,kind,oid = metadata.split()
        assert kind == b"blob" and mode in (b"100644",b"100755")
        entries.append((name.decode(),oid.decode()))
    raw = subprocess.check_output(["git","cat-file","--batch"],input="".join(oid+"\n" for _,oid in entries).encode(),cwd=root)
    offset,contents = 0,{}
    for name,oid in entries:
        end = raw.index(b"\n",offset)
        header = raw[offset:end].split()
        assert header[:2] == [oid.encode(),b"blob"]
        size = int(header[2])
        contents[name] = raw[end+1:end+1+size]
        offset = end+size+2
    return contents
canonical = blobs(manifest["commit"])
assert {n:sha(data) for n,data in canonical.items()} == manifest["files"]
protected,legacy = [],[]
for n,data in canonical.items():
    if not n.startswith("results/"):continue
    actual = (root/n).read_bytes()
    if actual != data:
        assert n in {"results/research-audit/model_preflight.json","results/research-audit/pilot_summary.json"}
        assert actual.replace(b"\r\n",b"\n") == data
        legacy.append(n)
    protected.append(n)
prior = blobs("3f21debf2fbf940d1ec4450ab7b22d1e91df4247")
publication = json.loads(prior["results/asd-adaptation-followup/publication.json"])
assert all(sha(prior[n]) == h for n,h in publication["files"].items())
analysis_manifest = json.loads((dest/"ANALYSIS_MANIFEST.json").read_text())
analysis_source = blobs(analysis_manifest["commit"])
assert all(sha(analysis_source[n]) == h == sha((root/n).read_bytes()) for n,h in analysis_manifest["files"].items())
analysis_changed = {n for n,h in analysis_manifest["files"].items() if n in manifest["files"] and h != manifest["files"][n]}
assert analysis_changed == {"scripts/run_asd_runtime_pipeline.py","scripts/validate_asd_runtime.py","scripts/write_asd_runtime_report.py","tests/test_asd_runtime_acceptance.py"}
frozen = {n:h for n,h in manifest["files"].items() if n not in analysis_changed and (n.startswith(("memorydraft/","configs/","examples/","tests/"))
          or n.startswith("scripts/") and "asd_runtime" in n or n in {"docs/ASD_RUNTIME_PROTOCOL.md"})}
source_crlf = []
for n,h in frozen.items():
    actual = (root/n).read_bytes()
    if sha(actual) != h:
        assert n in {"configs/dflash-manifest.json","configs/qwen3-download-manifest.json"}
        assert sha(actual.replace(b"\r\n",b"\n")) == h,n
        source_crlf.append(n)
archive_path = root/"artifacts/asd-runtime-results.zip"
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    assert all((dest/n).read_bytes() == archive.read(n) for n in archive.namelist())
infra = {name:subprocess.check_output(["git","rev-parse",name],cwd=root).decode().strip()
         for name in ("main","codex/infra-interview-v0.3","infra-v0.3-interview^{}")}
assert set(infra.values()) == {"27e36199f8a65acdb9a9c42ccb744ddfa9a9402b"}
result = dict(status="complete",execution_commit=manifest["commit"],source_sha256=manifest["source_sha256"],
    canonical_git_blobs=len(canonical),frozen_runtime_files=len(frozen),previous_result_files=len(protected),
    previous_publication_files=len(publication["files"]),previous_publication_commit="3f21debf2fbf940d1ec4450ab7b22d1e91df4247",
    legacy_crlf=legacy,archive_bytes=archive_path.stat().st_size,archive_sha256=sha(archive_path.read_bytes()),infra_refs=infra,
    analysis_commit=analysis_manifest["commit"],analysis_files_verified=len(analysis_manifest["files"]),analysis_changed=sorted(analysis_changed),
    original_zero_tolerance_gate_passed=False,source_checkout_crlf=source_crlf)
(dest/"postrun-audit.json").write_bytes((json.dumps(result,indent=2)+"\n").encode())
print(json.dumps(result))
