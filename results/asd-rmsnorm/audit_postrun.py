"""Task-local audit of the frozen compilation experiment and prior evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from scripts.diagnose_transport_online import write

root = Path.cwd()
base = root/"results/asd-rmsnorm"
freeze = "39f99df5352796f7c2d02bc2b1d030cc19bd5657"
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=root)


manifest = json.loads((base/"ASD_RMSNORM_SOURCE_MANIFEST.json").read_text())
execution = json.loads((base/"execution-verification.json").read_text())
assert manifest["commit"] == execution["commit"] == freeze
assert execution["status"] == "complete"
assert execution["files_verified_before"] == execution["files_verified_after"] == len(manifest["files"])
assert sha(json.dumps(manifest["files"], sort_keys=True).encode()) == manifest["source_sha256"]
frozen = []
for name in ("memorydraft/backends/regional_compile.py", "scripts/run_asd_compile_probe.py", "scripts/run_asd_compile_pipeline.py", "scripts/run_asd_rmsnorm_probe.py", "scripts/run_asd_rmsnorm_pipeline.py", "scripts/write_asd_rmsnorm_report.py", "tests/test_asd_rmsnorm.py",
             "scripts/write_asd_compile_report.py", "docs/ASD_RMSNORM_PROTOCOL.md", "memorydraft/backends/asd_cpu.py",
             "memorydraft/backends/asd.py", "memorydraft/backends/dflash_reuse.py", "memorydraft/runtime/engine.py",
             "memorydraft/runtime/config.py", "memorydraft/evaluation/asd_suite.py", "scripts/run_asd_suite.py",
             "scripts/write_asd_cpu_report.py"):
    original = git("show", freeze+":"+name).decode()
    assert original == (root/name).read_text(encoding="utf-8"), name
    frozen.append(dict(path=name, sha256=sha(original.encode())))
runtime_unchanged = []
for name in ("memorydraft/backends/asd_cpu.py", "memorydraft/backends/asd.py", "memorydraft/backends/dflash_reuse.py",
             "memorydraft/runtime/engine.py", "memorydraft/runtime/config.py", "memorydraft/evaluation/asd_suite.py",
             "scripts/run_asd_suite.py"):
    assert git("show", "38ce3b3:"+name).decode() == (root/name).read_text(encoding="utf-8"), name
    runtime_unchanged.append(name)
protected = {}
counts = []
for i in (0, 1):
    info = json.loads((base/f"part-{i}/provenance.json").read_text())
    assert info["status"] == "complete" and info["original_unchanged"]
    assert info["counts"] == dict(warmup=10, preflight=60, timed=144, diagnostics=6, rmsnorm_states=96)
    assert info["compiler"]["float32_matmul_precision"] == "highest" and not info["compiler"]["allow_tf32"]
    assert info["compiler"]["graph_count"] <= 8 and info["compiler"]["backend_seconds"] <= 300
    counts.append(info["counts"])
    for name, digest in info["original_sha256"].items():
        assert name not in protected or protected[name] == digest
        protected[name] = digest
legacy = []
for name, digest in protected.items():
    data = (root/name).read_bytes()
    if sha(data) != digest:
        assert name in {"results/research-audit/model_preflight.json", "results/research-audit/pilot_summary.json"}
        assert sha(data.replace(b"\r\n", b"\n")) == digest, name
        legacy.append(name)
publications = []
for name, commit in (("results/asd-suite-stage1/publication.json", "dcfc01de31b4e619d89872e8cb7487648aeb6ce5"),
                     ("results/asd-block-preflight/publication.json", "2a4ab50f72098c9cba74450d951f276fbd8c25aa"),
                     ("results/asd-block-suite/publication.json", "d72f6b6e9a306f86016426d84ebf8c9b040eb852"),
                     ("results/asd-execution/publication.json", "1e56ecbe071ec305fb2982c467099d98a725c079"),
                     ("results/asd-cpu/publication.json", "38ce3b393dcf724d5e51a8a33a46765b25f8c6e9"),
                     ("results/asd-compile/publication.json", "4f3aa2ab92391f0ee5211f4563fd5d64010a92e3")):
    info = json.loads((root/name).read_text())
    for path, digest in info["files"].items():
        assert sha(git("show", commit+":"+path)) == digest, path
    publications.append(dict(path=name, commit=commit, files_verified=len(info["files"])))
refs = {ref: git("rev-parse", ref).decode().strip() for ref in ("main", "codex/infra-interview-v0.3", "infra-v0.3-interview^{}")}
assert set(refs.values()) == {"27e36199f8a65acdb9a9c42ccb744ddfa9a9402b"}
archive = root/"artifacts/asd-rmsnorm-results.zip"
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        assert (base/name).read_bytes() == z.read(name), name
write(base/"postrun-audit.json", dict(status="complete", source_commit=freeze, source_sha256=manifest["source_sha256"],
      execution_files=len(manifest["files"]), frozen_files=frozen, prior_runtime_unchanged=runtime_unchanged,
      protected_results_verified=len(protected), legacy_crlf_normalized=legacy, prior_publications=publications,
      infra_refs=refs, counts=counts, result_zip=dict(bytes=archive.stat().st_size, sha256=sha(archive.read_bytes()))))
print(json.dumps(dict(protected_results=len(protected), frozen_files=len(frozen), prior_publications=publications, counts=counts)))
