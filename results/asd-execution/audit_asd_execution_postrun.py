"""Task-local provenance audit. Does not modify any previous evidence."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from scripts.diagnose_transport_online import write

root = Path.cwd()
base = root / "results/asd-execution"
freeze = "4290e2b7a6277709f8c8b90517ea2bdbe93cbabe"
sha = lambda value: hashlib.sha256(value).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=root)


manifest = json.loads((base / "ASD_EXECUTION_SOURCE_MANIFEST.json").read_text())
execution = json.loads((base / "execution-verification.json").read_text())
assert manifest["commit"] == execution["commit"] == freeze
assert execution["status"] == "complete"
assert execution["files_verified_before"] == execution["files_verified_after"] == len(manifest["files"])
assert sha(json.dumps(manifest["files"], sort_keys=True).encode()) == manifest["source_sha256"]
frozen = []
for name in ("scripts/run_asd_execution_probe.py", "scripts/run_asd_execution_pipeline.py",
             "scripts/run_asd_suite.py", "scripts/run_asd_budget_probe.py",
             "memorydraft/backends/asd.py", "memorydraft/backends/dflash_reuse.py",
             "memorydraft/runtime/engine.py", "memorydraft/runtime/config.py",
             "memorydraft/evaluation/asd_suite.py", "docs/ASD_EXECUTION_PROTOCOL.md"):
    before = git("show", freeze + ":" + name).decode()
    assert before == (root / name).read_text(encoding="utf-8"), name
    frozen.append(dict(path=name, sha256=sha(before.encode())))


class UnwrapMarks(ast.NodeTransformer):
    def visit_With(self, node):
        node = self.generic_visit(node)
        call = node.items[0].context_expr
        if len(node.items) == 1 and isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "mark":
            return node.body
        return node


path = "memorydraft/backends/dflash_reuse.py"
old = UnwrapMarks().visit(ast.parse(git("show", "d72f6b6:" + path).decode()))
new = UnwrapMarks().visit(ast.parse((root / path).read_text(encoding="utf-8")))
assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False), "Change beyond profiler scopes"
protected = {}
for shard in (0, 1):
    data = json.loads((base / f"part-{shard}/provenance.json").read_text())
    assert data["status"] == "complete" and data["original_unchanged"]
    assert data["counts"]["timed"] == 72 and data["counts"]["profile"] == 24 and data["counts"]["capture"] == 18
    for name, digest in data["original_sha256"].items():
        assert name not in protected or protected[name] == digest
        protected[name] = digest
legacy = []
for name, digest in protected.items():
    raw = (root / name).read_bytes()
    if sha(raw) != digest:
        assert name in {"results/research-audit/model_preflight.json", "results/research-audit/pilot_summary.json"}
        assert sha(raw.replace(b"\r\n", b"\n")) == digest, name
        legacy.append(name)
publications = []
for name, commit in (("results/asd-suite-stage1/publication.json", "dcfc01de31b4e619d89872e8cb7487648aeb6ce5"),
                     ("results/asd-block-preflight/publication.json", "2a4ab50f72098c9cba74450d951f276fbd8c25aa"),
                     ("results/asd-block-suite/publication.json", "d72f6b6e9a306f86016426d84ebf8c9b040eb852")):
    data = json.loads((root / name).read_text())
    for path, digest in data["files"].items():
        assert sha(git("show", commit + ":" + path)) == digest, path
    publications.append(dict(path=name, commit=commit, files_verified=len(data["files"])))
refs = {ref: git("rev-parse", ref).decode().strip() for ref in ("main", "codex/infra-interview-v0.3", "infra-v0.3-interview^{}")}
assert set(refs.values()) == {"27e36199f8a65acdb9a9c42ccb744ddfa9a9402b"}
archive = root / "artifacts/asd-execution-results.zip"
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        assert (base / name).read_bytes() == z.read(name), name
write(base / "postrun-audit.json", dict(status="complete", source_commit=freeze, source_sha256=manifest["source_sha256"],
      execution_files=len(manifest["files"]), frozen_files=frozen, backend_only_added_profiler_scopes=True,
      postrun_report_correction="Explicitly retain CUPTI device events without CPU links; do not force equal event counts. Timing, selection, scopes and resident replay execution unchanged.",
      protected_results_verified=len(protected), legacy_crlf_normalized=legacy, prior_publications=publications,
      infra_refs=refs, result_zip=dict(bytes=archive.stat().st_size, sha256=sha(archive.read_bytes()))))
print(json.dumps(dict(protected_results=len(protected), frozen_files=len(frozen), prior_publications=publications)))
