"""Build a reviewable source-only deployment archive; never collect models/secrets."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


root = Path(__file__).resolve().parents[1]
paths = []
for directory in ("memorydraft", "scripts", "tests", "configs", "docs", "data/synthetic"):
    paths.extend(p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
paths.extend(root / name for name in ("pyproject.toml", "README.md", "AGENTS.md", ".gitignore"))
manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
payload = {"files": manifest, "source_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
           "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
(root / "SOURCE_MANIFEST.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
    for path in paths + [root / "SOURCE_MANIFEST.json"]:
        archive.add(path, arcname=path.relative_to(root).as_posix())
print(payload["source_sha256"])
