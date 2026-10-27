"""Export only application source, never deployment credentials or chat databases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package(root: Path, destination: Path):
    paths = sorted(p for p in (root / "memorydraft").rglob("*") if p.is_file()
                   and "__pycache__" not in p.parts and p.suffix in {".py", ".json", ".txt", ".html", ".css", ".js"})
    paths += [root / "pyproject.toml", root / "README.md"]
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.writestr(p.relative_to(root).as_posix(), p.read_bytes())
        archive.writestr("SOURCE.json", json.dumps({"sha256": identity, "files": manifest}, indent=2))
    return {"source_sha256": identity, "files": len(paths), "archive": str(destination.resolve()),
            "archive_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(package(Path(__file__).resolve().parents[1], args.output), indent=2))


if __name__ == "__main__":
    main()
