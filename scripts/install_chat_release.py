"""Extract and verify a source-only chat release; switching services is explicit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def install(archive, root, expected):
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("SOURCE.json"))
        identity = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
        if identity != expected or manifest["sha256"] != identity:
            raise ValueError("Source identity mismatch")
        if set(bundle.namelist()) != {*manifest["files"], "SOURCE.json"}:
            raise ValueError("Unexpected release files")
        destination = root.resolve() / "releases" / identity[:12]
        if destination.exists():
            raise FileExistsError(destination)
        for name, digest in manifest["files"].items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Invalid release path")
            content = bundle.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("Release file mismatch: " + name)
        destination.mkdir(parents=True)
        bundle.extractall(destination)
        for name, digest in manifest["files"].items():
            if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Extracted file mismatch: " + name)
    print(json.dumps({"verified": True, "source_sha256": identity, "files": len(manifest["files"]),
                      "release": str(destination)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    install(args.archive, args.root, args.sha256)
