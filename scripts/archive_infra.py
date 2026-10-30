"""Export an immutable, byte-exact interview copy from a committed Git reference."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def git(*args, **kwargs):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE, **kwargs).stdout


def export(reference, destination, *, version="0.3"):
    if version not in {"0.3", "0.4", "0.5"}:
        raise ValueError("Unknown infra archive version")
    destination = Path(destination).resolve()
    if not destination.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Interview exports must stay under this workspace's artifacts directory")
    commit = git("rev-parse", reference + "^{commit}").decode().strip()
    destination.mkdir(parents=True, exist_ok=False)
    project = destination / "project"
    # A separate clone keeps provenance independent of subsequent research commits.
    git("-c", "core.autocrlf=false", "clone", "--no-hardlinks", "--single-branch",
        "--branch", reference, str(ROOT), str(project), stderr=subprocess.PIPE)
    entries = []
    for line in git("ls-tree", "-rz", commit).split(b"\0"):
        if line:
            metadata, name = line.split(b"\t", 1)
            mode, kind, oid = metadata.decode().split()
            if kind != "blob":
                raise ValueError("Submodules need a separate provenance audit")
            entries.append((name.decode("utf-8"), oid, mode))
    batch = git("cat-file", "--batch", input=("\n".join(e[1] for e in entries) + "\n").encode())
    stream, hashes = io.BytesIO(batch), {}
    archive = destination / f"MemoryDraft-infra-v{version}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, oid, mode in entries:
            header = stream.readline().decode().split()
            if header[:2] != [oid, "blob"]:
                raise ValueError("Unexpected Git object")
            data = stream.read(int(header[2]))
            if stream.read(1) != b"\n":
                raise ValueError("Truncated Git object")
            path = project / name
            if not path.resolve().is_relative_to(project):
                raise ValueError("Archive path escapes project")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            item = zipfile.ZipInfo("MemoryDraft/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = int(mode, 8) << 16
            output.writestr(item, data)
            hashes[name] = hashlib.sha256(data).hexdigest()
    bundle = destination / f"MemoryDraft-infra-v{version}.bundle"
    git("bundle", "create", str(bundle), reference)
    subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=project, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if any(hashlib.sha256((project / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
        raise ValueError("Export differs from committed source")
    manifest = {"commit": commit, "reference": reference, "version": version, "files": hashes,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "scope": "Tracked source, synthetic/public evidence and independent Git history; excludes weights, environments and private data."}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (destination / "START_HERE.md").write_text(
        "# MemoryDraft infra interview copy\n\n"
        f"Frozen commit: `{commit}`. Independent repository: `project/`.\n\n" +
        ("Read project/docs/INFRA_FIDELITY_GUIDE.md first for the high-fidelity/optimized tree release.\n\n" if version == "0.5" else
         "Read project/docs/INFRA_TREE_GUIDE.md first for the cost-tree/ASD release.\n\n" if version == "0.4" else "") +
        "Read project/README.md, project/docs/INFRA_GUIDE.md, project/docs/INFRA_VALIDATION.md "
        "and project/docs/RUNTIME_MIGRATION_RESULTS.md.\n\n"
        "Install into a new environment: `python -m pip install -e \".[serve,client,test]\"`. "
        "CPU transport demo needs no weights. GPU instructions and validated dependency profiles are in INFRA_GUIDE.md. "
        "The virtual environment and model weights are intentionally excluded. "
        "Use the ZIP for source delivery or the bundle for offline Git restoration. "
        "manifest.json hashes the exact committed bytes. OT research is separate.\n", encoding="utf-8")
    return {key: value for key, value in manifest.items() if key != "files"} | {"files": len(hashes), "directory": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="codex/infra-interview-v0.3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", choices=("0.3", "0.4", "0.5"), default="0.3")
    args = parser.parse_args()
    print(json.dumps(export(args.ref, args.output, version=args.version), indent=2))


if __name__ == "__main__":
    main()
