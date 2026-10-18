"""Freeze explicitly acquired, commit-pinned release notes; no network access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SELECTION = (
    ("psf/requests", ("v2.32.3", "v2.32.4", "v2.32.5"), "Apache-2.0"),
    ("encode/httpx", ("0.26.0", "0.27.0", "0.28.0"), "BSD-3-Clause"),
    ("pallets/click", ("8.1.8", "8.2.0", "8.2.1"), "BSD-3-Clause"),
    ("pallets/flask", ("3.0.3", "3.1.0", "3.1.1"), "BSD-3-Clause"),
    ("Kludex/starlette", ("0.45.0", "0.46.0", "0.47.0"), "BSD-3-Clause"),
    ("Textualize/rich", ("v13.9.0", "v14.0.0", "v14.1.0"), "MIT"),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def section(text, tag):
    version = re.escape(tag.removeprefix("v"))
    patterns = (rf"(?m)^## \[?{version}(?:\]|\s|$).*$",
                rf"(?m)^Version {version}$", rf"(?m)^{version} \([^\n]+\)$")
    found = next((match for pattern in patterns if (match := re.search(pattern, text))), None)
    if found is None:
        raise ValueError("Missing exact release heading: " + tag)
    tail = text[found.end():]
    boundary = re.search(r"(?m)^(?:## (?:\[)?\d+\.\d+|Version \d+\.\d+|\d+\.\d+\.\d+ \()", tail)
    return text[found.start():found.end() + (boundary.start() if boundary else len(tail))].rstrip() + "\n"


def prepare(cache, output):
    if (output / "projects.json").exists():
        raise ValueError("Preserve the frozen dataset; choose a new output directory")
    projects = []
    for repository, tags, spdx in SELECTION:
        identity = repository.replace("/", "--")
        release_notes = []
        for tag in tags:
            raw = json.loads((cache / f"{identity}--{tag}.json").read_text(encoding="utf-8"))
            if (raw["repo"] != repository or raw["tag"] != tag or not re.fullmatch(r"[0-9a-f]{40}", raw["commit"])
                    or raw["url"] != f"https://raw.githubusercontent.com/{repository}/{raw['commit']}/{raw['path']}"):
                raise ValueError("Unexpected source identity")
            notes = section(raw["text"], tag)
            release_notes.append(dict(tag=tag, version=tag.removeprefix("v"), commit=raw["commit"],
                url=raw["url"], path=raw["path"], source_sha256=sha(raw["text"].encode()),
                notes_sha256=sha(notes.encode()), notes=notes))
        license_data = json.loads((cache / f"{identity}--license.json").read_text(encoding="utf-8"))
        if license_data["commit"] != release_notes[-1]["commit"]:
            raise ValueError("License must be pinned to the final selected release")
        license_path = "licenses/" + identity + ".txt"
        (output / "licenses").mkdir(parents=True, exist_ok=True)
        (output / license_path).write_bytes(license_data["text"].encode())
        projects.append(dict(id=identity, repository=repository, releases=release_notes,
            license=dict(spdx=spdx, path=license_path, commit=license_data["commit"],
                         url=license_data["url"], sha256=sha(license_data["text"].encode()))))
    payload = dict(version=1, selection_rule="Named versions fixed before all model output; no outcome-based replacement",
                   acquired_on="2026-10-18", projects=projects)
    (output / "projects.json").write_bytes((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode())
    return payload


def freeze(output):
    destination = output / "manifest.json"
    if destination.exists():
        raise ValueError("Preserve the pre-run manifest")
    paths = [output / "projects.json", output / "labels.json", *sorted((output / "licenses").glob("*.txt"))]
    manifest = dict(version=1, files={p.relative_to(output).as_posix(): sha(p.read_bytes()) for p in paths})
    destination.write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    return sha(destination.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/raw/release-triage")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/release-triage")
    parser.add_argument("--freeze-labels", action="store_true")
    args = parser.parse_args()
    if args.freeze_labels:
        print("Pre-run manifest SHA-256:", freeze(args.output))
    else:
        payload = prepare(args.cache, args.output)
        print("Frozen", len(payload["projects"]), "projects / 18 release notes; labels still require pre-run review")
