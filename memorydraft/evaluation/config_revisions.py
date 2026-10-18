"""Frozen public-project replay tasks and source/quality audits (no network)."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


SYSTEM = "Apply the requested configuration edit exactly. Follow the output format. Do not explain."
METHODS_BY_MODE = {"full": ("AR", "DFlash", "Adapted", "Global", "Context", "Arctic"),
                   "diff": ("DFlash", "Global", "Context")}


def load_projects(directory: Path, split: str):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen public source hash mismatch: " + relative)
    payload = json.loads((directory / "projects.json").read_text(encoding="utf-8"))
    seen = set()
    for project in payload["projects"]:
        if project["repository"] in seen or project["repository"] not in payload["repositories"][project["split"]]:
            raise ValueError("Duplicate project or incorrect split")
        seen.add(project["repository"])
        license_data = (directory / project["license"]["path"]).read_bytes()
        if hashlib.sha256(license_data).hexdigest() != project["license"]["sha256"]:
            raise ValueError("License hash mismatch")
        revisions = project["revisions"]
        if len(revisions) != 3 or len(project["turns"]) != 2:
            raise ValueError("Expected two consecutive file revisions")
        for revision in revisions:
            if (hashlib.sha256(revision["text"].encode()).hexdigest() != revision["sha256"]
                    or revision["date"] > payload["cutoff"]):
                raise ValueError("File hash or historical cutoff mismatch")
        for turn, before, after in zip(project["turns"], revisions, revisions[1:]):
            patch = "".join(difflib.unified_diff(before["text"].splitlines(keepends=True),
                                               after["text"].splitlines(keepends=True),
                                               fromfile="a/" + project["path"], tofile="b/" + project["path"]))
            if (turn["before"] != before["text"] or turn["expected"] != after["text"]
                    or turn["before_commit"] != before["commit"] or turn["after_commit"] != after["commit"]
                    or turn["patch"] != patch or turn["before"] == turn["expected"]):
                raise ValueError("Revision chain or derived patch mismatch")
    if seen != {name for group in payload["repositories"].values() for name in group}:
        raise ValueError("Missing predeclared repository; do not silently filter projects")
    return [p for p in payload["projects"] if p["split"] == split]


def request_text(project, turn, mode="full"):
    item = project["turns"][turn]
    if mode not in {"full", "diff"}:
        raise ValueError("Unknown editor output contract")
    contract = ("Return the entire updated YAML file, without markdown fences. Preserve all unaffected text."
                if mode == "full" else "Return only the unified diff, without markdown fences or explanations.")
    return (f"Repository: {project['repository']}\nFile: {project['path']}\n"
            f"Apply exactly the supplied patch to the original file. {contract}\n"
            f"Original file:\n<original>\n{item['before']}</original>\n"
            f"Requested change (unified diff):\n<patch>\n{item['patch']}</patch>\n")


def messages(text):
    return [dict(role="system", content=SYSTEM), dict(role="user", content=text)]


def working_event(project, mode, turn):
    from memorydraft.memory.store import MemoryEvent

    return MemoryEvent(project["id"] + ":" + mode, "working-file", 1 if turn == 0 else 9 + 10 * turn,
                       project["turns"][turn]["before"], source_role="user", project_id=project["id"])


def answer_event(project, mode, turn, text):
    from memorydraft.memory.store import MemoryEvent

    return MemoryEvent(project["id"] + ":" + mode, "artifact", 11 + 10 * turn, text, project_id=project["id"])


def quality(text, project, turn, mode):
    expected = project["turns"][turn]["expected" if mode == "full" else "patch"]
    exact = text.rstrip("\n") == expected.rstrip("\n")
    parsed, equivalent, error, applies = None, None, None, None
    if mode == "full":
        import yaml

        try:
            actual = yaml.load(text, Loader=yaml.BaseLoader)
            parsed = isinstance(actual, dict)
            equivalent = parsed and actual == yaml.load(expected, Loader=yaml.BaseLoader)
        except yaml.YAMLError as failure:
            parsed, equivalent, error = False, False, type(failure).__name__
    else:
        # Apply only a patch whose sole destination is the declared public file.
        # No shell or workflow commands are executed, and git rejects path traversal.
        scratch = Path(__file__).resolve().parents[2] / ".test-tmp-config-patches"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="check-", dir=scratch) as directory:
            root = Path(directory)
            path = root / project["path"]
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Unsafe frozen project path")
            path.parent.mkdir(parents=True)
            path.write_bytes(project["turns"][turn]["before"].encode())
            # Isolate this text check from the enclosing repo and Windows checkout settings.
            environment = dict(os.environ, GIT_CEILING_DIRECTORIES=str(scratch.resolve()))
            command = ["git", "-c", "core.autocrlf=false", "apply"]
            stat = subprocess.run([*command, "--numstat", "-"], cwd=root, input=text.encode(),
                                  capture_output=True, env=environment)
            paths = [line.split(b"\t")[-1].decode() for line in stat.stdout.splitlines()]
            applies = False
            if stat.returncode == 0 and paths == [project["path"]]:
                result = subprocess.run([*command, "--whitespace=nowarn", "-"], cwd=root,
                                        input=text.encode(), capture_output=True, env=environment)
                applies = result.returncode == 0 and path.read_bytes().rstrip(b"\n") == project["turns"][turn]["expected"].encode().rstrip(b"\n")
    return dict(exact_file_or_patch=exact, yaml_mapping=parsed, yaml_equivalent=equivalent,
                patch_applies_to_expected=applies, parse_error=error)
