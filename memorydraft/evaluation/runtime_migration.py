"""Frozen workload replay and strict migration audit; no decoding policy changes."""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

from memorydraft.evaluation.reuse_attribution import legal_snapshots
from memorydraft.evaluation.reuse_cases import confirmation_projects, messages
from memorydraft.memory.store import MemoryEvent
from memorydraft.runtime.contracts import GenerationRequest


METHODS = ("Legacy-D", "Legacy-R", "Runtime-D", "Runtime-R", "Runtime-R-aligned")
LEGACY_COMMITS = {"legacy_decode.py": "dd0077fb6ed278837f9e3de48f946008519816ff",
                  "legacy_suffix.py": "f43239af59c49c17ee4b9d7749473259e501251d"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_hashes(source):
    return {path.name: file_hash(path) for path in sorted(Path(source).iterdir()) if path.is_file()}


def read_rows(path):
    path = Path(path)
    data = path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class ReplayTask:
    project_id: str
    turn: int
    request: GenerationRequest
    reference: dict
    snapshot_id: str


def frozen_inputs(source):
    """Reconstruct legal completed history, without generating or committing answers."""
    source = Path(source)
    projects = json.loads((source / "projects.json").read_text(encoding="utf-8"))
    initial = json.loads((source / "initial-history.json").read_text(encoding="utf-8"))
    references = read_rows(source / "references.jsonl")
    if projects != confirmation_projects():
        raise ValueError("Frozen requests differ from the original confirmation")
    snapshots = legal_snapshots(projects, initial, references)
    labels = {(r["project_id"], r["turn"]): r for r in references}
    events = [MemoryEvent(**event) for event in initial]
    tasks = []
    for project in projects:
        for turn, text in enumerate(project["currents"]):
            label = labels[project["id"], turn]
            request = GenerationRequest(f"{project['id']}:{turn}", project["id"], 10 + 10 * turn,
                                        tuple(messages(text)), project["budget"])
            tasks.append(ReplayTask(project["id"], turn, request, label, snapshots[project["id"], turn].snapshot_id))
            if turn + 1 < len(project["currents"]):
                # This answer was already finalized before the subsequent original request.
                events.append(MemoryEvent(project["id"], "artifact", request.order + 1,
                                          label["text"], project_id=project["id"]))
    return tasks, events


def check_result(result, task):
    if (result.token_ids != task.reference["token_ids"] or result.prompt_sha256 != task.reference["prompt_sha256"]
            or result.snapshot_id != task.snapshot_id or result.request_id != task.request.id):
        raise ValueError("Generation differs from frozen tokens, Target prompt or legal snapshot")


def audit_rows(rows, tasks, *, repeats=3):
    definitions = {task.request.id: task for task in tasks}
    expected = {(task.request.id, method, repeat) for task in tasks for method in METHODS for repeat in range(repeats)}
    seen = set()
    for row in rows:
        key = row["request_id"], row["method"], row["repeat"]
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate migration pair")
        task = definitions[row["request_id"]]
        if (not row["correct"] or row["token_ids"] != task.reference["token_ids"]
                or row["prompt_sha256"] != task.reference["prompt_sha256"] or row["snapshot_id"] != task.snapshot_id
                or row["project_id"] != task.project_id or row["turn"] != task.turn
                or row["budget"] != task.request.max_new_tokens or row["latency_seconds"] <= 0):
            raise ValueError("Migration row differs from frozen input, output or snapshot")
        counters = row["counters"]
        if counters["rounds"] != counters["draft_rounds"] + counters["reuse_rounds"]:
            raise ValueError("Invalid round counters")
        seen.add(key)
    if seen != expected:
        raise ValueError("Incomplete migration pairs")
    return {"rows": len(rows), "requests": len(tasks), "projects": len({t.project_id for t in tasks}),
            "times": {method: sum(row["latency_seconds"] for row in rows if row["method"] == method) for method in METHODS},
            "counters": {method: {key: sum(row["counters"][key] for row in rows if row["method"] == method)
                                   for key in ("rounds", "draft_rounds", "reuse_rounds", "reuse_accepted_tokens")}
                         for method in METHODS}}
