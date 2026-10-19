"""Frozen draft-guided experiment inputs and paired accounting."""
from __future__ import annotations

import json
import math
from pathlib import Path

from memorydraft.evaluation.runtime_migration import check_result
from memorydraft.evaluation.transport_validation import ValidationTask
from memorydraft.memory.store import MemoryEvent, Snapshot
from memorydraft.runtime.contracts import GenerationRequest


METHODS = ("DFlash", "Suffix-R", "OT-R", "Guided-Edit", "Guided-OT")
RETRIEVAL = {"DFlash": "suffix", "Suffix-R": "suffix", "OT-R": "ot",
             "Guided-Edit": "edit-guided", "Guided-OT": "ot-guided"}
COUNTERS = ("rounds", "draft_rounds", "reuse_rounds", "reuse_accepted_tokens", "alignment_calls",
            "alignment_candidates", "alignment_proposals", "alignment_accepted_tokens", "alignment_seconds",
            "guided_lookups", "guided_no_match", "draft_forward_rounds")


def load_inputs(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks, events, users = [], [], set()
    if data["schema"] != 1 or len(data["projects"]) != 8:
        raise ValueError("Expected eight frozen projects")
    for project in data["projects"]:
        if project["group"] not in {"edited-continuation", "control"} or len(project["requests"]) != 2:
            raise ValueError("Unexpected frozen group or request count")
        history = tuple(MemoryEvent(**e) for e in project["history"])
        project_user = project["requests"][0]["request"]["user_id"]
        if project_user in users:
            raise ValueError("Project users must be isolated")
        users.add(project_user)
        for item in project["requests"]:
            request = GenerationRequest.from_dict(item["request"])
            old, new = item["edit"]["old"], item["edit"]["new"]
            if (request.user_id != project_user or request.max_new_tokens != 512
                    or any(e.user_id != project_user or e.order >= request.order for e in history)
                    or project["source"].count(old) != 1
                    or project["source"].replace(old, new, 1) != item["expected_text"]
                    or project["source"] not in request.messages[-1]["content"]):
                raise ValueError("Changed frozen input, history or edit contract")
            snapshot = Snapshot(request.user_id, request.order, history)
            tasks.append(ValidationTask(project["id"], project["group"], request,
                                        snapshot.snapshot_id, item["expected_text"]))
        events.extend(history)
    if (len({t.request.id for t in tasks}) != 16 or len({t.project_id for t in tasks}) != 8
            or sum(t.group == "edited-continuation" for t in tasks) != 8):
        raise ValueError("Expected sixteen distinct requests and four projects per group")
    return tasks, events


def audit(rows, tasks, methods, repeats):
    definitions = {t.request.id: t for t in tasks}
    expected = {(t.request.id, m, r) for t in tasks for m in methods for r in range(repeats)}
    if len(rows) != len(expected) or {(r["request_id"], r["method"], r["repeat"]) for r in rows} != expected:
        raise ValueError("Missing or duplicate guided experiment observations")
    for row in rows:
        task = definitions[row["request_id"]]
        if (not row["correct"] or row["token_ids"] != task.reference["token_ids"]
                or row["prompt_sha256"] != task.reference["prompt_sha256"] or row["snapshot_id"] != task.snapshot_id
                or row["project_id"] != task.project_id
                or row["group"] != getattr(task, "group", "development")
                or not math.isfinite(row["latency_seconds"]) or row["latency_seconds"] <= 0):
            raise ValueError("Changed reference, prompt, snapshot or timing")
        c = row["counters"]
        if (c["rounds"] != c["draft_rounds"] + c["reuse_rounds"]
                or c.get("guided_lookups", 0) > 4 or c.get("alignment_calls", 0) > 4
                or c.get("alignment_candidates", 0) > 8 * c.get("alignment_calls", 0)
                or c.get("alignment_accepted_tokens", 0) > c["reuse_accepted_tokens"]
                or row["method"].startswith("Guided-")
                and not c["draft_rounds"] <= c["draft_forward_rounds"] <= c["rounds"]):
            raise ValueError("Invalid guided accounting")
    groups = {"all": rows}
    for group in {getattr(t, "group", "development") for t in tasks}:
        groups[group] = [r for r in rows if getattr(definitions[r["request_id"]], "group", "development") == group]
    return {group: {method: dict(seconds=sum(r["latency_seconds"] for r in selected if r["method"] == method),
            counters={k: sum(r["counters"].get(k, 0) for r in selected if r["method"] == method) for k in COUNTERS})
            for method in methods} for group, selected in groups.items()}
