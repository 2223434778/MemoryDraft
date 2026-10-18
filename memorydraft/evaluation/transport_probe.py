"""Frozen old workloads and audits for the exploratory OT-R pilot."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

from memorydraft.evaluation.runtime_migration import ReplayTask, frozen_inputs, read_rows
from memorydraft.memory.store import MemoryEvent, Snapshot
from memorydraft.runtime.contracts import GenerationRequest


METHODS = ("DFlash", "Suffix-R", "Edit-R", "OT-R")
REPEATS = 2


def probe_inputs(root: Path):
    tasks, events = frozen_inputs(root / "results/reuse-confirmation")
    # Exactly the historical infra acceptance event, not any current output.
    event = MemoryEvent("example", "past-quotation", 1, "alpha bravo charlie delta echo foxtrot.\n" * 4)
    events.append(event)
    old = read_rows(root / "results/infra-v0.3/benchmark-rows.jsonl")
    for line in (root / "examples/requests.jsonl").read_text(encoding="utf-8").splitlines():
        request = GenerationRequest.from_dict(json.loads(line))
        matches = [row for row in old if row["request_id"] == request.id]
        if not matches or any((row["token_ids"], row["prompt_sha256"], row["snapshot_id"])
                              != (matches[0]["token_ids"], matches[0]["prompt_sha256"], matches[0]["snapshot_id"])
                              for row in matches):
            raise ValueError("Infra reference is missing or inconsistent")
        snapshot = Snapshot(request.user_id, request.order, (event,))
        if snapshot.snapshot_id != matches[0]["snapshot_id"]:
            raise ValueError("Infra history differs from original acceptance")
        tasks.append(ReplayTask("infra-" + request.id, 0, request, matches[0], snapshot.snapshot_id))
    return tasks, events


def prefix_length(proposal, reference):
    if proposal is None:
        return 0
    count = 0
    for actual, expected in zip(proposal.tokens, reference):
        if actual != expected:
            break
        count += 1
    return count


def raw_span_oracle(index, reference, positions=None, limit=32):
    """Gold-only diagnostic; not called by retrieval or the online generation loop."""
    if not reference:
        return 0
    if positions is None:
        positions = ((position, end) for start, end in zip(index.starts, index.ends)
                     for position in range(start, end) if index.tokens[position] == reference[0])
    best = 0
    for position, end in positions:
        count = 0
        for actual, expected in zip(index.tokens[position:min(position + limit, end)], reference):
            if actual != expected:
                break
            count += 1
        best = max(best, count)
    return best


def audit(rows, tasks):
    definitions = {task.request.id: task for task in tasks}
    expected = {(task.request.id, method, repeat) for task in tasks for method in METHODS for repeat in range(REPEATS)}
    seen = set()
    for row in rows:
        key = row["request_id"], row["method"], row["repeat"]
        if key not in expected or key in seen:
            raise ValueError("Unexpected, duplicate or unpaired OT observation")
        task = definitions[row["request_id"]]
        if (not row["correct"] or row["token_ids"] != task.reference["token_ids"]
                or row["prompt_sha256"] != task.reference["prompt_sha256"] or row["snapshot_id"] != task.snapshot_id
                or row["project_id"] != task.project_id or row["budget"] != task.request.max_new_tokens
                or not math.isfinite(row["latency_seconds"]) or row["latency_seconds"] <= 0):
            raise ValueError("OT result differs from frozen output, input, snapshot or timing contract")
        counters = row["counters"]
        if (counters["rounds"] != counters["draft_rounds"] + counters["reuse_rounds"]
                or counters.get("alignment_calls", 0) > 4
                or counters.get("alignment_accepted_tokens", 0) > counters["reuse_accepted_tokens"]):
            raise ValueError("Invalid OT round or budget accounting")
        seen.add(key)
    if seen != expected:
        raise ValueError("Missing OT observations")
    groups = {"all": rows, "old-json": [r for r in rows if r["project_id"].startswith("confirm-")],
              "infra-explain": [r for r in rows if r["project_id"] == "infra-explain"],
              "infra-copy": [r for r in rows if r["project_id"] == "infra-copy"]}
    return {group: {method: {
        "observations": sum(r["method"] == method for r in selected),
        "seconds": sum(r["latency_seconds"] for r in selected if r["method"] == method),
        "counters": {key: sum(r["counters"].get(key, 0) for r in selected if r["method"] == method)
                     for key in ("rounds", "draft_rounds", "reuse_rounds", "reuse_accepted_tokens",
                                 "alignment_calls", "alignment_candidates", "alignment_proposals",
                                 "alignment_accepted_tokens", "alignment_seconds")}}
        for method in METHODS} for group, selected in groups.items()}


def task_definitions(tasks):
    return [{"request": asdict(t.request), "project_id": t.project_id, "snapshot_id": t.snapshot_id}
            for t in tasks]
