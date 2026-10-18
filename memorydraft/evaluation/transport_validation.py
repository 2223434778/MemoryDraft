"""Frozen independent controlled cases, strict audits and project-level comparisons."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

from memorydraft.evaluation.runtime_migration import check_result
from memorydraft.evaluation.transport_probe import METHODS, REPEATS
from memorydraft.memory.store import MemoryEvent, Snapshot
from memorydraft.runtime.benchmark import percentile
from memorydraft.runtime.contracts import GenerationRequest


GROUPS = ("local-edit", "ambiguity", "control")
COUNTERS = ("rounds", "draft_rounds", "reuse_rounds", "reuse_accepted_tokens", "alignment_calls",
            "alignment_candidates", "alignment_proposals", "alignment_accepted_tokens", "alignment_seconds")


@dataclass(frozen=True)
class ValidationTask:
    project_id: str
    group: str
    request: GenerationRequest
    snapshot_id: str
    expected_text: str
    reference: dict | None = None


def load_inputs(path: Path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    projects = value["projects"]
    if value["schema"] != 1 or len(projects) != 12 or len({p["id"] for p in projects}) != 12:
        raise ValueError("Expected twelve distinct frozen projects")
    if any(sum(p["group"] == group for p in projects) != 4 for group in GROUPS):
        raise ValueError("Expected four projects in each predefined group")
    tasks, events = [], []
    users = set()
    for project in projects:
        history = tuple(MemoryEvent(**e) for e in project["history"])
        if len(project["requests"]) != 2:
            raise ValueError("Expected two independent requests per project")
        project_user = None
        for item in project["requests"]:
            request = GenerationRequest.from_dict(item["request"])
            if request.max_new_tokens != 512 or any(e.user_id != request.user_id or e.order >= request.order for e in history):
                raise ValueError("Invalid output budget or history user/time")
            if project_user is not None and project_user != request.user_id:
                raise ValueError("Project requests must share their isolated user")
            project_user = request.user_id
            edit = item["edit"]
            if (edit["old"] not in project["source"]
                    or project["source"].replace(edit["old"], edit["new"], 1) != item["expected_text"]
                    or project["source"] not in request.messages[-1]["content"]):
                raise ValueError("Task contract must be derived from the supplied current input")
            snapshot = Snapshot(request.user_id, request.order, history)
            tasks.append(ValidationTask(project["id"], project["group"], request, snapshot.snapshot_id, item["expected_text"]))
        if project_user in users:
            raise ValueError("Projects must use separate users")
        users.add(project_user)
        events.extend(history)
    if len({t.request.id for t in tasks}) != 24:
        raise ValueError("Expected twenty-four distinct requests")
    return tasks, events


def contract_check(text, expected):
    # Trailing LF count is ignored; every other character remains significant.
    return {"artifact_correct": text.rstrip("\n") == expected.rstrip("\n"), "exact_text": text == expected}


def attach_references(tasks, references):
    keyed = {row["request_id"]: row for row in references}
    if len(keyed) != len(references) or keyed.keys() != {t.request.id for t in tasks}:
        raise ValueError("Missing, duplicate or unexpected fresh AR reference")
    attached = []
    for task in tasks:
        row = keyed[task.request.id]
        if (row["method"] != "AR" or row["snapshot_id"] != task.snapshot_id
                or not row["token_ids"] or any(type(t) is not int or t < 0 for t in row["token_ids"])
                or len(row["token_ids"]) > task.request.max_new_tokens
                or not isinstance(row["prompt_sha256"], str) or len(row["prompt_sha256"]) != 64
                or row["finish_reason"] not in {"eos", "length"}
                or not math.isfinite(row["latency_seconds"]) or row["latency_seconds"] <= 0
                or row["contract"] != contract_check(row["text"], task.expected_text)):
            raise ValueError("Invalid fresh reference, legal snapshot or artifact contract")
        attached.append(replace(task, reference=row))
    return attached


def audit_rows(rows, tasks):
    definitions = {t.request.id: t for t in tasks}
    expected = {(t.request.id, m, r) for t in tasks for m in METHODS for r in range(REPEATS)}
    seen = set()
    for row in rows:
        key = row["request_id"], row["method"], row["repeat"]
        if key not in expected or key in seen:
            raise ValueError("Missing, duplicate or unexpected validation pair")
        task = definitions[row["request_id"]]
        if (task.reference is None or not row["correct"] or row["token_ids"] != task.reference["token_ids"]
                or row["prompt_sha256"] != task.reference["prompt_sha256"] or row["snapshot_id"] != task.snapshot_id
                or row["project_id"] != task.project_id or row["group"] != task.group
                or row["budget"] != task.request.max_new_tokens or row["contract"] != task.reference["contract"]
                or row["finish_reason"] != task.reference["finish_reason"]
                or not math.isfinite(row["latency_seconds"]) or row["latency_seconds"] <= 0):
            raise ValueError("Changed validation output, input, snapshot or timing contract")
        counters = row["counters"]
        if (any(not isinstance(counters.get(k, 0), (int, float)) or not math.isfinite(counters.get(k, 0))
                or counters.get(k, 0) < 0 for k in COUNTERS)
                or counters["rounds"] != counters["draft_rounds"] + counters["reuse_rounds"]
                or counters.get("alignment_calls", 0) > 4
                or counters.get("alignment_candidates", 0) > 8 * counters.get("alignment_calls", 0)
                or counters.get("alignment_proposals", 0) > counters.get("alignment_calls", 0)
                or counters.get("alignment_accepted_tokens", 0) > counters["reuse_accepted_tokens"]
                or (row["method"] in {"DFlash", "Suffix-R"} and counters.get("alignment_calls", 0))
                or (row["method"] == "DFlash" and counters["reuse_rounds"])):
            raise ValueError("Invalid round, retrieval or alignment accounting")
        seen.add(key)
    if seen != expected:
        raise ValueError("Incomplete validation pairs")

    def summarize(selected):
        return {method: {"observations": sum(r["method"] == method for r in selected),
                         "seconds": sum(r["latency_seconds"] for r in selected if r["method"] == method),
                         "counters": {k: sum(r["counters"].get(k, 0) for r in selected if r["method"] == method)
                                      for k in COUNTERS}} for method in METHODS}

    projects = {p: summarize([r for r in rows if r["project_id"] == p]) for p in sorted({t.project_id for t in tasks})}
    groups = {g: summarize(rows if g == "all" else [r for r in rows if r["group"] == g]) for g in ("all", *GROUPS)}
    return {"rows": len(rows), "requests": len(tasks), "projects": projects, "groups": groups,
            "contracts": {g: {"correct": sum(t.reference["contract"]["artifact_correct"] for t in tasks if g == "all" or t.group == g),
                               "requests": sum(g == "all" or t.group == g for t in tasks),
                               "length_stops": sum(t.reference["finish_reason"] == "length" for t in tasks if g == "all" or t.group == g)}
                          for g in ("all", *GROUPS)}}


def comparisons(summary, tasks):
    result = {}
    for group in ("all", *GROUPS):
        projects = sorted({t.project_id for t in tasks if group == "all" or t.group == group})
        table = summary["projects"]
        comparisons_for_group = {}
        for baseline in ("DFlash", "Suffix-R", "Edit-R"):
            def ratio(selected):
                return sum(table[p]["OT-R"]["seconds"] for p in selected) / sum(table[p][baseline]["seconds"] for p in selected)
            # Enumerate the small four-project bootstrap exactly; sample the 12-project aggregate deterministically.
            if len(projects) <= 4:
                resampled = itertools.product(projects, repeat=len(projects))
            else:
                rng = random.Random(20261018)
                resampled = (rng.choices(projects, k=len(projects)) for _ in range(4000))
            distribution = [ratio(sample) for sample in resampled]
            comparisons_for_group[baseline] = {"ot_over_baseline": ratio(projects),
                                               "project_bootstrap_95": [percentile(distribution, .025), percentile(distribution, .975)],
                                               "projects_faster": sum(table[p]["OT-R"]["seconds"] < table[p][baseline]["seconds"] for p in projects),
                                               "projects": len(projects)}
        contract = summary["contracts"][group]
        suffix, edit = comparisons_for_group["Suffix-R"], comparisons_for_group["Edit-R"]
        comparisons_for_group["engineering_gate"] = bool(
            group in {"local-edit", "ambiguity"} and suffix["ot_over_baseline"] <= .95
            and edit["ot_over_baseline"] <= .98 and suffix["projects_faster"] >= 3
            and edit["projects_faster"] >= 3 and contract["correct"] == contract["requests"]
            and not contract["length_stops"])
        result[group] = comparisons_for_group
    return result


def prompt_identity(tokenizer, request, template_options):
    ids = tokenizer.apply_chat_template(list(request.messages), add_generation_prompt=True, **template_options)
    return {"prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), "prompt_tokens": len(ids)}


def check_generation(result, task):
    check_result(result, task)
    if result.text != task.reference["text"] or result.finish_reason != task.reference["finish_reason"]:
        raise ValueError("Text/stop reason differs despite strict token equality")
