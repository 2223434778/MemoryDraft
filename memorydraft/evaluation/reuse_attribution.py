"""Post-hoc candidate attribution for the frozen synthetic confirmation only."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from memorydraft.evaluation.release_triage import continuation_map, oracle_coverage
from memorydraft.memory.store import MemoryEvent, Snapshot


EXTRA_TOKEN_BUDGET = 1536
ANCHOR, LIMIT, MINIMUM = 6, 32, 8


@dataclass(frozen=True)
class ConfigFacts:
    project: str
    collection: str
    fields: tuple[str, ...]
    names: tuple[str, ...]
    defaults: dict[str, Any]
    changed: str
    timeout: int


def parse_request(request: str) -> ConfigFacts:
    """Accept the existing benchmark grammar, not arbitrary natural language."""
    match = re.fullmatch(
        r"Generate the configuration\. Top-level fields in order: project=([\w-]+), (\w+)=an array\. "
        r"The (\w+) values in order are ([\w, ]+)\. "
        r"Each array entry has fields in this order: ([\w, ]+)\. "
        r"All entries have these defaults: (\{[^{}]+\})\. "
        r"Return the complete valid JSON with two-space indentation and one field per line, "
        r"without markdown fences or other text\. "
        r"Override only timeout_seconds for (\w+) to (\d+)\.", request)
    if match is None:
        raise ValueError("Request does not follow the frozen confirmation grammar")
    project, collection, key, names, fields, defaults, changed, timeout = match.groups()
    fields, names = tuple(fields.split(", ")), tuple(names.split(", "))
    defaults = json.loads(defaults)
    if (len(set(fields)) != len(fields) or len(set(names)) != len(names)
            or fields[0] != key or list(defaults) != list(fields[1:])
            or "timeout_seconds" not in defaults or changed not in names
            or any(type(v) not in (str, int, bool) for v in defaults.values())):
        raise ValueError("Inconsistent fields, names, defaults or override")
    return ConfigFacts(project, collection, fields, names, defaults, changed, int(timeout))


def candidates(request: str) -> dict[str, str]:
    """All resources depend only on request text; no project labels or answers."""
    facts = parse_request(request)
    outputs = {}
    for kind in ("Structure", "Static", "Expanded", "Compiled"):
        entries = []
        for index, name in enumerate(facts.names):
            entry = {}
            for field in facts.fields:
                if field == facts.fields[0]:
                    value = name if kind in {"Expanded", "Compiled"} else f"item_{index:02d}"
                else:
                    value = facts.defaults[field]
                    if kind == "Structure":
                        value = False if type(value) is bool else 0 if type(value) is int else "example"
                    if kind == "Compiled" and name == facts.changed and field == "timeout_seconds":
                        value = facts.timeout
                entry[field] = value
            entries.append(entry)
        outputs[kind] = json.dumps(
            {"project": facts.project if kind in {"Expanded", "Compiled"} else "example", facts.collection: entries}, indent=2)
    return outputs


def legal_snapshots(projects, initial, references):
    """Restore only the latest same-project artifact available before each request."""
    if len(initial) != len(projects) or len({e["user_id"] for e in initial}) != len(initial):
        raise ValueError("Missing or duplicate initial history")
    events = {e["user_id"]: MemoryEvent(**e) for e in initial}
    labels = {(r["project_id"], r["turn"]): r for r in references}
    expected = {(p["id"], t) for p in projects for t in range(len(p["currents"]))}
    if set(labels) != expected or len(labels) != len(references) or set(events) != {p["id"] for p in projects}:
        raise ValueError("Missing, duplicate or cross-project references/history")
    snapshots = {}
    for project in projects:
        event = events[project["id"]]
        for turn in range(len(project["currents"])):
            row = labels[project["id"], turn]
            order = 10 + 10 * turn
            if (event.order >= order or event.project_id != project["id"] or event.record_id != "artifact"
                    or event.source_role != "assistant" or event.status != "active"):
                raise ValueError("Future or cross-project history")
            snapshot = Snapshot(project["id"], order, (event,))
            if snapshot.snapshot_id != row["snapshot_id"] or row["order"] != order:
                raise ValueError("Frozen reference snapshot mismatch")
            snapshots[project["id"], turn] = snapshot
            event = MemoryEvent(project["id"], "artifact", order + 1, row["text"], project_id=project["id"])
    return snapshots


def resource_sets(request: str, snapshot: Snapshot) -> dict[str, list[str]]:
    generated = candidates(request)
    past = [r.text for r in snapshot.records]
    return {"Context": [], "Structure": [generated["Structure"]], "Static": [generated["Static"]],
            "Global": past, "StaticGlobal": [generated["Static"], *past],
            "Expanded": [generated["Expanded"]], "Compiled": [generated["Compiled"]]}


def coverage(tokens, prompt: str, resources: dict[str, list[str]], tokenizer):
    result, positions = {}, {}
    for method, texts in resources.items():
        size = sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)
        if size > EXTRA_TOKEN_BUDGET:
            raise ValueError("Candidate resource exceeds the fixed budget; never silently truncate")
        covered, lengths = oracle_coverage(tokens, continuation_map([prompt, *texts], tokenizer, ANCHOR, LIMIT),
                                           ANCHOR, MINIMUM)
        positions[method] = covered
        result[method] = dict(extra_tokens=size, covered=len(covered),
                              prefix_positions=len(lengths), sum_best_lengths=sum(lengths),
                              long_prefix_positions=sum(length >= MINIMUM for length in lengths),
                              max_best_length=max(lengths, default=0))
    for method in ("Context", "Structure", "Static", "Expanded", "Compiled"):
        result[method]["history_outside"] = len(positions["Global"] - positions[method])
    return result, positions
