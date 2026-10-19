"""Frozen independent-project inputs and audits for prefix-risk handoff selection."""
from __future__ import annotations

import json

from memorydraft.evaluation.guided_validation import audit as guided_audit
from memorydraft.evaluation.runtime_migration import file_hash
from memorydraft.evaluation.transport_validation import ValidationTask
from memorydraft.memory.store import MemoryEvent, Snapshot
from memorydraft.runtime.contracts import GenerationRequest
from memorydraft.evaluation.splice_probe import prefix_acceptance
from memorydraft.retrieval.prefix_risk import PrefixRiskPolicy

METHODS = ("Suffix-R", "Guided-Edit", "Splice-Edit", "Risk-NoConfidence", "Risk-Splice")
RETRIEVAL = dict(zip(METHODS, ("suffix", "edit-guided", "edit-splice", "edit-risk-splice", "edit-risk-splice")))


def inputs(root):
    data = json.loads((root / "data/research/prefix-risk/cases.json").read_text())
    calibration = json.loads((root / "data/research/prefix-risk/calibration.json").read_text())
    projects = data["projects"]
    if (data["schema"] != 1 or len(projects) != 6 or len({p["id"] for p in projects}) != 6
            or {p["id"] for p in projects} & set(calibration["training_projects"])
            or sum(p["group"] == "edited-continuation" for p in projects) != 4
            or sum(p["group"] == "control" for p in projects) != 2
            or calibration["training_sha256"] != file_hash(root / "results/acceptance/collection/states.jsonl.gz")):
        raise ValueError("Changed frozen project split or training data")
    tasks, events, users = [], [], set()
    for p in projects:
        history = tuple(MemoryEvent(**e) for e in p["history"])
        user = p["requests"][0]["request"]["user_id"]
        if len(p["requests"]) != 2 or user in users:
            raise ValueError("Changed independent project coverage")
        users.add(user)
        for item in p["requests"]:
            request = GenerationRequest.from_dict(item["request"])
            old, new = item["edit"]["old"], item["edit"]["new"]
            if (request.user_id != user or request.max_new_tokens != 512
                    or p["source"].count(old) != 1 or p["source"].replace(old, new, 1) != item["expected_text"]
                    or p["source"] not in request.messages[-1]["content"]
                    or any(e.user_id != user or e.order >= request.order for e in history)):
                raise ValueError("Changed edit contract or legal memory boundary")
            tasks.append(ValidationTask(p["id"], p["group"], request, Snapshot(user, request.order, history).snapshot_id, item["expected_text"]))
        events.extend(history)
    if len({t.request.id for t in tasks}) != 12:
        raise ValueError("Missing independent requests")
    return tasks, events, calibration


def protected(root):
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root / "results").rglob("*"))
            if p.is_file() and len(p.relative_to(root / "results").parts) > 1
            and p.relative_to(root / "results").parts[0] not in {"raw", "prefix-risk"}}


def audit(rows, tasks, repeats, *, instrumented=False):
    checked = [{**r, "latency_seconds": r["instrumented_latency_seconds"]} for r in rows] if instrumented else rows
    summary = guided_audit(checked, tasks, METHODS, repeats)
    for r in rows:
        c = r["counters"]
        if r["method"] in METHODS[2:]:
            if (c["splice_prefix_accepted_tokens"] + c["splice_history_accepted_tokens"] != c["alignment_accepted_tokens"]
                    or not c["draft_rounds"] <= c["draft_forward_rounds"] <= c["rounds"]
                    or c.get("risk_switches", 0) > c.get("risk_evaluations", 0)):
                raise ValueError("Invalid risk/splice accounting")
    keys = ("rounds", "draft_forward_rounds", "alignment_calls", "alignment_candidates", "alignment_seconds",
            "splice_prefix_accepted_tokens", "splice_history_accepted_tokens", "splice_zero_history",
            "risk_evaluations", "risk_switches", "risk_unsafe_unchanged", "risk_single_boundary")
    for group, methods in summary.items():
        selected = rows if group == "all" else [r for r in rows if r["group"] == group]
        for method, v in methods.items():
            v["counters"].update({k: sum(r["counters"].get(k, 0) for r in selected if r["method"] == method) for k in keys})
            if instrumented:
                v.pop("seconds")
    return summary


def timing_gate(rows):
    # Timing remains informative for a single independent change. If the policy
    # never changes any candidate, stop rather than measuring an unchanged path.
    switches = sum(r["counters"].get("risk_switches", 0) for r in rows if r["method"] == "Risk-Splice")
    return dict(passed=switches > 0, switches=switches)


def audit_decisions(rows, tasks, calibration):
    references = {t.request.id: t.reference["token_ids"] for t in tasks}
    for row in rows:
        if row["method"] not in calibration["models"]:
            if row["decisions"]:
                raise ValueError("Risk decision recorded for a baseline")
            continue
        policy = PrefixRiskPolicy(**calibration["models"][row["method"]])
        calls = {c["generated"]: c for c in row["calls"]}
        switched, evaluations, singles, unsafe = 0, 0, 0, 0
        for d in row["decisions"]:
            pool, draft = d["candidates"], d["draft_tokens"]
            remaining = references[row["request_id"]][d["generated"]:]
            if d["draft_accepted"] != prefix_acceptance(draft, remaining):
                raise ValueError("Changed post-hoc Draft label")
            if pool != sorted(pool, key=lambda c: (-c["cheap"], c["prefix_length"], c["position"])):
                raise ValueError("Changed Edit candidate ordering")
            for c in pool:
                b = c["prefix_length"]
                if (not 4 <= b <= len(draft) < len(c["tokens"]) <= 32 or c["tokens"][:b] != draft[:b]
                        or c["cheap"] < .65 or c["accepted"] != prefix_acceptance(c["tokens"], remaining)):
                    raise ValueError("Changed raw splice or post-hoc candidate label")
            if len({c["prefix_length"] for c in pool}) > 1:
                if not policy.confidence and d["confidence"] != [[0., 0., 0.]] * len(draft):
                    raise ValueError("Changed confidence ablation")
                baseline, selected, curve = policy.select(pool, d["confidence"])
                if len(curve) != len(d["survival"]) or any(abs(a-b) > 1e-12 for a,b in zip(curve, d["survival"])):
                    raise ValueError("Changed calibrated prefix risk")
                evaluations += 1
                unsafe += int(selected == baseline and curve[pool[baseline]["prefix_length"] - 1] < .5)
            else:
                baseline = selected = 0
                if d["confidence"] or d["survival"]:
                    raise ValueError("Confidence computed for an identical boundary")
                singles += 1
            if (d["baseline"], d["selected"]) != (baseline, selected):
                raise ValueError("Changed risk-assisted handoff decision")
            p = calls[d["generated"]]["proposal"]
            if list(p["tokens"]) != pool[selected]["tokens"] or p["draft_prefix"] != pool[selected]["prefix_length"]:
                raise ValueError("Recorded decision differs from actual proposal")
            switched += int(selected != baseline)
        if (evaluations, switched, singles, unsafe) != tuple(row["counters"][k] for k in
                ("risk_evaluations", "risk_switches", "risk_single_boundary", "risk_unsafe_unchanged")):
            raise ValueError("Changed risk accounting")
