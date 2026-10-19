"""Frozen full-plan ablations and audits over already observed decoding states."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from memorydraft.evaluation.guided_validation import audit as paired_audit
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.evaluation.splice_probe import COUNTERS, inputs as splice_inputs, protected_hashes


METHODS = ("Suffix-R", "Guided-Edit", "Splice-Edit", "Splice-OT", "Splice-TailOT",
           "Splice-OrderOT", "Splice-PlanOT", "Splice-AllOT")
RETRIEVAL = dict(zip(METHODS, ("suffix", "edit-guided", "edit-splice", "ot-splice", "ot-tail-splice",
                              "ot-order-splice", "ot-plan-splice", "ot-all-splice")))
SCORED = METHODS[2:]
IDS = ("payment-runbook:1", "shipment-contract:1", "confirm-schema-0:0", "confirm-schema-0:1",
       "inventory-query:0", "inventory-query:1", "parcel-queue:0", "copy")


def inputs(root, *, subset=False):
    tasks, events = splice_inputs(root)
    if subset:
        tasks = [t for t in tasks if t.request.id in IDS]
        events = [e for e in events if e.user_id in {t.request.user_id for t in tasks}]
        if {t.request.id for t in tasks} != set(IDS):
            raise ValueError("Changed eight-request matrix scope")
    return tasks, events


def protected(root):
    files = protected_hashes(root, include_splice=True)
    files.update({p.relative_to(root).as_posix(): file_hash(p)
                  for p in sorted((root / "results/ot-splice-cooldown").rglob("*")) if p.is_file()})
    return files


def observed_rows(root):
    return [r for name in ("ot-splice", "ot-splice-cooldown")
            for r in read_rows(root / "results" / name / "mechanism/rows.jsonl") if r["method"] in SCORED[:3]]


def state_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def audit(rows, tasks, repeats, *, instrumented):
    checked = [{**r, "latency_seconds": r["instrumented_latency_seconds"]} for r in rows] if instrumented else rows
    summary = paired_audit(checked, tasks, METHODS, repeats)
    keys = (*COUNTERS, "matrix_rejections", "matrix_solves")
    for group, methods in summary.items():
        selected = rows if group == "all" else [r for r in rows if r["group"] == group]
        for method, values in methods.items():
            if instrumented:
                values.pop("seconds")
            values["counters"] = {k: sum(r["counters"].get(k, 0) for r in selected if r["method"] == method) for k in keys}
    for row in rows:
        if row["method"] in SCORED:
            c = row["counters"]
            if (not c["draft_rounds"] <= c["draft_forward_rounds"] <= c["rounds"]
                    or c["splice_prefix_accepted_tokens"] + c["splice_history_accepted_tokens"] != c["alignment_accepted_tokens"]
                    or c.get("matrix_rejections", 0) > c["alignment_candidates"]
                    or c.get("matrix_solves", 0) > 2 * c["alignment_candidates"]):
                raise ValueError("Invalid full-plan accounting")
    return summary


def timing_gate(rows):
    wins = {}
    for method in METHODS[-3:]:
        requests = []
        for request in sorted({r["request_id"] for r in rows}):
            keyed = {r["method"]: r["counters"] for r in rows if r["request_id"] == request}
            current = keyed[method]
            if any(current["rounds"] < keyed[base]["rounds"] or
                   current["rounds"] == keyed[base]["rounds"] and
                   current["alignment_proposals"] < keyed[base]["alignment_proposals"]
                   for base in ("Splice-OT", "Splice-TailOT")):
                requests.append(request)
        wins[method] = requests
    return dict(passed=any(wins.values()), requests=wins)
