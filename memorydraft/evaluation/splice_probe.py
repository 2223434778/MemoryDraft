"""A bounded, explicitly post-hoc splice experiment over already observed tasks."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from memorydraft.evaluation.guided_validation import COUNTERS as GUIDED_COUNTERS, audit as guided_audit
from memorydraft.evaluation.guided_validation import load_inputs as guided_inputs
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.evaluation.transport_validation import attach_references


METHODS = ("Suffix-R", "Guided-Edit", "Guided-OT", "Splice-Edit", "Splice-OT", "Splice-TailOT")
RETRIEVAL = dict(zip(METHODS, ("suffix", "edit-guided", "ot-guided", "edit-splice", "ot-splice", "ot-tail-splice")))
COUNTERS = (*GUIDED_COUNTERS, "splice_proposals", "splice_internal_proposals", "splice_prefix_accepted_tokens",
            "splice_history_accepted_tokens", "splice_extension_accepted_tokens", "splice_zero_history")


def inputs(root):
    from scripts.diagnose_transport_online import inputs as old_inputs

    tasks, events = old_inputs(root)
    new, history = guided_inputs(root / "data/research/ot-guided/cases.json")
    new = attach_references(new, read_rows(root / "results/ot-guided/holdout/fresh-ar.jsonl"))
    new = [t for t in new if t.group == "edited-continuation"]
    tasks += new
    users = {t.request.user_id for t in new}
    events += [e for e in history if e.user_id in users]
    if len(tasks) != 16 or len({t.request.id for t in tasks}) != 16:
        raise ValueError("Expected the frozen sixteen already observed requests")
    return tasks, events


def input_digest(tasks, events):
    value = dict(tasks=[asdict(t) for t in tasks], events=[asdict(e) for e in events])
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def protected_hashes(root):
    return {p.relative_to(root).as_posix(): file_hash(p) for name in
            ("ot-r", "ot-validation", "ot-diagnosis", "ot-guided", "infra-v0.3", "reuse-confirmation")
            for p in sorted((root / "results" / name).rglob("*")) if p.is_file()}


def prefix_acceptance(tokens, reference):
    return next((i for i, (a, b) in enumerate(zip(tokens, reference)) if a != b), min(len(tokens), len(reference)))


def audit(rows, tasks, repeats, *, instrumented=False):
    checked = [{**r, "latency_seconds": r["instrumented_latency_seconds"]} for r in rows] if instrumented else rows
    summary = guided_audit(checked, tasks, METHODS, repeats)
    for group, methods in summary.items():
        selected = rows if group == "all" else [r for r in rows if r["group"] == group]
        for method, values in methods.items():
            if instrumented:
                values.pop("seconds")
            values["counters"] = {k: sum(r["counters"].get(k, 0) for r in selected if r["method"] == method)
                                  for k in COUNTERS}
    for row in rows:
        if row["method"].startswith("Splice-"):
            c = row["counters"]
            if (not c["draft_rounds"] <= c["draft_forward_rounds"] <= c["rounds"]
                    or c["splice_prefix_accepted_tokens"] + c["splice_history_accepted_tokens"] != c["alignment_accepted_tokens"]
                    or not 0 <= c["splice_extension_accepted_tokens"] <= c["splice_history_accepted_tokens"]
                    or c["splice_internal_proposals"] > c["splice_proposals"]):
                raise ValueError("Invalid splice attribution")
    return summary


def mechanism_gate(rows):
    """Require two requests with fewer rounds than BOTH guided baselines, per arm."""
    improvements = {}
    for method in METHODS[3:]:
        won = []
        for request in sorted({r["request_id"] for r in rows}):
            keyed = {r["method"]: r for r in rows if r["request_id"] == request}
            candidate = keyed[method]
            if (candidate["counters"]["rounds"] < min(keyed[m]["counters"]["rounds"] for m in METHODS[1:3])
                    and any(c["proposal"] and c["proposal"]["draft_prefix"] > 4
                            and c["oracle_accepted"] > c["draft_oracle_accepted"] for c in candidate["calls"])):
                won.append(request)
        improvements[method] = won
    return dict(passed=any(len(v) >= 2 for v in improvements.values()), improved_requests=improvements)
