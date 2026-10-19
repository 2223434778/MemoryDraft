"""Project-excluded calibration and bounded development checks for cost routing."""
from __future__ import annotations

from memorydraft.evaluation.guided_validation import audit as paired_audit
from memorydraft.evaluation.matrix_probe import inputs, observed_rows, protected as prior_protected
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.retrieval.benefit import BenefitPolicy, pool_features


METHODS = ("Suffix-R", "Guided-Edit", "Splice-Edit", "Splice-TailOT", "Gate-OT", "Budget-Edit", "Budget-OT")
RETRIEVAL = dict(zip(METHODS, ("suffix", "edit-guided", "edit-splice", "ot-tail-splice",
                              "ot-gate-splice", "edit-budget-splice", "ot-budget-splice")))
BENEFIT_COUNTERS = ("benefit_decisions", "benefit_ot_calls", "benefit_ot_candidates", "benefit_draft", "benefit_edit", "benefit_short")


def protected(root):
    hashes = prior_protected(root)
    hashes.update({p.relative_to(root).as_posix(): file_hash(p)
                   for p in sorted((root / "results/ot-matrix").rglob("*")) if p.is_file()})
    return hashes


def calibration(root):
    rows = observed_rows(root)
    samples = []
    for f in read_rows(root / "results/ot-matrix/offline/frames.jsonl"):
        drafts = [c["draft_tokens"] for r in rows if r["request_id"] == f["request_id"]
                  for c in r["calls"] if c["generated"] == f["generated"]
                  and c["draft_oracle_accepted"] == f["draft_accepted"]]
        draft = next(d for d in drafts if all(c["tokens"][:c["prefix_length"]] == d[:c["prefix_length"]]
                                           for c in f["candidates"]))
        outcomes = dict(draft=[f["draft_accepted"], len(draft)])
        for route, method in (("edit", "Splice-Edit"), ("ot", "Splice-TailOT")):
            selected = f["decisions"][method]["selected"]
            outcomes[route] = [f["decisions"][method]["accepted"],
                               len(f["candidates"][selected]["tokens"]) if selected is not None else len(draft)]
        samples.append(dict(project=f["project_id"], features=pool_features(f["candidates"], draft),
                            outcomes=outcomes, request_id=f["request_id"], generated=f["generated"]))
    return samples


def policy_data(samples, costs, excluded_project):
    kept = [{k: s[k] for k in ("project", "features", "outcomes")} for s in samples if s["project"] != excluded_project]
    data = dict(samples=kept, costs=costs)
    BenefitPolicy(**data)
    if any(s["project"] == excluded_project for s in data["samples"]):
        raise ValueError("Current project leaked into calibration")
    return data


def audit(rows, tasks, repeats, *, instrumented):
    from memorydraft.evaluation.splice_probe import COUNTERS

    checked = [{**r, "latency_seconds": r["instrumented_latency_seconds"]} for r in rows] if instrumented else rows
    summary = paired_audit(checked, tasks, METHODS, repeats)
    for group, methods in summary.items():
        selected = rows if group == "all" else [r for r in rows if r["group"] == group]
        for method, value in methods.items():
            if instrumented:
                value.pop("seconds")
            value["counters"] = {k: sum(r["counters"].get(k, 0) for r in selected if r["method"] == method)
                                 for k in (*COUNTERS, *BENEFIT_COUNTERS)}
    for r in rows:
        c = r["counters"]
        is_splice = r["method"].startswith("Splice-") or r["method"] in METHODS[-3:]
        if (is_splice and c.get("splice_prefix_accepted_tokens", 0) + c.get("splice_history_accepted_tokens", 0)
                != c.get("alignment_accepted_tokens", 0) or c.get("benefit_ot_candidates", 0) > c.get("alignment_candidates", 0)):
            raise ValueError("Invalid benefit accounting")
    return summary


def timing_gate(rows):
    # This small attempt is informative even if the policy abstains from all OT:
    # require actual routing/length decisions, not a favourable measured latency.
    decisions = sum(r["counters"].get("benefit_decisions", 0) for r in rows)
    return dict(passed=decisions > 0, decisions=decisions)
