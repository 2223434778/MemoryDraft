"""Project-excluded acceptance calibration; diagnostic predictions, not speedups."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from memorydraft.evaluation.benefit_probe import calibration
from memorydraft.evaluation.matrix_probe import state_digest
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.evaluation.splice_probe import inputs, prefix_acceptance
from memorydraft.retrieval.acceptance import AcceptancePredictor
from scripts.diagnose_transport_online import write

ROOT = Path(__file__).resolve().parents[1]


def metrics(curve, accepted):
    if not curve or any(not math.isfinite(p) or not 0 <= p <= 1 for p in curve):
        raise ValueError("Invalid survival probabilities")
    if any(b > a + 1e-12 for a, b in zip(curve, curve[1:])) or not 0 <= accepted <= len(curve):
        raise ValueError("Invalid survival order/label")
    prediction = sum(curve)
    return dict(predicted=prediction, actual=accepted, error=abs(prediction - accepted), bias=prediction - accepted,
                brier=sum((p - float(accepted >= k)) ** 2 for k, p in enumerate(curve, 1)) / len(curve))


def old_curve(samples, row):
    nearest = {}
    for sample in samples:
        if sample["project"] == row["project_id"]:
            continue
        distance = sum((a - b) ** 2 for a, b in zip(sample["features"], row["features"]))
        if sample["project"] not in nearest or distance < nearest[sample["project"]][0]:
            nearest[sample["project"]] = distance, sample
    chosen = sorted(nearest.values(), key=lambda v: (v[0], v[1]["project"]))[:3]
    weights = [1 / (.25 + d) for d, _ in chosen]
    route, length = row["choice"]["route"], row["choice"]["length"]
    curve = [sum(w * float(s["outcomes"][route][0] >= k) for w, (_, s) in zip(weights, chosen)) / sum(weights)
             for k in range(1, length + 1)]
    if not math.isclose(sum(curve), row["predicted_accepted"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Old predictor reconstruction changed")
    return curve


def audit_states(states, tasks):
    expected = {t.request.id: t for t in tasks}
    seen = set()
    for s in states:
        t = expected[s["request_id"]]
        if s["project_id"] != t.project_id or s["snapshot_id"] != t.snapshot_id:
            raise ValueError("Changed project/snapshot")
        if s["confirmed_sha256"] != state_digest(t.reference["token_ids"][:s["generated"]]):
            raise ValueError("Changed confirmed prefix")
        remaining = t.reference["token_ids"][s["generated"]:]
        if len(s["confidence"]) != len(s["draft_tokens"]) or not s["confidence"]:
            raise ValueError("Missing Draft confidence")
        if any(len(v) != 3 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in v) for v in s["confidence"]):
            raise ValueError("Invalid confidence")
        if s["draft_accepted"] != prefix_acceptance(s["draft_tokens"], remaining):
            raise ValueError("Changed Draft label")
        identity = state_digest([s["request_id"], s["generated"], s["draft_tokens"],
                                 [(c["tokens"], c["prefix_length"], c["position"]) for c in s["candidates"]]])
        if identity != s["state_sha256"] or identity in seen:
            raise ValueError("Duplicate/changed observation")
        seen.add(identity)
        for c in [*s["candidates"], *(c for frame in s["legacy"] for c in frame["candidates"])]:
            b = c["prefix_length"]
            if (not 0 < b <= len(s["draft_tokens"]) < len(c["tokens"])
                    or c["tokens"][:b] != s["draft_tokens"][:b]
                    or c["accepted"] != prefix_acceptance(c["tokens"], remaining)
                    or len(c["token_probability"]) != len(s["draft_tokens"])
                    or any(not math.isfinite(p) or not 0 <= p <= 1 for p in c["token_probability"])):
                raise ValueError("Invalid candidate features/labels")


def mean_metrics(rows, model):
    return {key: float(np.mean([r[model][key] for r in rows])) for key in ("error", "bias", "brier")}


def evaluate(states, root=ROOT, *, calibrated=None):
    tasks, _ = inputs(root)
    audit_states(states, tasks)
    projects = sorted({s["project_id"] for s in states})
    models = calibrated if calibrated is not None else {
        p: {name: AcceptancePredictor.fit(states, excluded_project=p, confidence=confidence)
            for name, confidence in (("no_confidence", False), ("confidence", True))} for p in projects}
    if set(models) != set(projects):
        raise ValueError("Missing project-excluded model")
    for project, fitted in models.items():
        if set(fitted) != {"confidence", "no_confidence"} or any(
                set(m.projects) != set(projects) - {project} or m.confidence != (name == "confidence")
                for name, m in fitted.items()):
            raise ValueError("Calibration leaked a project or changed the ablation")
    draft, candidates = [], []
    for s in states:
        fitted = models[s["project_id"]]
        identity = {k: s[k] for k in ("request_id", "project_id", "generated", "state_sha256")}
        draft.append({**identity, **{name: metrics(m.draft_survival(s), s["draft_accepted"]) for name, m in fitted.items()}})
        for i, c in enumerate(s["candidates"]):
            candidates.append({**identity, "candidate": i,
                               **{name: metrics(m.candidate_survival(s, c), c["accepted"]) for name, m in fitted.items()}})
    legacy = {}
    for s in states:
        for frame in s["legacy"]:
            legacy.setdefault(frame["frame"], (s, frame))
    frames = read_rows(root / "results/ot-matrix/offline/frames.jsonl")
    old = {(r["request_id"], r["generated"]): r for r in read_rows(root / "results/ot-benefit/offline/rows.jsonl")}
    samples = calibration(root)
    paired, ordering = [], []
    for f in frames:
        if f["state_sha256"] not in legacy:
            raise ValueError("Missing original fixed-action observation: " + f["state_sha256"])
        s, observed = legacy[f["state_sha256"]]
        row = old[(f["request_id"], f["generated"])]
        for a, b in zip(f["candidates"], observed["candidates"], strict=True):
            if any(a[k] != b[k] for k in ("tokens", "prefix_length", "position", "cheap", "source")):
                raise ValueError("Changed old fixed candidate")
        route, length = row["choice"]["route"], row["choice"]["length"]
        selected = None if route == "draft" else f["decisions"][{"edit": "Splice-Edit", "ot": "Splice-TailOT"}[route]]["selected"]
        c = observed["candidates"][selected] if selected is not None else None
        actual = min(c["accepted"] if c else s["draft_accepted"], length)
        if actual != row["actual_accepted"]:
            raise ValueError("Changed old fixed action label")
        fitted = models[s["project_id"]]
        pair = {"request_id": f["request_id"], "project_id": f["project_id"], "generated": f["generated"],
                "route": route, "length": length, "old": metrics(old_curve(samples, row), actual)}
        for name, model in fitted.items():
            curve = model.candidate_survival(s, c) if c else model.draft_survival(s)
            pair[name] = metrics(curve[:length], actual)
        paired.append(pair)
        options = {}
        for method in ("Splice-Edit", "Splice-TailOT"):
            selected = f["decisions"][method]["selected"]
            c = observed["candidates"][selected] if selected is not None else None
            options[method] = {"actual": c["accepted"] if c else s["draft_accepted"],
                               **{name: sum(m.candidate_survival(s, c) if c else m.draft_survival(s)) for name, m in fitted.items()}}
        ordering.append({"request_id": f["request_id"], "generated": f["generated"], "options": options})
    project_deltas = [np.mean([r["old"]["error"] - r["confidence"]["error"] for r in paired if r["project_id"] == p])
                      for p in sorted({r["project_id"] for r in paired})]
    rng = np.random.default_rng(20261019)
    boot = rng.choice(project_deltas, size=(5000, len(project_deltas)), replace=True).mean(axis=1)
    interval = np.quantile(boot, [.025, .975]).tolist()
    summary = {name: mean_metrics(paired, name) for name in ("old", "no_confidence", "confidence")}
    a, b, c = summary["old"], summary["no_confidence"], summary["confidence"]
    gate = dict(coverage=len(paired) == 14, mae_20_percent=c["error"] <= .8 * a["error"], brier_better_old=c["brier"] < a["brier"],
                confidence_adds_value=c["error"] < b["error"] and c["brier"] < b["brier"], project_interval_positive=interval[0] > 0)
    gate["passed"] = all(gate.values())
    return dict(paired=paired, ordering=ordering, models={p: {name: asdict(m) for name, m in v.items()} for p, v in models.items()},
                draft=draft, candidates=candidates,
                summary=dict(states=len(states), candidates=len(candidates), projects=len(projects), paired=summary,
                             broad={scope: {name: mean_metrics(rows, name) for name in ("no_confidence", "confidence")}
                                    for scope, rows in (("draft", draft), ("candidates", candidates))},
                             project_mean_mae_improvement=float(np.mean(project_deltas)), project_bootstrap_95=interval, gate=gate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads((args.input / "provenance.json").read_text())
    if provenance["status"] != "complete" or provenance["rows"] != 48 or provenance["previous_paths_checked"] != 24:
        raise ValueError("Incomplete strict diagnostic collection")
    result = evaluate(read_rows(args.input / "states.jsonl"))
    args.output.mkdir(parents=True, exist_ok=False)
    for key, value in result.items():
        write(args.output / (key + ".json"), value)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
