"""Full-plan lexical OT experiments; these scores never authorize token acceptance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from memorydraft.retrieval.splice import SpliceReuseMatcher


@dataclass(frozen=True)
class MatrixAssessment:
    score: float
    admissible: bool
    features: dict[str, float]


def local_order_penalty(matched, radius=4):
    """Neighbor-row matched flow supporting an increasing local step, not hard monotonicity."""
    n, m = matched.shape
    mass = matched.sum(axis=1)
    conditional = np.divide(matched, mass[:, None], out=np.zeros_like(matched), where=mass[:, None] > 1e-12)
    support, neighbors = np.zeros_like(matched), np.zeros((n, 1))
    for row in range(n):
        if row and mass[row - 1] > 1e-12:
            for shift in range(1, min(radius, m - 1) + 1):
                support[row, shift:] += conditional[row - 1, :-shift]
            neighbors[row] += 1
        if row + 1 < n and mass[row + 1] > 1e-12:
            for shift in range(1, min(radius, m - 1) + 1):
                support[row, :-shift] += conditional[row + 1, shift:]
            neighbors[row] += 1
    return np.clip(np.divide(neighbors - support, neighbors, out=np.zeros_like(matched), where=neighbors > 0), 0, 1)


def plan_features(plan, equal, a, *, draft_prefix, hint_tokens=4):
    """Measure support BEFORE the forced exact anchor, avoiding tautological evidence."""
    n, m = equal.shape
    if not 0 <= draft_prefix <= n or not 1 <= hint_tokens <= n:
        raise ValueError("Invalid query/anchor boundaries")
    matched = plan[:n, :m] * equal
    mass = matched.sum(axis=1)
    row_support = np.clip(mass / a[:n], 0, 1)
    concentration = np.divide(matched.max(axis=1), mass, out=np.zeros(n), where=mass > 1e-12)
    dropped = np.clip(plan[:n, m] / a[:n], 0, 1)
    end, start = n - hint_tokens, max(0, n - hint_tokens - 8)
    if end <= start:
        return dict(support=0., concentration=0., continuity=0., dropped=1., draft_weakest=0.)
    conditional = np.divide(matched, mass[:, None], out=np.zeros_like(matched), where=mass[:, None] > 1e-12)
    links = []
    # Include the link entering the anchor, but not links wholly inside it.
    for row in range(start, min(end, n - 1)):
        value = sum(float(np.dot(conditional[row, :-shift], conditional[row + 1, shift:]))
                    for shift in range(1, min(hint_tokens, m - 1) + 1))
        links.append(value * min(row_support[row], row_support[row + 1]))
    draft_start = n - draft_prefix
    return dict(support=float(row_support[start:end].mean()),
                concentration=float(concentration[start:end].mean()),
                continuity=float(np.clip(np.mean(links), 0, 1)) if links else 0.,
                dropped=float(dropped[start:end].mean()),
                draft_weakest=float(row_support[draft_start:end].min()) if draft_start < end else 1.)


def assess_alignment(query, history, config, solver, *, draft_prefix, tail=False, order=False, structure=False):
    if not query or not history:
        return MatrixAssessment(float("-inf"), False, {})
    n, m = len(query), len(history)
    equal = np.asarray(query)[:, None] == np.asarray(history)[None, :]
    distance = np.abs(np.arange(n)[:, None] - n + 1 - (np.arange(m)[None, :] - m + 1)) / max(n, m)
    cost = np.full((n + 1, m + 1), config.drop_cost, dtype=np.float64)
    cost[:n, :m] = (~equal).astype(np.float64) + config.position_weight * distance
    cost[-1, -1] = 0
    a, b = np.r_[np.full(n, .5 / n), .5], np.r_[np.full(m, .5 / m), .5]
    for iteration in range(2 if order else 1):
        plan = solver(a, b, cost, config.entropy, numItermax=config.iterations, stopThr=1e-7, warn=False)
        if (plan.shape != cost.shape or not np.isfinite(plan).all() or (plan < 0).any()
                or max(np.abs(plan.sum(axis=1) - a).max(), np.abs(plan.sum(axis=0) - b).max()) > 1e-3):
            return MatrixAssessment(float("-inf"), False, {})
        if order and iteration == 0:
            cost[:n, :m] += config.position_weight * local_order_penalty(plan[:n, :m] * equal)
    global_score = float((plan[:n, :m] * equal).sum() / .5
                         - config.position_weight * (plan[:n, :m] * distance).sum() / .5)
    contribution = (plan[:n, :m] * (equal - config.position_weight * distance)).sum(axis=1)
    count = min(8, n)
    score = (float(.5 * contribution.sum() / .5 + .5 * contribution[-count:].sum() / a[-count-1:-1].sum())
             if tail else global_score)
    features = plan_features(plan, equal, a, draft_prefix=draft_prefix)
    admissible = True
    if structure:
        # Fixed majority-support rule, not a fitted or calibrated probability.
        admissible = (features["support"] >= .5 and features["concentration"] >= .5
                      and features["continuity"] >= .5 and features["dropped"] <= .5)
        reliability = .4 * features["support"] + .3 * features["concentration"] + .3 * features["continuity"]
        score = .5 * score + .5 * reliability
    return MatrixAssessment(score if admissible else float("-inf"), admissible, features)


class MatrixReuseMatcher(SpliceReuseMatcher):
    def __init__(self, index, *, variant, **kwargs):
        if variant not in {"order", "plan", "all"}:
            raise ValueError("Unknown matrix ablation")
        super().__init__(index, method="ot", **kwargs)
        self.variant, self.lookup = variant, "ot-" + variant + "-splice"
        self.stats.update(matrix_rejections=0, matrix_solves=0)

    def candidate_score(self, item):
        self.stats["matrix_solves"] += 2 if self.variant in {"order", "all"} else 1
        assessment = assess_alignment(item["query"], item["history"], self.config, self.solver,
                                      draft_prefix=item["prefix_length"], tail=self.variant == "all",
                                      order=self.variant in {"order", "all"}, structure=self.variant in {"plan", "all"})
        self.stats["matrix_rejections"] += int(not assessment.admissible)
        return assessment.score
