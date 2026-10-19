"""Candidate-specific, conditional survival calibration for diagnostic use.

S(k)=P(accepted>=k). A splice shares the Draft survival up to its boundary;
its later survival is P(Draft reaches boundary)*P(history survives | boundary).
No OT plan, answer, user ID or project ID is an inference feature.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def prefix_features(state, length, confidence):
    values = state["confidence"][:length]
    extra = [min(v[0] for v in values), sum(v[0] for v in values) / length,
             max(v[1] for v in values), min(v[2] for v in values)] if confidence else [0.] * 4
    return [1., length / 15, *extra]


def tail_features(state, candidate, length, confidence):
    boundary = candidate["prefix_length"]
    draft, tokens = state["draft_tokens"], candidate["tokens"]
    agreement = next((i for i, (a, b) in enumerate(zip(draft, tokens)) if a != b), min(len(draft), len(tokens)))
    observed = min(length, len(draft))
    probabilities = candidate["token_probability"][boundary:observed]
    extra = [min(probabilities), sum(probabilities) / len(probabilities)] if confidence and probabilities else [0., 0.]
    return [1., (length - boundary) / 32, boundary / 15, candidate["cheap"], agreement / 15,
            float(candidate["source"] == "context:current-prompt"),
            (observed - boundary) / (length - boundary), *extra]


def _fit(rows, width):
    from scipy.optimize import minimize

    if not rows:
        raise ValueError("No observed conditional survival labels")
    x = np.asarray([r[0] for r in rows], dtype=np.float64)
    y = np.asarray([r[1] for r in rows], dtype=np.float64)
    weights = np.asarray([r[2] for r in rows], dtype=np.float64)
    weights /= weights.sum()
    penalty = np.r_[.0001, np.full(width - 1, .01)]

    def objective(beta):
        scores = x @ beta
        loss = np.sum(weights * (np.logaddexp(0, scores) - y * scores)) + .5 * np.sum(penalty * beta * beta)
        gradient = x.T @ (weights * (_sigmoid(scores) - y)) + penalty * beta
        return float(loss), gradient

    fitted = minimize(objective, np.zeros(width), jac=True, method="L-BFGS-B",
                      options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8})
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError("Survival calibration failed: " + str(fitted.message))
    return fitted.x.tolist()


@dataclass(frozen=True)
class AcceptancePredictor:
    prefix_weights: list[float]
    tail_weights: list[float]
    confidence: bool
    projects: tuple[str, ...]

    def __post_init__(self):
        if (len(self.prefix_weights) != 6 or len(self.tail_weights) != 9
                or not all(math.isfinite(x) for x in [*self.prefix_weights, *self.tail_weights])
                or type(self.confidence) is not bool):
            raise ValueError("Invalid survival model")

    @classmethod
    def fit(cls, states, *, excluded_project, confidence=True):
        train = [s for s in states if s["project_id"] != excluded_project]
        projects = tuple(sorted({s["project_id"] for s in train}))
        if len(projects) < 3:
            raise ValueError("Need at least three other calibration projects")
        counts = Counter(s["project_id"] for s in train)
        prefix, tail = [], []
        for s in train:
            size = len(s["draft_tokens"])
            accepted = s["draft_accepted"]
            if not 0 <= accepted <= size:
                raise ValueError("Invalid Draft acceptance label")
            weight = 1 / counts[s["project_id"]]
            # All cutoffs up to the observed span are labelled. Full acceptance
            # is right-censored: it never invents rejection at the next token.
            for k in range(1, size + 1):
                prefix.append((prefix_features(s, k, confidence), float(accepted >= k), weight / size))
            candidates = [c for c in s["candidates"] if c["accepted"] >= c["prefix_length"]]
            for c in candidates:
                boundary, end = c["prefix_length"], len(c["tokens"])
                if not 0 < boundary <= size < end or not 0 <= c["accepted"] <= end:
                    raise ValueError("Invalid historical survival label")
                for k in range(boundary + 1, end + 1):
                    tail.append((tail_features(s, c, k, confidence), float(c["accepted"] >= k),
                                 weight / len(candidates) / (end - boundary)))
        return cls(_fit(prefix, 6), _fit(tail, 9), confidence, projects)

    def draft_survival(self, state):
        x = np.asarray([prefix_features(state, k, self.confidence) for k in range(1, len(state["draft_tokens"]) + 1)])
        return np.minimum.accumulate(_sigmoid(x @ self.prefix_weights)).tolist()

    def candidate_survival(self, state, candidate):
        boundary, length = candidate["prefix_length"], len(candidate["tokens"])
        if not 0 < boundary <= len(state["draft_tokens"]) < length:
            raise ValueError("Expected a retained Draft prefix and longer raw-history candidate")
        prefix = self.draft_survival(state)[:boundary]
        x = np.asarray([tail_features(state, candidate, k, self.confidence) for k in range(boundary + 1, length + 1)])
        conditional = np.minimum.accumulate(_sigmoid(x @ self.tail_weights))
        return prefix + (prefix[-1] * conditional).tolist()
