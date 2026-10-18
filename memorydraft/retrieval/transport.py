"""Bounded lexical transport alignment for retrieval proposals, never acceptance.

POT solves entropic OT with a dustbin on each side. The dustbins let edits remain
unmatched; a positional ground cost favors locally ordered token matches. This
is a retrieval heuristic, not SpecTr's distribution-coupling verifier or OPW.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import time
from typing import Sequence

import numpy as np

from memorydraft.retrieval.reuse import ReuseMatcher, ReuseProposal


@dataclass(frozen=True)
class AlignmentConfig:
    window: int = 32
    min_context: int = 8
    shortlist: int = 8
    anchor_tokens: int = 4
    positions_per_anchor: int = 64
    max_shift: int = 2
    max_fallback_calls: int = 4
    minimum_score: float = 0.65
    position_weight: float = 0.25
    entropy: float = 0.06
    drop_cost: float = 0.35
    iterations: int = 80

    def __post_init__(self):
        integer_limits = (("window", 128), ("min_context", 128), ("shortlist", 32),
                          ("anchor_tokens", 16), ("positions_per_anchor", 256),
                          ("max_fallback_calls", 64), ("iterations", 500))
        if any(type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= maximum
               for key, maximum in integer_limits):
            raise ValueError("Invalid alignment budget")
        if (self.min_context > self.window or type(self.max_shift) is not int or not 0 <= self.max_shift <= 8
                or not math.isfinite(self.minimum_score) or not 0 <= self.minimum_score <= 1
                or not math.isfinite(self.position_weight) or not 0 <= self.position_weight <= 2
                or not math.isfinite(self.entropy) or not 0.01 <= self.entropy <= 1
                or not math.isfinite(self.drop_cost) or not 0 < self.drop_cost <= 2):
            raise ValueError("Invalid alignment controls")


def edit_score(query: Sequence[int], history: Sequence[int]) -> float:
    """Unit-cost Levenshtein similarity with linear auxiliary memory."""
    previous = list(range(len(history) + 1))
    for i, token in enumerate(query, 1):
        row = [i]
        for j, old in enumerate(history, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (token != old)))
        previous = row
    return 1 - previous[-1] / max(len(query), len(history), 1)


def transport_score(query: Sequence[int], history: Sequence[int], config: AlignmentConfig,
                    solver=None) -> float:
    """Score token identity carried by a finite, approximately balanced OT plan."""
    if not query or not history:
        return 0.0
    if solver is None:
        from ot import sinkhorn
        solver = sinkhorn
    n, m = len(query), len(history)
    equal = np.asarray(query)[:, None] == np.asarray(history)[None, :]
    distance = np.abs(np.arange(n)[:, None] - n + 1 - (np.arange(m)[None, :] - m + 1)) / max(n, m)
    cost = np.full((n + 1, m + 1), config.drop_cost, dtype=np.float64)
    cost[:n, :m] = (~equal).astype(np.float64) + config.position_weight * distance
    cost[-1, -1] = 0
    a, b = np.r_[np.full(n, 0.5 / n), 0.5], np.r_[np.full(m, 0.5 / m), 0.5]
    plan = solver(a, b, cost, config.entropy, numItermax=config.iterations, stopThr=1e-7, warn=False)
    if (plan.shape != cost.shape or not np.isfinite(plan).all() or (plan < 0).any()
            or max(np.abs(plan.sum(axis=1) - a).max(), np.abs(plan.sum(axis=0) - b).max()) > 1e-3):
        return float("-inf")
    # This score is an alignment heuristic, not an acceptance probability.
    return float((plan[:n, :m] * equal).sum() / 0.5
                 - config.position_weight * (plan[:n, :m] * distance).sum() / 0.5)


class AlignedReuseMatcher(ReuseMatcher):
    def __init__(self, index, *, method: str, exact_min_match: int = 6,
                 config: AlignmentConfig = AlignmentConfig(), cursor: bool = False):
        if method not in {"ot", "edit"} or exact_min_match < index.min_match:
            raise ValueError("Invalid alignment method or exact match threshold")
        super().__init__(index, cursor=cursor)
        self.method, self.config, self.exact_min_match = method, config, exact_min_match
        self.solver = None
        if method == "ot":
            try:
                from ot import sinkhorn
            except ImportError as error:
                raise ImportError('OT retrieval requires the optional dependency: pip install ".[transport]"') from error
            self.solver = sinkhorn
        self.positions = defaultdict(list)
        self.bounds = {}
        for start, end, source in zip(index.starts, index.ends, index.sources):
            for position in range(start + config.min_context, end):
                self.positions[index.tokens[position - 1]].append(position)
                self.bounds[position] = (start, end, source)
        self.stats = dict(alignment_calls=0, alignment_candidates=0, alignment_seconds=0.0,
                          alignment_proposals=0, alignment_accepted_tokens=0)

    def shortlist(self, confirmed: Sequence[int]) -> list[int]:
        config = self.config
        query = tuple(confirmed[-config.window:])
        if len(query) < config.min_context:
            return []
        counts = Counter(query)
        # Rarer observed tokens avoid spending the bounded pool on JSON punctuation.
        anchors = sorted(range(len(query)), key=lambda offset: (
            len(self.positions.get(query[offset], ())) or 10**12, -offset))[:config.anchor_tokens]
        pool = set()
        for offset in anchors:
            remaining = len(query) - offset - 1
            for old in self.positions.get(query[offset], ())[:config.positions_per_anchor]:
                lower, upper, _ = self.bounds[old]
                for shift in range(-config.max_shift, config.max_shift + 1):
                    position = old + remaining + shift
                    if lower + config.min_context <= position < upper:
                        pool.add(position)
        def key(position):
            lower, _, _ = self.bounds[position]
            history = self.index.tokens[max(lower, position - len(query)):position]
            overlap = sum((counts & Counter(history)).values()) / max(len(query), len(history))
            return (-overlap, -int(query[-1] == history[-1]), position)
        return sorted(pool, key=key)[:config.shortlist]

    def rank(self, confirmed: Sequence[int], max_tokens: int) -> ReuseProposal | None:
        """One bounded decision; the offline diagnostic calls this without an online budget."""
        if max_tokens < 1:
            return None
        prefix = tuple(confirmed[-self.config.window:])
        best = None
        for position in self.shortlist(prefix):
            if self.blocked == (tuple(confirmed[-self.index.max_match:]), position):
                continue
            lower, end, source = self.bounds[position]
            history = tuple(self.index.tokens[max(lower, position - len(prefix)):position])
            self.stats["alignment_candidates"] += 1
            score = (transport_score(prefix, history, self.config, self.solver) if self.method == "ot"
                     else edit_score(prefix, history))
            candidate = (score, -position)
            if best is None or candidate > best[0]:
                best = (candidate, position, end, source, history)
        if best is None or best[0][0] < self.config.minimum_score:
            return None
        _, position, end, source, history = best
        matched = 0
        for current, old in zip(reversed(prefix), reversed(history)):
            if current != old:
                break
            matched += 1
        return ReuseProposal(tuple(self.index.tokens[position:min(position + max_tokens, end)]),
                             source, matched, position, end, self.method, context_qualified=True)

    def propose(self, confirmed: Sequence[int], max_tokens: int) -> ReuseProposal | None:
        exact = super().propose(confirmed, max_tokens)
        if exact is not None and exact.matched >= self.exact_min_match:
            return exact
        if (len(confirmed) < self.config.min_context or max_tokens < 1
                or self.stats["alignment_calls"] >= self.config.max_fallback_calls):
            return None
        started = time.perf_counter()
        self.stats["alignment_calls"] += 1
        try:
            proposal = self.rank(confirmed, max_tokens)
            self.stats["alignment_proposals"] += int(proposal is not None)
            return proposal
        finally:
            self.stats["alignment_seconds"] += time.perf_counter() - started

    def observe(self, proposal, accepted, confirmed):
        super().observe(proposal, accepted, confirmed)
        if proposal.lookup in {"ot", "edit"}:
            self.stats["alignment_accepted_tokens"] += accepted
