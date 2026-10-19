"""Bounded Draft-prefix / raw-history splicing; Target remains the sole verifier."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import time

import numpy as np

from memorydraft.retrieval.guided import GuidedReuseMatcher
from memorydraft.retrieval.reuse import ReuseProposal
from memorydraft.retrieval.transport import edit_score, transport_score


@dataclass(frozen=True)
class SpliceProposal(ReuseProposal):
    # position is the first historical token AFTER this retained Draft prefix.
    draft_prefix: int = 0
    original_draft: int = 0


def tail_transport_score(query, history, config, solver):
    """Same lexical Sinkhorn plan, with half the score assigned to the last 8 rows.

    This is endpoint emphasis, not a monotonic OT solver or acceptance probability.
    """
    if not query or not history:
        return 0.0
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
    contribution = (plan[:n, :m] * (equal - config.position_weight * distance)).sum(axis=1)
    tail = min(8, n)
    return float(0.5 * contribution.sum() / 0.5 + 0.5 * contribution[-tail:].sum() / a[-tail-1:-1].sum())


class SpliceReuseMatcher(GuidedReuseMatcher):
    def __init__(self, index, *, method, tail_weighted=False, **kwargs):
        if type(tail_weighted) is not bool or tail_weighted and method != "ot":
            raise ValueError("Tail weighting requires OT")
        super().__init__(index, method=method, **kwargs)
        self.tail_weighted = tail_weighted
        self.lookup = ("ot-tail" if tail_weighted else method) + "-splice"
        self.stats.update(splice_proposals=0, splice_internal_proposals=0,
                          splice_prefix_accepted_tokens=0, splice_history_accepted_tokens=0,
                          splice_extension_accepted_tokens=0, splice_zero_history=0)

    def splice_shortlist(self, confirmed, draft_tokens, max_tokens):
        from pydivsufsort import sa_search

        # Multiple overlapping anchors can yield the identical candidate. Score it
        # only once, choosing its representation by the shared cheap prefilter.
        unique = {}
        for offset in range(len(draft_tokens) - self.hint_tokens + 1):
            prefix_length = offset + self.hint_tokens
            prefix = tuple(draft_tokens[:prefix_length])
            query = tuple([*confirmed, *prefix][-self.config.window:])
            hint = np.asarray(prefix[-self.hint_tokens:], dtype=">u4").tobytes()
            count, left = sa_search(self.index.data, self.index.sa, hint)
            if not count:
                continue
            for raw in self.index.sa[left:left + min(count, self.config.positions_per_anchor)]:
                if int(raw) % 4:
                    continue
                anchor = int(raw) // 4
                record = bisect_right(self.index.starts, anchor) - 1
                if record < 0:
                    continue
                lower, end = self.index.starts[record], self.index.ends[record]
                position = anchor + self.hint_tokens
                if (anchor < lower + self.config.min_context or position >= end
                        or self.blocked == (tuple(confirmed[-self.index.max_match:]), position)):
                    continue
                tokens = prefix + tuple(self.index.tokens[position:min(end, position + max_tokens - prefix_length)])
                if len(tokens) <= len(draft_tokens):
                    continue
                history = tuple(self.index.tokens[max(lower, position - len(query)):position])
                cheap = edit_score(query, history)
                item = dict(cheap=cheap, position=position, end=end, source=self.index.sources[record],
                            prefix_length=prefix_length, query=query, history=history, tokens=tokens)
                previous = unique.get(tokens)
                key = (-cheap, prefix_length, position)
                if previous is None or key < (-previous["cheap"], previous["prefix_length"], previous["position"]):
                    unique[tokens] = item
        return sorted(unique.values(), key=lambda c: (-c["cheap"], c["prefix_length"], c["position"]))[:self.config.shortlist]

    def propose_with_draft(self, confirmed, draft_tokens, max_tokens):
        if (len(confirmed) < self.config.min_context or not self.hint_tokens <= len(draft_tokens) <= self.config.window
                or max_tokens <= len(draft_tokens) or not self.index.tokens
                or self.stats["guided_lookups"] >= self.config.max_fallback_calls):
            return None
        started = time.perf_counter()
        self.stats["guided_lookups"] += 1
        try:
            candidates = self.splice_shortlist(confirmed, draft_tokens, max_tokens)
            if not candidates:
                self.stats["guided_no_match"] += 1
                return None
            self.stats["alignment_calls"] += 1
            ranked = []
            for item in candidates:
                self.stats["alignment_candidates"] += 1
                if self.method == "edit":
                    score = item["cheap"]
                else:
                    scorer = tail_transport_score if self.tail_weighted else transport_score
                    score = scorer(item["query"], item["history"], self.config, self.solver)
                ranked.append((score, -item["prefix_length"], -item["position"], item))
            score, _, _, item = max(ranked, key=lambda x: x[:3])
            if score < self.config.minimum_score:
                return None
            self.stats["alignment_proposals"] += 1
            self.stats["splice_proposals"] += 1
            self.stats["splice_internal_proposals"] += int(item["prefix_length"] > self.hint_tokens)
            return SpliceProposal(item["tokens"], item["source"], 0, item["position"], item["end"],
                                  self.lookup, True, item["prefix_length"], len(draft_tokens))
        finally:
            self.stats["alignment_seconds"] += time.perf_counter() - started

    def observe(self, proposal, accepted, confirmed):
        if not isinstance(proposal, SpliceProposal):
            return super().observe(proposal, accepted, confirmed)
        if not 0 <= accepted <= len(proposal.tokens):
            raise ValueError("Invalid accepted prefix length")
        historical = max(0, accepted - proposal.draft_prefix)
        self.stats["alignment_accepted_tokens"] += accepted
        self.stats["splice_prefix_accepted_tokens"] += min(accepted, proposal.draft_prefix)
        self.stats["splice_history_accepted_tokens"] += historical
        self.stats["splice_extension_accepted_tokens"] += max(0, accepted - proposal.original_draft)
        self.stats["splice_zero_history"] += int(historical == 0)
        # A rejection inside the retained Draft prefix has no historical position.
        if proposal.draft_prefix <= accepted < len(proposal.tokens):
            self.blocked = (tuple(confirmed[-self.index.max_match:]), proposal.position + historical)
