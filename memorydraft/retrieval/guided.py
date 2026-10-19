"""Draft-guided approximate reuse. Hints select proposals, never accept tokens."""
from __future__ import annotations

from bisect import bisect_right
import time
from typing import Sequence

import numpy as np

from memorydraft.retrieval.reuse import ReuseMatcher, ReuseProposal
from memorydraft.retrieval.transport import AlignmentConfig, edit_score, transport_score


class GuidedReuseMatcher(ReuseMatcher):
    def __init__(self, index, *, method: str, exact_min_match: int = 6,
                 config: AlignmentConfig = AlignmentConfig(), hint_tokens: int = 4):
        if method not in {"ot", "edit"} or exact_min_match < index.min_match:
            raise ValueError("Invalid guided retrieval method")
        if type(hint_tokens) is not int or not 1 <= hint_tokens <= config.window:
            raise ValueError("Invalid draft hint length")
        super().__init__(index, cursor=False)
        self.method, self.config = method, config
        self.exact_min_match, self.hint_tokens = exact_min_match, hint_tokens
        self.solver = None
        if method == "ot":
            from ot import sinkhorn
            self.solver = sinkhorn
        self.stats = dict(alignment_calls=0, alignment_candidates=0, alignment_seconds=0.0,
                          alignment_proposals=0, alignment_accepted_tokens=0,
                          guided_lookups=0, guided_no_match=0)

    def propose(self, confirmed: Sequence[int], max_tokens: int):
        exact = super().propose(confirmed, max_tokens)
        return exact if exact is not None and exact.matched >= self.exact_min_match else None

    def shortlist(self, confirmed: Sequence[int], hint: Sequence[int], max_tokens: int,
                  minimum_tokens: int):
        from pydivsufsort import sa_search

        count, left = sa_search(self.index.data, self.index.sa, np.asarray(hint, dtype=">u4").tobytes())
        if not count:
            return []
        query = tuple([*confirmed, *hint][-self.config.window:])
        candidates = []
        for offset in self.index.sa[left:left + min(count, self.config.positions_per_anchor)]:
            if int(offset) % 4:
                continue
            position = int(offset) // 4
            record = bisect_right(self.index.starts, position) - 1
            if record < 0:
                continue
            lower, end = self.index.starts[record], self.index.ends[record]
            if (position < lower + self.config.min_context or position + len(hint) > end
                    or min(max_tokens, end - position) < minimum_tokens
                    or self.blocked == (tuple(confirmed[-self.index.max_match:]), position)):
                continue
            history = tuple(self.index.tokens[max(lower, position + len(hint) - len(query)):position + len(hint)])
            candidates.append((edit_score(query, history), position, end, self.index.sources[record], history))
        return sorted(candidates, key=lambda item: (-item[0], item[1]))[:self.config.shortlist]

    def propose_with_draft(self, confirmed: Sequence[int], draft_tokens: Sequence[int], max_tokens: int):
        if (len(confirmed) < self.config.min_context or len(draft_tokens) < self.hint_tokens
                or max_tokens <= len(draft_tokens) or not self.index.tokens
                or self.stats["guided_lookups"] >= self.config.max_fallback_calls):
            return None
        started = time.perf_counter()
        self.stats["guided_lookups"] += 1
        try:
            hint = tuple(draft_tokens[:self.hint_tokens])
            candidates = self.shortlist(confirmed, hint, max_tokens, len(draft_tokens) + 1)
            if not candidates:
                self.stats["guided_no_match"] += 1
                return None
            self.stats["alignment_calls"] += 1
            query = tuple([*confirmed, *hint][-self.config.window:])
            ranked = []
            for cheap, position, end, source, history in candidates:
                self.stats["alignment_candidates"] += 1
                score = (transport_score(query, history, self.config, self.solver)
                         if self.method == "ot" else cheap)
                ranked.append((score, -position, position, end, source))
            score, _, position, end, source = max(ranked)
            if score < self.config.minimum_score:
                return None
            matched = 0
            record = bisect_right(self.index.starts, position) - 1
            lower = self.index.starts[record]
            before = self.index.tokens[max(lower, position - self.config.window):position]
            for current, old in zip(reversed(confirmed[-self.config.window:]), reversed(before)):
                if current != old:
                    break
                matched += 1
            self.stats["alignment_proposals"] += 1
            return ReuseProposal(tuple(self.index.tokens[position:min(position + max_tokens, end)]),
                                 source, matched, position, end, self.method + "-guided", context_qualified=True)
        finally:
            self.stats["alignment_seconds"] += time.perf_counter() - started

    def observe(self, proposal, accepted, confirmed):
        super().observe(proposal, accepted, confirmed)
        if proposal.lookup == self.method + "-guided":
            self.stats["alignment_accepted_tokens"] += accepted
