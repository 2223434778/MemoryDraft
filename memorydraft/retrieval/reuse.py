"""Request-local continuation cursor over an immutable, user-scoped index."""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from typing import Sequence

from memorydraft.retrieval.suffix import PositionedMatch, SuffixArrayIndex


@dataclass(frozen=True)
class ReuseProposal:
    tokens: tuple[int, ...]
    source: str
    matched: int
    position: int
    end: int
    lookup: str
    # Qualification of the retrieval context never changes Target verification.
    context_qualified: bool = False


class ReuseMatcher:
    def __init__(self, index: SuffixArrayIndex, *, cursor: bool = True, max_positions: int = 64,
                 local_window: int = 96):
        if max_positions < 1 or local_window < 0:
            raise ValueError("Invalid lookup budget")
        self.index, self.cursor_enabled = index, cursor
        self.max_positions, self.local_window = max_positions, local_window
        self.last: PositionedMatch | None = None
        self.next_position: int | None = None
        self.blocked: tuple[tuple[int, ...], int] | None = None
        self.lookups = dict(cursor=0, local=0, global_lookup=0)

    def _at(self, prefix: tuple[int, ...], position: int, budget: int, mode: str):
        if self.last is None:
            return None
        start = self.last.end
        # Locate the segment start; never match across record or special-token boundaries.
        record = bisect_right(self.index.starts, self.last.position) - 1
        lower = self.index.starts[record]
        if not lower <= position < start:
            return None
        for n in range(min(len(prefix), self.index.max_match, position - lower), self.index.min_match - 1, -1):
            if tuple(self.index.tokens[position - n:position]) == prefix[-n:]:
                if self.blocked == (prefix[-self.index.max_match:], position):
                    return None
                return ReuseProposal(tuple(self.index.tokens[position:min(position + budget, start)]),
                                     self.last.source, n, position, start, mode)
        return None

    def propose(self, confirmed: Sequence[int], max_tokens: int) -> ReuseProposal | None:
        if max_tokens <= 0 or not self.index.tokens:
            return None
        prefix = tuple(confirmed)
        if self.cursor_enabled and self.last is not None and self.next_position is not None:
            result = self._at(prefix, self.next_position, max_tokens, "cursor")
            if result is None:
                low = max(0, self.next_position - self.local_window)
                high = min(self.last.end, self.next_position + self.local_window + 1)
                for position in range(low, high):
                    result = self._at(prefix, position, max_tokens, "local")
                    if result is not None:
                        break
            if result is not None:
                self.lookups[result.lookup] += 1
                return result
        self.lookups["global_lookup"] += 1
        item = self.index.locate(prefix, max_tokens, self.max_positions)
        if item is None or self.blocked == (prefix[-self.index.max_match:], item.position):
            return None
        return ReuseProposal(item.tokens, item.source, item.matched, item.position, item.end, "global")

    def observe(self, proposal: ReuseProposal, accepted: int, confirmed: Sequence[int]):
        if not 0 <= accepted <= len(proposal.tokens):
            raise ValueError("Invalid accepted prefix length")
        self.last = PositionedMatch(proposal.tokens, proposal.source, proposal.matched,
                                    proposal.position, proposal.end)
        self.next_position = proposal.position + accepted + 1
        if accepted < len(proposal.tokens):
            # Block only the failed continuation under its actual preceding context.
            context = tuple(confirmed)[-self.index.max_match:]
            self.blocked = (context, proposal.position + accepted)
