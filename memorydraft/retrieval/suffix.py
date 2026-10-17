"""REST-style token continuation using a mature suffix-array implementation.

Encode uint32 tokens as fixed-width big-endian bytes: pydivsufsort's search API
accepts bytes even though its array builder accepts wider integer arrays.
Only token-aligned matches are eligible. A reserved delimiter and explicit
record ends prevent continuations crossing records or special tokens.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PositionedMatch:
    tokens: tuple[int, ...]
    source: str
    matched: int
    position: int
    end: int


class SuffixArrayIndex:
    def __init__(self, snapshot, tokenizer, min_match=3, max_match=32, *, prompt_text=None):
        from pydivsufsort import divsufsort

        if not 1 <= min_match <= max_match:
            raise ValueError("Invalid match bounds")
        self.min_match, self.max_match = min_match, max_match
        self.starts, self.ends, self.sources = [], [], []
        data = []
        special = set(tokenizer.all_special_ids)
        texts = []
        for record in snapshot.records:
            # User text requires an explicit quotation policy; start with finalized assistant text.
            if record.source_role != "assistant":
                continue
            texts.append((record.record_id, record.text))
        if prompt_text is not None:
            # Current input is an explicit candidate resource, never a past memory event.
            texts.append(("context:current-prompt", prompt_text))
        for source, text in texts:
            segment = []
            for token in [*tokenizer.encode(text, add_special_tokens=False), None]:
                if token is None or token in special:
                    if segment:
                        self.starts.append(len(data))
                        data.extend(segment)
                        self.ends.append(len(data))
                        self.sources.append(source)
                        data.append(2**32 - 1)
                    segment = []
                else:
                    if not 0 <= token < 2**32 - 1:
                        raise ValueError("Token cannot be represented without collision")
                    segment.append(token)
        self.tokens = data
        self.data = np.frombuffer(np.asarray(data, dtype=">u4").tobytes(), dtype=np.uint8).copy()
        self.sa = divsufsort(self.data) if data else np.empty(0, dtype=np.int32)
        self.bytes_used = self.data.nbytes + self.sa.nbytes

    def match(self, prefix, max_tokens):
        from pydivsufsort import sa_search

        if not self.tokens or max_tokens <= 0:
            return [], None, 0
        for n in range(min(len(prefix), self.max_match), self.min_match - 1, -1):
            query = np.asarray(prefix[-n:], dtype=">u4").tobytes()
            count, left = sa_search(self.data, self.sa, query)
            if not count:
                continue
            candidates, sources = Counter(), {}
            for offset in self.sa[left:left + count]:
                if int(offset) % 4:
                    continue
                start = int(offset) // 4
                record = bisect_right(self.starts, start) - 1
                end = start + n
                if record < 0 or end >= self.ends[record]:
                    continue
                continuation = tuple(self.tokens[end:min(end + max_tokens, self.ends[record])])
                candidates[continuation] += 1
                sources.setdefault(continuation, self.sources[record])
            if candidates:
                best = min(candidates, key=lambda x: (-candidates[x], x))
                return list(best), sources[best], n
        return [], None, 0

    def locate(self, prefix, max_tokens, max_positions=64):
        """Bounded positional lookup for the reuse cursor; preserve legacy match()."""
        from pydivsufsort import sa_search

        if max_positions < 1:
            raise ValueError("max_positions must be positive")
        if not self.tokens or max_tokens <= 0:
            return None
        for n in range(min(len(prefix), self.max_match), self.min_match - 1, -1):
            count, left = sa_search(self.data, self.sa, np.asarray(prefix[-n:], dtype=">u4").tobytes())
            if not count:
                continue
            found = []
            for offset in self.sa[left:left + min(count, max_positions)]:
                if int(offset) % 4:
                    continue
                start = int(offset) // 4
                record = bisect_right(self.starts, start) - 1
                position = start + n
                if record >= 0 and position < self.ends[record]:
                    found.append(PositionedMatch(tuple(self.tokens[position:min(position + max_tokens, self.ends[record])]),
                                                 self.sources[record], n, position, self.ends[record]))
            if found:
                counts = Counter(item.tokens for item in found)
                return min(found, key=lambda item: (-counts[item.tokens], item.tokens, item.position))
        return None
