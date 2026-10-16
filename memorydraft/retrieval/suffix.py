"""REST-style token continuation using a mature suffix-array implementation.

Encode uint32 tokens as fixed-width big-endian bytes: pydivsufsort's search API
accepts bytes even though its array builder accepts wider integer arrays.
Only token-aligned matches are eligible. A reserved delimiter and explicit
record ends prevent continuations crossing records or special tokens.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter

import numpy as np


class SuffixArrayIndex:
    def __init__(self, snapshot, tokenizer, min_match=3, max_match=32):
        from pydivsufsort import divsufsort

        if not 1 <= min_match <= max_match:
            raise ValueError("Invalid match bounds")
        self.min_match, self.max_match = min_match, max_match
        self.starts, self.ends, self.sources = [], [], []
        data = []
        special = set(tokenizer.all_special_ids)
        for record in snapshot.records:
            # User text requires an explicit quotation policy; start with finalized assistant text.
            if record.source_role != "assistant":
                continue
            segment = []
            for token in [*tokenizer.encode(record.text, add_special_tokens=False), None]:
                if token is None or token in special:
                    if segment:
                        self.starts.append(len(data))
                        data.extend(segment)
                        self.ends.append(len(data))
                        self.sources.append(record.record_id)
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
