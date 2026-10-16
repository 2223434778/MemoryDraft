from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from memorydraft.memory.store import Snapshot


@dataclass(frozen=True)
class Chunk:
    source_id: str
    text: str
    token_ids: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class MemoryBundle:
    snapshot_id: str
    chunks: tuple[Chunk, ...]
    retriever: str

    @property
    def bundle_id(self):
        return hashlib.sha256(repr((self.snapshot_id, self.chunks, self.retriever)).encode()).hexdigest()


class LexicalEncoder:
    """Diagnostic cosine baseline, explicitly not a semantic embedding model."""

    name = "lexical-cosine-diagnostic-v1"

    def scores(self, query: str, texts: list[str]) -> list[float]:
        def encode(text):
            return Counter(re.findall(r"[a-zA-Z_0-9]+|[\u4e00-\u9fff]", text.lower()))
        q = encode(query)
        qn = math.sqrt(sum(v * v for v in q.values()))
        result = []
        for text in texts:
            t = encode(text)
            norm = qn * math.sqrt(sum(v * v for v in t.values()))
            result.append(sum(v * t[k] for k, v in q.items()) / norm if norm else 0.0)
        return result


class SemanticEncoder:
    def __init__(self, model: str, revision: str):
        if not revision or revision == "main":
            raise ValueError("Pin the embedding model to a commit revision")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, revision=revision, device="cpu", trust_remote_code=False)
        self.name = f"{model}@{revision}"

    def scores(self, query: str, texts: list[str]) -> list[float]:
        # No cross-snapshot derived cache: future summaries/embeddings cannot leak.
        vectors = self.model.encode([query, *texts], normalize_embeddings=True)
        return (vectors[1:] @ vectors[0]).tolist()


def retrieve(query: str, snapshot: Snapshot, tokenizer, encoder, *, top_k: int = 8,
             token_budget: int = 2048, chunk_tokens: int = 256) -> MemoryBundle:
    if min(top_k, token_budget, chunk_tokens) <= 0:
        raise ValueError("Retrieval budgets must be positive")
    candidates = []
    special = set(tokenizer.all_special_ids)
    for record in snapshot.records:
        ids = tokenizer.encode(record.text, add_special_tokens=False)
        # Never let R cross a special token or an original record boundary.
        segment = []
        segments = []
        for token in ids:
            if token in special:
                if segment:
                    segments.append(segment)
                segment = []
            else:
                segment.append(token)
        if segment:
            segments.append(segment)
        for segment in segments:
            for offset in range(0, len(segment), chunk_tokens):
                piece = tuple(segment[offset:offset + chunk_tokens])
                candidates.append((record.record_id, tokenizer.decode(piece), piece))
    if not candidates:
        return MemoryBundle(snapshot.snapshot_id, (), encoder.name)
    scores = encoder.scores(query, [c[1] for c in candidates])
    chunks, seen, used = [], set(), 0
    for i in sorted(range(len(candidates)), key=lambda i: (-scores[i], i)):
        source, text, ids = candidates[i]
        if scores[i] <= 0 or ids in seen:
            continue
        ids = ids[:token_budget - used]
        if not ids:
            break
        chunks.append(Chunk(source, tokenizer.decode(ids), ids, float(scores[i])))
        seen.add(ids)
        used += len(ids)
        if len(chunks) >= top_k or used >= token_budget:
            break
    return MemoryBundle(snapshot.snapshot_id, tuple(chunks), encoder.name)


class ContinuationIndex:
    def __init__(self, bundle: MemoryBundle, min_match: int = 3, max_match: int = 32):
        if not 1 <= min_match <= max_match:
            raise ValueError("Invalid match bounds")
        self.min_match, self.max_match = min_match, max_match
        self.entries = defaultdict(list)
        for chunk in bundle.chunks:
            for end in range(min_match, len(chunk.token_ids)):
                key = chunk.token_ids[end - min_match:end]
                self.entries[key].append((chunk, end))

    def match(self, prefix: list[int], max_tokens: int) -> tuple[list[int], str | None, int]:
        if len(prefix) < self.min_match or max_tokens <= 0:
            return [], None, 0
        best, source, length = [], None, 0
        for chunk, end in self.entries.get(tuple(prefix[-self.min_match:]), ()):
            matched = self.min_match
            while matched < min(len(prefix), end, self.max_match) and prefix[-matched - 1] == chunk.token_ids[end - matched - 1]:
                matched += 1
            if matched > length:
                best = list(chunk.token_ids[end:end + max_tokens])
                source, length = chunk.source_id, matched
        return best, source, length
