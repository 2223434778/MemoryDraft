"""Deterministic transport demo. It performs no model inference or GPU work."""
from __future__ import annotations

import hashlib
import json
import time

from memorydraft.runtime.contracts import GenerationResult, RequestCancelled


class DemoEngine:
    methods = ("AR",)

    def describe(self):
        return {"backend": "demo", "synthetic": True, "methods": list(self.methods),
                "note": "Transport demo only; no model inference or performance claim"}

    def generate(self, request, method, *, on_tokens=None, should_stop=None, profile=False):
        if method not in self.methods:
            raise ValueError("Demo only supports AR")
        started = time.perf_counter()
        complete = list(b"MemoryDraft transport demo.")
        tokens = complete[:request.max_new_tokens]
        first = None
        for token in tokens:
            if should_stop and should_stop():
                raise RequestCancelled("Request cancelled")
            if first is None:
                first = time.perf_counter() - started
            if on_tokens:
                on_tokens([token])
        return GenerationResult(request.id, method, tokens, bytes(tokens).decode(), 0,
                                hashlib.sha256(json.dumps(request.messages, sort_keys=True).encode()).hexdigest(),
                                "demo:no-memory", time.perf_counter() - started, first,
                                "length" if len(tokens) < len(complete) else "stop")

    def close(self):
        pass
