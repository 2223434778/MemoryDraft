"""Decode committed prefixes, including tokens that split a UTF-8 character."""
from __future__ import annotations


class PrefixText:
    def __init__(self, engine):
        self.engine = engine
        self.tokens: list[int] = []

    def append(self, tokens: list[int]) -> str:
        self.tokens.extend(tokens)
        tokenizer = getattr(self.engine, "tokenizer", None)
        if tokenizer is not None:
            text = tokenizer.decode(self.tokens, skip_special_tokens=True)
        elif self.engine.describe().get("backend") == "demo":
            text = bytes(self.tokens).decode("utf-8", errors="replace")
        else:
            raise ValueError("Text streaming requires the engine tokenizer")
        # Send replacement snapshots, not concatenated single-token decodings.
        # A byte fallback character can span multiple committed blocks.
        return text.rstrip("\ufffd")
