"""Bounded, engine-local reuse of explicitly registered public Target prefixes.

Entries own their storage. A hit materializes a new DynamicCache, so decode,
crop and cancellation can never mutate another request's state.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from transformers import DynamicCache

from memorydraft.runtime.contracts import RequestCancelled


def tokenizer_sha256(tokenizer):
    value = dict(tokenizer=tokenizer.backend_tokenizer.to_str(), chat_template=tokenizer.chat_template)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class PrefixCacheSpec:
    path: str
    sha256: str
    max_bytes: int = 2 * 2**30
    max_entries: int = 3

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("Prefix registry path is required")
        if (not isinstance(self.sha256, str) or len(self.sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("Prefix registry requires a lowercase SHA256")
        if any(type(v) is not int or v < 1 for v in (self.max_bytes, self.max_entries)):
            raise ValueError("Invalid public prefix cache limits")


@dataclass(frozen=True)
class Entry:
    kv: tuple
    features: torch.Tensor
    bytes: int


class PublicPrefixCache:
    def __init__(self, spec, *, target, tokenizer, target_revision, template_options, layers, max_input_tokens):
        raw = Path(spec.path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec.sha256:
            raise ValueError("Public prefix registry checksum mismatch")
        registry = json.loads(raw)
        if (registry.get("schema") != 1 or registry.get("visibility") != "public"
                or registry.get("target_revision") != target_revision
                or registry.get("tokenizer_sha256") != tokenizer_sha256(tokenizer)
                or registry.get("template_options") != template_options):
            raise ValueError("Public prefix registry identity mismatch")
        prefixes = registry.get("prefixes")
        if not isinstance(prefixes, list) or not 1 <= len(prefixes) <= 128:
            raise ValueError("Expected a bounded public prefix allowlist")
        self.prefixes = []
        seen = set()
        vocab_size = len(tokenizer)
        for item in prefixes:
            tokens = item.get("token_ids")
            if (not isinstance(item.get("id"), str) or not item["id"] or not item.get("source")
                    or not isinstance(tokens, list) or not 1 <= len(tokens) < max_input_tokens
                    or any(type(t) is not int or not 0 <= t < vocab_size for t in tokens)):
                raise ValueError("Invalid public prefix registration")
            key = tuple(tokens)
            if key in seen:
                raise ValueError("Duplicate public prefix")
            seen.add(key)
            self.prefixes.append((key, item["id"]))
        self.prefixes.sort(key=lambda item: len(item[0]), reverse=True)
        self.spec, self.target, self.layers = spec, target, tuple(layers)
        self.entries = OrderedDict()
        self.bytes = 0
        self.last = {}

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def describe(self):
        return dict(registry_sha256=self.spec.sha256, registered_prefixes=len(self.prefixes),
                    max_bytes=self.spec.max_bytes, max_entries=self.spec.max_entries,
                    bytes=self.bytes, entries=len(self.entries), scope="registered public inputs only",
                    storage="GPU KV and selected Target features; per-request copies")

    @torch.inference_mode()
    def prefill(self, target, input_ids, positions, layers, should_stop=None):
        if target is not self.target or tuple(layers) != self.layers:
            raise ValueError("Public prefix cache is bound to one Target and feature layout")
        def check():
            if should_stop and should_stop():
                raise RequestCancelled("Request cancelled during prefix prefill")
        check()
        tokens = tuple(input_ids[0].tolist())
        key = next((p for p, _ in self.prefixes if len(p) < len(tokens) and tokens[:len(p)] == p), None)
        length = len(key) if key else 0
        entry = self.entries.get(key)
        self.last = dict(prefix_cache_hit=int(entry is not None), prefix_cache_prefix_tokens=length,
                         prefix_cache_reused_tokens=length if entry is not None else 0,
                         prefix_cache_filled_bytes=0, prefix_cache_evictions=0)
        cache = DynamicCache()
        if entry is not None:
            self.entries.move_to_end(key)
            for i, (k, v) in enumerate(entry.kv):
                cache.update(k.clone(), v.clone(), i)
            # cat below owns new feature storage, including the cached portion.
            check()
            output = target(input_ids[:, length:], position_ids=positions[:, length:], past_key_values=cache,
                            use_cache=True, logits_to_keep=1, output_hidden_states=True)
            suffix = torch.cat([output.hidden_states[i+1] for i in layers], dim=-1)
            features = torch.cat([entry.features, suffix], dim=1)
        else:
            output = target(input_ids, position_ids=positions, past_key_values=cache,
                            use_cache=True, logits_to_keep=1, output_hidden_states=True)
            features = torch.cat([output.hidden_states[i+1] for i in layers], dim=-1)
            check()
            if key is not None:
                kv = cache.to_legacy_cache()
                if any(k.shape[-2] != len(tokens) or v.shape[-2] != len(tokens) for k,v in kv):
                    raise ValueError("Public prefix cache requires full attention KV")
                size = sum(t[..., :length, :].numel()*t.element_size() for pair in kv for t in pair)
                size += features[:, :length].numel()*features.element_size()
                if size <= self.spec.max_bytes:
                    while self.entries and (self.bytes+size > self.spec.max_bytes
                                            or len(self.entries) >= self.spec.max_entries):
                        _, old = self.entries.popitem(last=False)
                        self.bytes -= old.bytes
                        self.last["prefix_cache_evictions"] += 1
                    owned = tuple((k[..., :length, :].clone(), v[..., :length, :].clone()) for k,v in kv)
                    stored = features[:, :length].clone()
                    self.entries[key] = Entry(owned, stored, size)
                    self.bytes += size
                    self.last["prefix_cache_filled_bytes"] = size
        check()
        if cache.get_seq_length() != len(tokens) or features.shape[1] != len(tokens):
            raise RuntimeError("Public prefix prefill frontier mismatch")
        self.last.update(prefix_cache_bytes=self.bytes, prefix_cache_entries=len(self.entries))
        return output.logits, features, cache
