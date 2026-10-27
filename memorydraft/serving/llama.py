"""Thin AR-only adapter for a private, local llama.cpp server.

The DFlash engine remains a separate process with its original execution profile.
Native endpoint semantics are pinned to llama.cpp b11907 (tools/server/README.md).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx

from memorydraft.runtime.contracts import GenerationResult, RequestCancelled
from memorydraft.serving.app import create_app


class NativeTokenizer:
    def __init__(self, client):
        self.client = client
        response = client.get("/props")
        response.raise_for_status()
        props = response.json()
        self.context_length = props.get("default_generation_settings", {}).get("n_ctx")
        if type(self.context_length) is not int or self.context_length < 1:
            raise ValueError("Native server must report its actual context length")
        self.special_ids = set()
        for key in ("bos_token", "eos_token"):
            text = props.get(key)
            if text:
                response_tokens = client.post("/tokenize", json={"content": text, "add_special": False, "parse_special": True})
                response_tokens.raise_for_status()
                ids = response_tokens.json()["tokens"]
                if len(ids) != 1:
                    raise ValueError("Native special token must map to one token ID")
                self.special_ids.update(ids)

    def apply_chat_template(self, messages, *, add_generation_prompt=True, **options):
        response = self.client.post("/apply-template", json={"messages": list(messages),
            "add_generation_prompt": add_generation_prompt, "chat_template_kwargs": {"enable_thinking": False}})
        response.raise_for_status()
        prompt = response.json()["prompt"]
        response = self.client.post("/tokenize", json={"content": prompt, "add_special": False, "parse_special": True})
        response.raise_for_status()
        return response.json()["tokens"]

    def decode(self, tokens, **options):
        if options.get("skip_special_tokens"):
            tokens = [i for i in tokens if i not in self.special_ids]
        response = self.client.post("/detokenize", json={"tokens": tokens})
        response.raise_for_status()
        return response.json()["content"]


class LlamaEngine:
    methods = ("AR",)

    def __init__(self, url="http://127.0.0.1:8012", *, model="qwen3.8-27b-q8", context=None, transport=None):
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"} or parsed.username or parsed.password:
            raise ValueError("llama.cpp must bind to loopback")
        self.client = httpx.Client(base_url=url, transport=transport, trust_env=False,
                                   timeout=httpx.Timeout(connect=3, read=30, write=10, pool=5))
        try:
            self.client.get("/health").raise_for_status()
            self.tokenizer = NativeTokenizer(self.client)
            native_context = self.tokenizer.context_length
            if context is None:
                context = native_context
            if type(context) is not int or not 1 <= context <= native_context:
                raise ValueError("Adapter context exceeds the native server context")
        except Exception:
            self.client.close()
            raise
        self.model = model
        self.config = SimpleNamespace(max_input_tokens=8192, template_options={})
        self.target = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=context))

    def describe(self):
        return {"backend": "llama.cpp", "model": self.model, "synthetic": False,
                "methods": list(self.methods), "speculation_events": False, "approximate_methods": []}

    def generate(self, request, method, *, on_tokens=None, should_stop=None, profile=False):
        if method != "AR":
            raise ValueError("This model currently supports AR only")
        started = time.perf_counter()
        prompt = self.tokenizer.apply_chat_template(request.messages)
        if len(prompt) > min(self.config.max_input_tokens, self.target.config.max_position_embeddings - request.max_new_tokens):
            raise ValueError("Input exceeds the configured context limit")
        tokens, first, stop = [], None, None
        with self.client.stream("POST", "/completion", json={"prompt": prompt, "n_predict": request.max_new_tokens,
                "temperature": 0, "stream": True, "return_tokens": True, "cache_prompt": False}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if should_stop and should_stop():
                    raise RequestCancelled("Request cancelled")
                if not line.startswith("data: "):
                    continue
                value = json.loads(line[6:])
                if "error" in value:
                    raise RuntimeError("llama.cpp generation failed")
                chunk = value.get("tokens", [])
                if not isinstance(chunk, list) or any(type(i) is not int or i < 0 for i in chunk):
                    raise ValueError("Invalid native token stream")
                if chunk:
                    if first is None:
                        first = time.perf_counter() - started
                    tokens.extend(chunk)
                    if len(tokens) > request.max_new_tokens:
                        raise ValueError("Native output exceeded the requested token limit")
                    if on_tokens:
                        on_tokens(chunk)
                if value.get("stop"):
                    stop = value.get("stop_type")
                    break
        if stop is None:
            raise RuntimeError("Native stream ended without a final result")
        return GenerationResult(request.id, method, tokens, self.tokenizer.decode(tokens, skip_special_tokens=True), len(prompt),
            hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest(), "no-memory",
            time.perf_counter() - started, first, "length" if stop == "limit" else "stop",
            execution_profile="llama.cpp-Q8_0")

    def close(self):
        self.client.close()


def app():
    return create_app(lambda: LlamaEngine(os.environ.get("MEMORYDRAFT_LLAMA_URL", "http://127.0.0.1:8012")),
                      queue_capacity=2, api_key=os.environ["MEMORYDRAFT_API_KEY"])
