"""Single-owner, fixed-target runtime over the existing audited decode loop."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
from importlib import resources
import json
from pathlib import Path
import threading
import time

import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from memorydraft.backends.dflash_reuse import ReuseConfig, generate_reuse
from memorydraft.memory.store import MemoryStore, Snapshot
from memorydraft.retrieval.reuse import ReuseMatcher
from memorydraft.retrieval.suffix import SuffixArrayIndex
from memorydraft.runtime.config import RuntimeConfig
from memorydraft.runtime.contracts import GenerationResult, RequestCancelled


def verify_dflash_assets(directory: Path):
    manifest = json.loads(resources.files("memorydraft").joinpath("assets/dflash-manifest.json").read_text())
    for item in [*manifest["code"], manifest["weight"]]:
        with (directory / item["filename"]).open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError("Audited DFlash asset checksum mismatch: " + item["filename"])
    return manifest


class InferenceEngine:
    def __init__(self, target, tokenizer, config: RuntimeConfig, draft=None):
        self.target, self.tokenizer, self.draft, self.config = target.eval(), tokenizer, draft, config
        if draft is not None:
            draft.eval()
        self.methods = ("AR", "DFlash", "R+DFlash") if draft is not None else ("AR",)
        self.lock = threading.Lock()
        self.reuse = ReuseConfig(**config.reuse)
        if config.retrieval == "ot":
            # Import optional solver libraries during load, before request timing.
            from ot import sinkhorn
        eos = target.generation_config.eos_token_id
        self.stops = [eos] if isinstance(eos, int) else list(eos or [])

    @classmethod
    def load(cls, config: RuntimeConfig):
        expected = "4.57.3" if config.backend == "dflash" else "4.52.2"
        if transformers.__version__ != expected:
            raise RuntimeError(f"Use the isolated {config.backend} profile: transformers=={expected}")
        if not Path(config.target).is_dir():
            raise ValueError("Target must be a prepared local directory; runtime never downloads models")
        if config.backend == "dflash":
            verify_dflash_assets(Path(config.draft))
        torch.set_num_threads(config.threads)
        torch.set_float32_matmul_precision("highest")
        tokenizer = AutoTokenizer.from_pretrained(config.target, local_files_only=True, trust_remote_code=False)
        target = AutoModelForCausalLM.from_pretrained(config.target, local_files_only=True, trust_remote_code=False,
                                                    torch_dtype=torch.float32, attn_implementation="eager")
        target = target.to(config.device).eval()
        draft = None
        if config.backend == "dflash":
            draft = AutoModel.from_pretrained(config.draft, local_files_only=True, trust_remote_code=True,
                                              torch_dtype=torch.float32, attn_implementation="eager")
            draft = draft.to(config.device).eval()
        return cls(target, tokenizer, config, draft)

    def describe(self):
        return {"backend": self.config.backend, "synthetic": False, "methods": list(self.methods),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "device": str(self.target.device), "dtype": str(self.target.dtype),
                "gpu": torch.cuda.get_device_name(self.target.device) if self.target.device.type == "cuda" else None,
                "target_revision": self.config.target_revision,
                "implementation": "DFlash uses the audited adapter with R disabled; native control remains in research scripts",
                "config": self.config.as_dict()}

    def synchronize(self):
        if self.target.device.type == "cuda":
            torch.cuda.synchronize(self.target.device)

    @torch.inference_mode()
    def generate(self, request, method, *, on_tokens=None, should_stop=None, profile=False):
        if method not in self.methods:
            raise ValueError("Unsupported method: " + method)
        if request.max_new_tokens > self.config.max_new_tokens:
            raise ValueError("Request exceeds configured output token limit")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("InferenceEngine requires a single owner; use the serving worker")
        try:
            self.synchronize()
            if self.target.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.target.device)
            started = time.perf_counter()
            first = None

            def check():
                if should_stop and should_stop():
                    raise RequestCancelled("Request cancelled")

            def commit(tokens):
                nonlocal first
                check()
                if tokens:
                    if first is None:
                        first = time.perf_counter() - started
                    if on_tokens:
                        on_tokens(tokens)

            def mark(name):
                return torch.profiler.record_function("memorydraft::" + name) if profile else nullcontext()

            check()
            with mark("snapshot"):
                if self.config.db:
                    with MemoryStore(self.config.db) as store:
                        snapshot = store.snapshot(request.user_id, request.order)
                else:
                    snapshot = Snapshot(request.user_id, request.order, ())
            with mark("tokenize"):
                ids = self.tokenizer.apply_chat_template(list(request.messages), add_generation_prompt=True,
                                                        return_tensors="pt", **self.config.template_options)
                ids = ids.to(self.target.device)
                prompt_tokens = ids.shape[1]
                context_limit = getattr(self.target.config, "max_position_embeddings", self.config.max_input_tokens)
                if prompt_tokens > self.config.max_input_tokens or prompt_tokens + request.max_new_tokens > context_limit:
                    raise ValueError("Request exceeds configured/model context limit")
                prompt_sha = hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest()
            matcher, index = None, None
            if method == "R+DFlash":
                with mark("index_build"):
                    index = SuffixArrayIndex(snapshot, self.tokenizer, min_match=3, max_match=32,
                                             prompt_text="\n".join(m["content"] for m in request.messages))
                    if self.config.retrieval == "suffix":
                        matcher = ReuseMatcher(index, cursor=False)
                    else:
                        from memorydraft.retrieval.transport import AlignedReuseMatcher, AlignmentConfig
                        matcher = AlignedReuseMatcher(index, method=self.config.retrieval,
                                                      exact_min_match=self.reuse.min_match,
                                                      config=AlignmentConfig(**self.config.alignment))
            check()
            counters = {}
            if method == "AR":
                class Streamer:
                    prompt_seen = False

                    def put(self, value):
                        if not self.prompt_seen:
                            self.prompt_seen = True
                        else:
                            commit(value.reshape(-1).tolist())

                    def end(self):
                        pass

                class CancelCriteria(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        return bool(should_stop and should_stop())

                with mark("ar_generate"):
                    output = self.target.generate(
                        ids, attention_mask=torch.ones_like(ids), do_sample=False, use_model_defaults=False,
                        max_new_tokens=request.max_new_tokens, use_cache=True, streamer=Streamer(),
                        stopping_criteria=StoppingCriteriaList([CancelCriteria()]),
                        pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.stops or None,
                    )
            else:
                detail = generate_reuse(self.target, self.draft, ids, max_new_tokens=request.max_new_tokens,
                                        stop_token_ids=self.stops, matcher=matcher, config=self.reuse,
                                        on_tokens=commit, should_stop=should_stop, profile=profile)
                output = detail.sequences
                counters = {"rounds": len(detail.rounds), "reuse_rounds": sum(r["route"] == "R" for r in detail.rounds),
                            "reuse_accepted_tokens": sum(r["accepted"] for r in detail.rounds if r["route"] == "R"),
                            "draft_rounds": sum(r["route"] == "D" for r in detail.rounds),
                            "index_bytes": index.bytes_used if index else 0,
                            "feature_peak_tokens": detail.feature_peak_tokens, "buffer_flushes": detail.buffer_flushes}
                counters.update(getattr(matcher, "stats", {}))
            self.synchronize()
            check()
            tokens = output[0, prompt_tokens:].tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            latency = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(self.target.device) if self.target.device.type == "cuda" else 0
            return GenerationResult(request.id, method, tokens, text, prompt_tokens, prompt_sha, snapshot.snapshot_id,
                                    latency, first, "eos" if tokens and tokens[-1] in self.stops else "length", peak, counters)
        finally:
            try:
                self.synchronize()
            finally:
                self.lock.release()

    def close(self):
        self.synchronize()
