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
from memorydraft.runtime.draft_adapter import DraftAdapterSpec, merge_draft_adapter


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
        from memorydraft.backends.asd import ASDConfig
        self._asd_config = ASDConfig(**config.asd) if config.asd else None
        if self._asd_config is not None:
            if draft is None:
                raise ValueError("ASD requires a loaded DFlash draft")
            self.methods += ("ASD",)
        self._adapter_spec = DraftAdapterSpec(**config.draft_adapter) if config.draft_adapter else None
        self._adapter_info = None
        if self._adapter_spec is not None:
            self._adapter_info = merge_draft_adapter(draft, self._adapter_spec)
        elif getattr(draft, "_memorydraft_adapter_sha256", None) is not None:
            raise ValueError("An adapted draft requires its matching draft_adapter config")
        from memorydraft.runtime.cost_tree import CostTreeSpec
        self._tree_spec = CostTreeSpec(**config.cost_tree) if config.cost_tree else None
        self._tree_config = None
        if self._tree_spec is not None:
            if draft is None:
                raise ValueError("Cost tree requires a loaded DFlash draft")
            self._tree_config = self._tree_spec.load(self)
            self.methods += ("BudgetTree",)
            if self._asd_config is not None:
                self.methods += ("BudgetTree-ASD",)
        self.lock = threading.Lock()
        self.reuse = ReuseConfig(**config.reuse)
        if config.retrieval.startswith("ot"):
            # Import optional solver libraries during load, before request timing.
            from ot import sinkhorn
        eos = target.generation_config.eos_token_id
        self.stops = [eos] if isinstance(eos, int) else list(eos or [])
        self.execution_profile = config.execution_profile
        self._closed = False
        self._regional = self._regional_context = None
        self._regional_setup_seconds = 0.
        self._prefix_spec = None
        self._prefix_cache = None
        if config.prefix_cache:
            from memorydraft.runtime.prefix_cache import PrefixCacheSpec, PublicPrefixCache
            self._prefix_spec = PrefixCacheSpec(**config.prefix_cache)
            self._prefix_cache = PublicPrefixCache(self._prefix_spec, target=target, tokenizer=tokenizer,
                target_revision=config.target_revision, template_options=config.template_options,
                layers=draft.target_layer_ids, max_input_tokens=config.max_input_tokens)
        if self.execution_profile != "eager":
            from memorydraft.backends.regional_compile import runtime_rmsnorm
            started = time.perf_counter()
            self._regional = runtime_rmsnorm(target, draft)
            self._regional_context = self._regional.activate(self.execution_profile)
            self._regional_context.__enter__()
            self._regional_setup_seconds = time.perf_counter()-started

    @classmethod
    def load(cls, config: RuntimeConfig):
        expected = "4.57.3" if config.backend == "dflash" else "4.52.2"
        if transformers.__version__ != expected:
            raise RuntimeError(f"Use the isolated {config.backend} profile: transformers=={expected}")
        if not Path(config.target).is_dir():
            raise ValueError("Target must be a prepared local directory; runtime never downloads models")
        if config.backend == "dflash":
            verify_dflash_assets(Path(config.draft))
        if config.draft_adapter:
            DraftAdapterSpec(**config.draft_adapter).verify()
        if config.cost_tree:
            from memorydraft.runtime.cost_tree import CostTreeSpec
            CostTreeSpec(**config.cost_tree).read()
        torch.set_num_threads(config.threads)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
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
                "execution_profile": self.execution_profile,
                "approximate_methods": [m for m in self.methods if m in {"ASD", "BudgetTree-ASD"}],
                "speculation_events": self.draft is not None,
                "asd": None if self._asd_config is None else {"budget": self._asd_config.budget,
                    "gate": self._asd_config.gate, "cap": self._asd_config.cap, "selection": "longest", "value": "remaining"},
                "draft_adapter": self._adapter_info,
                "cost_tree": self._tree_spec.describe() if self._tree_spec is not None else None,
                "prefix_cache": self._prefix_cache.describe() if self._prefix_cache is not None else None,
                "regional_compilation": None if self._regional is None else {
                    "backend": self._regional.backend.name, "graphs": len(self._regional.backend.events),
                    "max_graphs": self._regional.backend.max_graphs,
                    "max_backend_seconds": self._regional.backend.max_seconds,
                    "setup_seconds": self._regional_setup_seconds,
                    "binding_lifetime": "engine; lazy compilation is charged inside generate"},
                "implementation": "DFlash uses the audited adapter with R disabled; native control remains in research scripts",
                "config": self.config.as_dict()}

    def synchronize(self):
        if self.target.device.type == "cuda":
            torch.cuda.synchronize(self.target.device)

    @torch.inference_mode()
    def generate(self, request, method, *, on_tokens=None, should_stop=None, profile=False, on_event=None):
        if method not in self.methods:
            raise ValueError("Unsupported method: " + method)
        if request.max_new_tokens > self.config.max_new_tokens:
            raise ValueError("Request exceeds configured output token limit")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("InferenceEngine requires a single owner; use the serving worker")
        try:
            if self._closed:
                raise RuntimeError("InferenceEngine is closed")
            if self.config.execution_profile != self.execution_profile:
                raise ValueError("Reload the engine to change execution_profile")
            from memorydraft.backends.asd import ASDConfig, ASDVerifier
            current_asd = ASDConfig(**self.config.asd) if self.config.asd else None
            current_adapter = DraftAdapterSpec(**self.config.draft_adapter) if self.config.draft_adapter else None
            if current_asd != self._asd_config or current_adapter != self._adapter_spec:
                raise ValueError("Reload the engine to change ASD or draft_adapter configuration")
            from memorydraft.runtime.cost_tree import CostTreeSpec
            current_tree = CostTreeSpec(**self.config.cost_tree) if self.config.cost_tree else None
            if current_tree != self._tree_spec:
                raise ValueError("Reload the engine to change cost_tree configuration")
            from memorydraft.runtime.prefix_cache import PrefixCacheSpec
            current_prefix = PrefixCacheSpec(**self.config.prefix_cache) if self.config.prefix_cache else None
            if current_prefix != self._prefix_spec:
                raise ValueError("Reload the engine to change public prefix cache configuration")
            self.synchronize()
            if self.target.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.target.device)
            started = time.perf_counter()
            compile_before = len(self._regional.backend.events) if self._regional is not None else 0
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
                    elif self.config.retrieval == "edit-risk-splice":
                        from memorydraft.retrieval.prefix_risk import PrefixRiskPolicy, PrefixRiskReuseMatcher
                        from memorydraft.retrieval.transport import AlignmentConfig
                        matcher = PrefixRiskReuseMatcher(index, policy=PrefixRiskPolicy(**self.config.prefix_risk),
                                                         exact_min_match=self.reuse.min_match,
                                                         config=AlignmentConfig(**self.config.alignment))
                    elif self.config.retrieval in {"ot-budget-splice", "ot-gate-splice", "edit-budget-splice"}:
                        from memorydraft.retrieval.benefit import BenefitPolicy, BenefitReuseMatcher
                        from memorydraft.retrieval.transport import AlignmentConfig
                        matcher = BenefitReuseMatcher(index, policy=BenefitPolicy(**self.config.benefit),
                                                      prompt_tokens=ids.shape[1],
                                                      allow_ot=self.config.retrieval != "edit-budget-splice",
                                                      adaptive=self.config.retrieval != "ot-gate-splice",
                                                      exact_min_match=self.reuse.min_match,
                                                      config=AlignmentConfig(**self.config.alignment))
                    elif self.config.retrieval in {"ot-order-splice", "ot-plan-splice", "ot-all-splice"}:
                        from memorydraft.retrieval.matrix import MatrixReuseMatcher
                        from memorydraft.retrieval.transport import AlignmentConfig
                        matcher = MatrixReuseMatcher(index, variant=self.config.retrieval.split("-")[1],
                                                     exact_min_match=self.reuse.min_match,
                                                     config=AlignmentConfig(**self.config.alignment))
                    elif self.config.retrieval.endswith("-splice"):
                        from memorydraft.retrieval.splice import SpliceReuseMatcher
                        from memorydraft.retrieval.transport import AlignmentConfig
                        matcher = SpliceReuseMatcher(index, method=self.config.retrieval.split("-")[0],
                                                     tail_weighted=self.config.retrieval == "ot-tail-splice",
                                                     exact_min_match=self.reuse.min_match,
                                                     config=AlignmentConfig(**self.config.alignment))
                    elif self.config.retrieval.endswith("-guided"):
                        from memorydraft.retrieval.guided import GuidedReuseMatcher
                        from memorydraft.retrieval.transport import AlignmentConfig
                        matcher = GuidedReuseMatcher(index, method=self.config.retrieval.split("-")[0],
                                                     exact_min_match=self.reuse.min_match,
                                                     config=AlignmentConfig(**self.config.alignment))
                    else:
                        from memorydraft.retrieval.transport import AlignedReuseMatcher, AlignmentConfig
                        matcher = AlignedReuseMatcher(index, method=self.config.retrieval,
                                                      exact_min_match=self.reuse.min_match,
                                                      config=AlignmentConfig(**self.config.alignment))
            check()
            counters = {}
            verifier = ASDVerifier(self._asd_config) if method == "ASD" else None
            tree_info = None
            if method == "BudgetTree-ASD":
                from memorydraft.backends.budget_tree import TreeASDVerifier
                verifier = TreeASDVerifier(self._asd_config)
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
                verification = {"verifier": verifier, "incremental_stop": True} if verifier is not None else {}
                if method in {"BudgetTree", "BudgetTree-ASD"}:
                    verification.update(tree_config=self._tree_config, incremental_stop=True)
                if self._prefix_cache is not None:
                    verification["prefill"] = self._prefix_cache.prefill
                if on_event is not None:
                    verification["on_event"] = on_event
                detail = generate_reuse(self.target, self.draft, ids, max_new_tokens=request.max_new_tokens,
                                        stop_token_ids=self.stops, matcher=matcher, config=self.reuse,
                                        on_tokens=commit, should_stop=should_stop, profile=profile, **verification)
                output = detail.sequences
                if method in {"BudgetTree", "BudgetTree-ASD"}:
                    tree_info = self._tree_spec.describe()
                    counters.update(tree_rounds=len(detail.rounds),
                        tree_verification_nodes=sum(r["tree_nodes"] for r in detail.rounds),
                        tree_built_nodes=sum(r["tree_budget"]["built_nodes"] for r in detail.rounds))
                counters.update({"rounds": len(detail.rounds), "reuse_rounds": sum(r["route"] == "R" for r in detail.rounds),
                            "reuse_accepted_tokens": sum(r["accepted"] for r in detail.rounds if r["route"] == "R"),
                            "draft_rounds": sum(r["route"] == "D" for r in detail.rounds),
                            "index_bytes": index.bytes_used if index else 0,
                            "feature_peak_tokens": detail.feature_peak_tokens, "buffer_flushes": detail.buffer_flushes})
                counters.update(getattr(matcher, "stats", {}))
                if self._prefix_cache is not None:
                    counters.update(self._prefix_cache.last)
                if self.config.retrieval.endswith(("-guided", "-splice")):
                    counters["draft_forward_rounds"] = sum(r["draft_forward"] for r in detail.rounds)
            self.synchronize()
            check()
            tokens = output[0, prompt_tokens:].tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            if self._regional is not None:
                events = self._regional.backend.events
                counters.update(compile_graphs_before=compile_before, compile_graphs_after=len(events),
                                compile_backend_seconds=sum(e.get("backend_seconds", 0.) for e in events[compile_before:]))
            latency = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(self.target.device) if self.target.device.type == "cuda" else 0
            return GenerationResult(request.id, method, tokens, text, prompt_tokens, prompt_sha, snapshot.snapshot_id,
                                    latency, first, "eos" if tokens and tokens[-1] in self.stops else "length", peak, counters,
                                    execution_profile=self.execution_profile, approximate=verifier is not None,
                                    asd=None if verifier is None else {**verifier.summary(), "gate": self._asd_config.gate, "cap": self._asd_config.cap},
                                    draft_adapter_sha256=self._adapter_spec.sha256 if self._adapter_spec is not None and method != "AR" else None,
                                    tree=tree_info)
        finally:
            try:
                self.synchronize()
            finally:
                self.lock.release()

    def close(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Cannot close InferenceEngine while a request owns it")
        try:
            if self._closed:
                return
            try:
                self.synchronize()
            finally:
                if self._regional_context is not None:
                    self._regional_context.__exit__(None, None, None)
                    self._regional_context = None
                if self._prefix_cache is not None:
                    self._prefix_cache.clear()
                self._closed = True
        finally:
            self.lock.release()
