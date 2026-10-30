from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from memorydraft.runtime.contracts import MAX_OUTPUT_TOKENS


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str = "transformers"
    target: str = ""
    target_revision: str = ""
    draft: str = ""
    device: str = "cuda:0"
    dtype: str = "float32"
    attention: str = "eager"
    db: str | None = None
    threads: int = 4
    max_input_tokens: int = 8192
    max_new_tokens: int = 1024
    template_options: dict = field(default_factory=lambda: {"enable_thinking": False})
    reuse: dict = field(default_factory=lambda: {"max_tokens": 32, "min_match": 6,
                                               "cooldown_rounds": 1, "max_pending_tokens": 1024})
    retrieval: str = "suffix"
    alignment: dict = field(default_factory=dict)
    benefit: dict = field(default_factory=dict)
    prefix_risk: dict = field(default_factory=dict)
    execution_profile: str = "eager"
    asd: dict = field(default_factory=dict)
    draft_adapter: dict = field(default_factory=dict)
    prefix_cache: dict = field(default_factory=dict)
    cost_tree: dict = field(default_factory=dict)
    tree_policy: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.backend not in {"transformers", "dflash", "demo"}:
            raise ValueError("backend must be transformers, dflash or demo")
        if self.execution_profile not in {"eager", "rmsnorm-inductor"}:
            raise ValueError("Unknown execution_profile")
        if self.execution_profile != "eager" and self.backend != "dflash":
            raise ValueError("RMSNorm execution requires the audited dflash backend")
        if type(self.cost_tree) is not dict:
            raise ValueError("cost_tree must be an object")
        if self.cost_tree:
            if self.backend != "dflash":
                raise ValueError("Cost tree requires the audited dflash backend")
            from memorydraft.runtime.cost_tree import CostTreeSpec
            try:
                CostTreeSpec(**self.cost_tree)
            except TypeError as error:
                raise ValueError("Invalid cost_tree fields") from error
        if type(self.asd) is not dict or type(self.draft_adapter) is not dict:
            raise ValueError("asd and draft_adapter must be objects")
        if self.asd:
            if self.backend != "dflash" or set(self.asd)-{"budget", "gate", "cap"}:
                raise ValueError("Runtime ASD requires dflash and supports only budget/gate/cap")
            from memorydraft.backends.asd import ASDConfig
            ASDConfig(**self.asd)
        if self.draft_adapter:
            if self.backend != "dflash" or set(self.draft_adapter) != {"path", "sha256"}:
                raise ValueError("draft_adapter requires dflash and path/sha256")
            from memorydraft.runtime.draft_adapter import DraftAdapterSpec
            DraftAdapterSpec(**self.draft_adapter)
        if type(self.prefix_cache) is not dict:
            raise ValueError("prefix_cache must be an object")
        if self.prefix_cache:
            if self.backend != "dflash":
                raise ValueError("Public prefix caching requires the audited dflash backend")
            from memorydraft.runtime.prefix_cache import PrefixCacheSpec
            PrefixCacheSpec(**self.prefix_cache)
        if type(self.tree_policy) is not dict:
            raise ValueError('tree_policy must be an object')
        if self.tree_policy:
            if self.backend != 'dflash' or not self.asd:
                raise ValueError('tree_policy requires dflash and explicit ASD settings')
            from memorydraft.runtime.tree_policy import TreePolicySpec
            try:
                policy = TreePolicySpec(**self.tree_policy)
            except TypeError as error:
                raise ValueError('Invalid tree_policy fields') from error
            if policy.buffered_kv and self.prefix_cache:
                raise ValueError('Buffered tree KV cannot adopt a public prefix cache')
            if self.cost_tree and self.cost_tree.get('top_k', 4) < 2:
                raise ValueError('High-fidelity tree requires top_k >= 2')
        if self.retrieval not in {"suffix", "ot", "edit", "ot-guided", "edit-guided",
                                  "edit-splice", "ot-splice", "ot-tail-splice",
                                  "ot-order-splice", "ot-plan-splice", "ot-all-splice",
                                  "ot-budget-splice", "ot-gate-splice", "edit-budget-splice", "edit-risk-splice"}:
            raise ValueError("Unknown retrieval method")
        if self.alignment or self.retrieval != "suffix":
            from memorydraft.retrieval.transport import AlignmentConfig
            AlignmentConfig(**self.alignment)
        if self.retrieval in {"ot-budget-splice", "ot-gate-splice", "edit-budget-splice"}:
            from memorydraft.retrieval.benefit import BenefitPolicy
            if not {"samples", "costs"} <= self.benefit.keys():
                raise ValueError("Benefit retrieval requires verified samples and measured costs")
            BenefitPolicy(**self.benefit)
        elif self.benefit:
            raise ValueError("Benefit calibration requires a benefit retrieval method")
        if self.retrieval == "edit-risk-splice":
            from memorydraft.retrieval.prefix_risk import PrefixRiskPolicy
            if "weights" not in self.prefix_risk:
                raise ValueError("Prefix risk retrieval requires frozen calibration weights")
            PrefixRiskPolicy(**self.prefix_risk)
        elif self.prefix_risk:
            raise ValueError("Prefix risk calibration requires edit-risk-splice retrieval")
        if self.dtype != "float32" or self.attention != "eager":
            raise ValueError("The validated runtime profile requires float32/eager")
        if any(type(v) is not int or v < 1 for v in
               (self.threads, self.max_input_tokens, self.max_new_tokens)) or self.max_new_tokens > MAX_OUTPUT_TOKENS:
            raise ValueError("Invalid thread or token limits")
        if self.backend != "demo" and not self.target:
            raise ValueError("A local target model directory is required")
        if self.backend == "dflash" and not self.draft:
            raise ValueError("A local audited DFlash directory is required")

    @classmethod
    def load(cls, path: str | Path) -> RuntimeConfig:
        path = Path(path).resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        # Paths are relative to the config file, so installed CLI use is independent of cwd.
        for key in ("target", "draft", "db"):
            if value.get(key):
                candidate = Path(value[key]).expanduser()
                value[key] = str(candidate if candidate.is_absolute() else path.parent / candidate)
        if isinstance(value.get("draft_adapter"), dict) and value["draft_adapter"].get("path"):
            candidate = Path(value["draft_adapter"]["path"]).expanduser()
            value["draft_adapter"]["path"] = str(candidate if candidate.is_absolute() else path.parent/candidate)
        if isinstance(value.get("prefix_cache"), dict) and value["prefix_cache"].get("path"):
            candidate = Path(value["prefix_cache"]["path"]).expanduser()
            value["prefix_cache"]["path"] = str(candidate if candidate.is_absolute() else path.parent/candidate)
        if isinstance(value.get("cost_tree"), dict) and value["cost_tree"].get("path"):
            candidate = Path(value["cost_tree"]["path"]).expanduser()
            value["cost_tree"]["path"] = str(candidate if candidate.is_absolute() else path.parent/candidate)
        return cls(**value)

    def as_dict(self):
        return asdict(self)


def load_engine(config: RuntimeConfig):
    if config.backend == "demo":
        from memorydraft.runtime.demo import DemoEngine
        return DemoEngine()
    from memorydraft.runtime.engine import InferenceEngine
    return InferenceEngine.load(config)
