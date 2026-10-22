from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


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

    def __post_init__(self):
        if self.backend not in {"transformers", "dflash", "demo"}:
            raise ValueError("backend must be transformers, dflash or demo")
        if self.execution_profile not in {"eager", "rmsnorm-inductor"}:
            raise ValueError("Unknown execution_profile")
        if self.execution_profile != "eager" and self.backend != "dflash":
            raise ValueError("RMSNorm execution requires the audited dflash backend")
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
               (self.threads, self.max_input_tokens, self.max_new_tokens)) or self.max_new_tokens > 4096:
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
        return cls(**value)

    def as_dict(self):
        return asdict(self)


def load_engine(config: RuntimeConfig):
    if config.backend == "demo":
        from memorydraft.runtime.demo import DemoEngine
        return DemoEngine()
    from memorydraft.runtime.engine import InferenceEngine
    return InferenceEngine.load(config)
