"""Checked, startup-only loading of FP32 q/v rank8/alpha16 dense adapters."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import time


@dataclass(frozen=True)
class DraftAdapterSpec:
    path: str
    sha256: str

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path or not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("draft_adapter requires a local path and lowercase SHA-256")

    def verify(self):
        path = Path(self.path)
        if not path.is_file() or path.suffix != ".safetensors":
            raise ValueError("draft_adapter must be a prepared local .safetensors file")
        if not 0 < path.stat().st_size <= 8*2**20:
            raise ValueError("draft_adapter exceeds the bounded rank8 asset size")
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.sha256:
            raise ValueError("draft_adapter SHA-256 mismatch")


def merge_draft_adapter(draft, spec: DraftAdapterSpec):
    import torch
    from safetensors.torch import load_file
    from memorydraft.evaluation.adaptation import AdapterBank

    if draft is None or getattr(draft, "_memorydraft_adapter_sha256", None) is not None:
        raise ValueError("Load a fresh base draft before merging an adapter")
    started = time.perf_counter()
    spec.verify()
    projections = [m for n,m in draft.named_modules() if n.rsplit(".",1)[-1] in {"q_proj","v_proj"}]
    if not projections or any(type(m) is not torch.nn.Linear or m.weight.dtype != torch.float32 for m in projections):
        raise ValueError("draft_adapter requires native FP32 Linear q/v projections")
    device = projections[0].weight.device
    state = load_file(spec.path, device="cpu")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    AdapterBank(draft).select(state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    draft._memorydraft_adapter_sha256 = spec.sha256
    return dict(sha256=spec.sha256, rank=8, alpha=16, parameters=sum(t.numel() for t in state.values()),
                application="startup_dense_merge", setup_seconds=time.perf_counter()-started)
