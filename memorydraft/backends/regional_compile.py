"""Research-only regional compilation with reversible instance bindings."""
from __future__ import annotations

from contextlib import contextmanager
import time
from types import MethodType

import torch


class CompileBackend:
    def __init__(self, name="inductor", *, max_graphs=8, max_seconds=300.):
        self.name, self.max_graphs, self.max_seconds = name, max_graphs, max_seconds
        self.events = []

    def __call__(self, graph, inputs):
        if len(self.events) >= self.max_graphs:
            raise RuntimeError("Regional compilation exceeded the frozen graph budget")
        from torch._dynamo.backends.registry import lookup_backend
        entry = dict(index=len(self.events), status="started",
                     operations=[dict(op=n.op, target=str(n.target)) for n in graph.graph.nodes],
                     inputs=[dict(dtype=str(x.dtype), shape=[str(v) for v in x.shape])
                             if isinstance(x, torch.Tensor) else dict(type=type(x).__name__) for x in inputs])
        self.events.append(entry)
        started = time.perf_counter()
        try:
            backend = lookup_backend(self.name)
            if self.name == "inductor":
                from torch._inductor import config
                with config.patch({"max_autotune": False, "triton.cudagraphs": False}):
                    result = backend(graph, inputs)
            else:
                result = backend(graph, inputs)
            entry["status"] = "complete"
        except Exception as error:
            entry.update(status="failed", error=type(error).__name__+": "+str(error))
            raise
        finally:
            entry["backend_seconds"] = time.perf_counter()-started
        if sum(e["backend_seconds"] for e in self.events) > self.max_seconds:
            raise RuntimeError("Regional compilation exceeded the frozen backend time budget")
        return result


class _RegionalCompiler:
    def __init__(self, target, draft, region_type, profile, *, backend="inductor", max_graphs=8, max_seconds=300.):
        self.profile = profile
        self.regions = [(owner+"."+name, module) for owner, model in (("target", target), ("draft", draft))
                        for name, module in model.named_modules() if isinstance(module, region_type)]
        if not self.regions or any(not any(n.startswith(owner+".") for n, _ in self.regions) for owner in ("target", "draft")):
            raise ValueError("Expected audited "+region_type.__name__+" in both Target and Draft")
        if any("forward" in m.__dict__ for _, m in self.regions):
            raise ValueError("Refusing to overwrite an existing instance forward adapter")
        self.backend = CompileBackend(backend, max_graphs=max_graphs, max_seconds=max_seconds)
        self.compiled = {cls: torch.compile(cls.forward, backend=self.backend, fullgraph=True, dynamic=True)
                         for cls in {type(m) for _, m in self.regions}}
        self.active = False
        self.switch_seconds = 0.

    @contextmanager
    def activate(self, profile):
        if profile not in {"eager", self.profile} or self.active:
            raise ValueError("Invalid or nested regional execution profile")
        changed = []
        started = time.perf_counter()
        self.active = True
        try:
            if profile == self.profile:
                for _, module in self.regions:
                    module.forward = MethodType(self.compiled[type(module)], module)
                    changed.append(module)
            self.switch_seconds += time.perf_counter()-started
            yield
        finally:
            started = time.perf_counter()
            for module in changed:
                del module.forward
            self.active = False
            self.switch_seconds += time.perf_counter()-started

    def summary(self):
        from torch._dynamo import utils
        metrics = {str(k): [float(v) for v in values] for k, values in utils.compilation_time_metrics.items()}
        return dict(backend=self.backend.name, fullgraph=True, dynamic=True,
                    max_autotune=False, cudagraphs=False, regions=[n for n, _ in self.regions],
                    graph_count=len(self.backend.events), events=self.backend.events,
                    backend_seconds=sum(e.get("backend_seconds", 0.) for e in self.backend.events),
                    compilation_time_metrics=metrics,
                    outer_compile_seconds=sum(metrics["_compile.compile_inner"]) if "_compile.compile_inner" in metrics else None,
                    compile_times=utils.compile_times(), switch_seconds=self.switch_seconds,
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    allow_tf32=torch.backends.cuda.matmul.allow_tf32)


class RegionalMLP(_RegionalCompiler):
    """Compile the existing Qwen3MLP.forward; never rewrite its arithmetic/KV."""
    def __init__(self, target, draft, **kwargs):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
        super().__init__(target, draft, Qwen3MLP, "mlp-inductor", **kwargs)


class RegionalRMSNorm(_RegionalCompiler):
    """Compile original RMSNorm reductions, including Q/K head normalization."""
    def __init__(self, target, draft, **kwargs):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        super().__init__(target, draft, Qwen3RMSNorm, "rmsnorm-inductor", **kwargs)

    def summary(self):
        return dict(super().summary(), inventory=[dict(region=n, type=type(m).__name__,
                    weight_shape=list(m.weight.shape), eps=m.variance_epsilon) for n, m in self.regions])
