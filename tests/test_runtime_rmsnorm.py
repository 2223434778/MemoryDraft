from dataclasses import asdict, replace
from types import SimpleNamespace
import json
import sys

import pytest
import torch
import transformers
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from memorydraft.backends import regional_compile as region
from memorydraft.runtime.config import RuntimeConfig
from memorydraft.runtime.contracts import RequestCancelled
from memorydraft.runtime.engine import InferenceEngine
from memorydraft.runtime.demo import DemoEngine
from memorydraft.runtime.benchmark import benchmark, check_regression, compare
from test_runtime import engine, request


def test_explicit_config_preserves_eager_default_and_rejects_incompatible_backend():
    assert RuntimeConfig(target="fixture").execution_profile == "eager"
    assert RuntimeConfig("transformers", "fixture", "", "", "cuda:0", "float32", "eager", "history.db").db == "history.db"
    for backend in ("demo", "transformers"):
        with pytest.raises(ValueError, match="dflash"):
            RuntimeConfig(target="fixture", backend=backend, execution_profile="rmsnorm-inductor")
    with pytest.raises(ValueError, match="execution_profile"):
        RuntimeConfig(target="fixture", execution_profile="unknown")
    config = RuntimeConfig(backend="dflash", target="fixture", draft="fixture", execution_profile="rmsnorm-inductor")
    assert RuntimeConfig(**asdict(config)) == config


@pytest.mark.parametrize("wrong", ["platform", "torch", "transformers", "device", "dtype", "draft"])
def test_unsupported_runtime_stack_fails_without_fallback(monkeypatch, wrong):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(torch, "__version__", "2.5.1+cu124")
    monkeypatch.setattr(transformers, "__version__", "4.57.3")
    target = SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.float32)
    draft = SimpleNamespace(device=target.device, dtype=target.dtype)
    if wrong == "platform":
        monkeypatch.setattr(sys, "platform", "win32")
    elif wrong == "torch":
        monkeypatch.setattr(torch, "__version__", "2.6.0")
    elif wrong == "transformers":
        monkeypatch.setattr(transformers, "__version__", "4.52.2")
    elif wrong == "device":
        target.device = torch.device("cpu")
    elif wrong == "dtype":
        draft.dtype = torch.bfloat16
    else:
        draft = None
    with pytest.raises((ValueError, RuntimeError), match="rmsnorm-inductor"):
        region.runtime_rmsnorm(target, draft)


@pytest.fixture
def compiled(engine, monkeypatch):
    """CPU Dynamo/eager contract fixture; not CUDA/Inductor performance evidence."""
    torch._dynamo.reset()
    original = engine.generate(request(), "AR").token_ids
    engine.close()
    engine.draft.norm = Qwen3RMSNorm(engine.target.config.hidden_size)
    monkeypatch.setattr(region, "runtime_rmsnorm", lambda target, draft: region.RegionalRMSNorm(target, draft, backend="eager"))
    config = replace(engine.config, backend="dflash", draft="fixture", execution_profile="rmsnorm-inductor")
    result = InferenceEngine(engine.target, engine.tokenizer, config, engine.draft)
    result.original_test_tokens = original
    try:
        yield result
    finally:
        result.close()


@pytest.mark.parametrize("method", ["AR", "DFlash", "R+DFlash"])
def test_persistent_profile_streams_committed_tokens_and_recovers_after_cancel(compiled, method):
    callbacks = []
    row = compiled.generate(request(), method, on_tokens=lambda tokens: callbacks.extend(tokens))
    assert callbacks == row.token_ids == compiled.original_test_tokens
    assert row.execution_profile == compiled.describe()["execution_profile"] == "rmsnorm-inductor"
    assert compiled._regional.active
    callbacks.clear()
    with pytest.raises(RequestCancelled):
        compiled.generate(request(), method, on_tokens=lambda tokens: callbacks.extend(tokens), should_stop=lambda: bool(callbacks))
    assert callbacks and not compiled.lock.locked() and compiled._regional.active
    assert compiled.generate(request(), method).token_ids == row.token_ids


def test_cold_compilation_is_inside_request_clock_and_warm_request_has_no_new_events(compiled):
    cold = compiled.generate(request(), "AR")
    counters = cold.counters
    assert counters["compile_graphs_before"] == 0 and counters["compile_graphs_after"] > 0
    assert 0 <= counters["compile_backend_seconds"] <= cold.latency_seconds
    warm = compiled.generate(request(), "AR")
    assert warm.counters["compile_graphs_before"] == warm.counters["compile_graphs_after"]
    assert warm.counters["compile_backend_seconds"] == 0
    assert warm.token_ids == cold.token_ids


def test_close_refuses_active_owner_then_restores_all_bindings_and_is_idempotent(compiled):
    modules = [m for _, m in compiled._regional.regions]
    pointers = [p.data_ptr() for m in modules for p in m.parameters()]
    compiled.lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="owns"):
            compiled.close()
        assert compiled._regional.active
    finally:
        compiled.lock.release()
    compiled.close()
    compiled.close()
    assert not compiled._regional.active and all("forward" not in m.__dict__ for m in modules)
    assert pointers == [p.data_ptr() for m in modules for p in m.parameters()]
    with pytest.raises(RuntimeError, match="closed"):
        compiled.generate(request(), "AR")


def test_profile_mutation_requires_reload_and_releases_owner(compiled):
    compiled.config = replace(compiled.config, execution_profile="eager")
    with pytest.raises(ValueError, match="Reload"):
        compiled.generate(request(), "AR")
    assert not compiled.lock.locked()


def test_compile_failure_is_not_hidden_and_owner_is_released(compiled):
    compiled._regional.backend.max_graphs = 0
    with pytest.raises(Exception, match="graph budget"):
        compiled.generate(request(), "AR")
    assert not compiled.lock.locked() and compiled._regional.active


def test_close_restores_bindings_even_if_synchronization_fails(compiled, monkeypatch):
    def fail():
        raise RuntimeError("fixture synchronization")
    monkeypatch.setattr(compiled, "synchronize", fail)
    with pytest.raises(RuntimeError, match="fixture synchronization"):
        compiled.close()
    assert not compiled.lock.locked() and not compiled._regional.active
    assert all("forward" not in m.__dict__ for _, m in compiled._regional.regions)


def test_benchmark_refuses_comparing_different_numerical_profiles(tmp_path):
    class FixtureEngine(DemoEngine):
        def __init__(self, profile):
            self.profile = profile

        def describe(self):
            return dict(synthetic=False, backend="unit-test-only", torch="test", transformers="test", gpu="test",
                        device="test", dtype="test", target_revision="pinned", execution_profile=self.profile)

        def generate(self, *args, **kwargs):
            row = super().generate(*args, **kwargs)
            row.execution_profile = self.profile
            return row

    # Synthetic transport data only exercises compatibility, not GPU performance.
    for profile in ("eager", "rmsnorm-inductor"):
        benchmark(FixtureEngine(profile), [request()], ["AR"], tmp_path/profile, repeats=1, warmup=False)
    with pytest.raises(ValueError, match="execution_profile"):
        check_regression(tmp_path/"rmsnorm-inductor", tmp_path/"eager")
    a = SimpleNamespace(token_ids=[1], prompt_sha256="p", snapshot_id="s", execution_profile="eager")
    b = SimpleNamespace(token_ids=[1], prompt_sha256="p", snapshot_id="s", execution_profile="rmsnorm-inductor")
    assert not compare(a, b)


@pytest.mark.parametrize("failure", [None, "warm_compile", "no_cold_compile", "wrong_profile", "failed_stream", "precision", "memory"])
def test_acceptance_report_rejects_incomplete_or_incompatible_evidence(tmp_path, failure):
    from scripts.run_rmsnorm_runtime_pipeline import check_results
    row = dict(execution_profile="rmsnorm-inductor", correct=True, peak_gpu_bytes=1,
               counters=dict(compile_graphs_before=4, compile_graphs_after=4))
    rows = [dict(row, counters=dict(row["counters"])) for _ in range(12)]
    cold = [dict(row, counters=dict(compile_graphs_before=0, compile_graphs_after=2)), row]
    streams = {m: dict(correct=True, execution_profile="rmsnorm-inductor") for m in ("AR", "DFlash", "R+DFlash")}
    compiler = dict(graph_count=4, backend_seconds=1, float32_matmul_precision="highest", allow_tf32=False)
    if failure == "warm_compile":
        rows[-1]["counters"]["compile_graphs_after"] += 1
    elif failure == "no_cold_compile":
        cold[0]["counters"]["compile_graphs_after"] = 0
    elif failure == "wrong_profile":
        rows[-1]["execution_profile"] = "eager"
    elif failure == "failed_stream":
        streams["R+DFlash"]["correct"] = False
    elif failure == "precision":
        compiler["allow_tf32"] = True
    elif failure == "memory":
        rows[-1]["peak_gpu_bytes"] = 47*2**30
    acceptance = dict(status="complete", history_unchanged=True, cancellation_recovered=True,
                      reuse_accepted_tokens=8, http_load=dict(failed=0), profiler=dict(cuda_kernel_events=1),
                      cold_start=cold, http_streams=streams, compiler=compiler)
    directory = tmp_path/"acceptance/benchmark"
    directory.mkdir(parents=True)
    (directory/"rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (directory.parent/"acceptance.json").write_text(json.dumps(acceptance))
    (tmp_path/"cli.json").write_text(json.dumps(dict(row, token_ids=[1], method="DFlash")))
    if failure is None:
        assert check_results(tmp_path, 100)["all_rows_correct"]
    else:
        with pytest.raises(ValueError):
            check_results(tmp_path, 100)
