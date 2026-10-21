from types import SimpleNamespace
import json

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from memorydraft.backends.regional_compile import CompileBackend, RegionalMLP


def model(n=1):
    config = Qwen3Config(hidden_size=16, intermediate_size=32, hidden_act="silu")
    return torch.nn.ModuleList([Qwen3MLP(config).eval() for _ in range(n)])


def test_profile_restores_exact_bindings_on_exception_and_keeps_weights():
    target, draft = model(), model()
    pointers = [p.data_ptr() for m in (target, draft) for p in m.parameters()]
    adapter = RegionalMLP(target, draft, backend="eager")
    with pytest.raises(RuntimeError, match="fixture"):
        with adapter.activate("mlp-inductor"):
            assert "forward" in target[0].__dict__ and "forward" in draft[0].__dict__
            raise RuntimeError("fixture")
    assert not adapter.active and all("forward" not in m.__dict__ for _, m in adapter.regions)
    assert pointers == [p.data_ptr() for m in (target, draft) for p in m.parameters()]
    with adapter.activate("eager"):
        assert "forward" not in target[0].__dict__


def test_repeated_real_mlp_instances_share_bounded_dynamic_graphs():
    # Dynamo/eager tests capture and binding semantics; this is NOT an Inductor
    # performance test. Actual CUDA Inductor is measured on the Linux server.
    torch.set_num_threads(1)
    torch.manual_seed(951)
    target, draft = model(9), model(2)
    adapter = RegionalMLP(target, draft, backend="eager")
    with torch.inference_mode():
        for length in (2, 7, 1, 16):
            x = torch.randn(1, length, 16)
            expected = [m(x) for _, m in adapter.regions]
            with adapter.activate("mlp-inductor"):
                actual = [m(x) for _, m in adapter.regions]
            assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert 1 <= len(adapter.backend.events) <= 4
    assert all(e["status"] == "complete" for e in adapter.backend.events)
    summary = adapter.summary()
    assert summary["graph_count"] == len(adapter.backend.events)
    json.dumps(summary)


def test_nested_and_unknown_profiles_are_rejected_and_recoverable():
    adapter = RegionalMLP(model(), model(), backend="eager")
    with adapter.activate("eager"):
        with pytest.raises(ValueError, match="nested"):
            with adapter.activate("eager"):
                pass
    with pytest.raises(ValueError, match="Invalid"):
        with adapter.activate("invalid"):
            pass
    assert not adapter.active


def test_missing_mlp_and_existing_adapter_are_not_silently_overwritten():
    with pytest.raises(ValueError, match="both"):
        RegionalMLP(model(), torch.nn.Linear(2, 2), backend="eager")
    target = model()
    target[0].forward = target[0].forward
    with pytest.raises(ValueError, match="overwrite"):
        RegionalMLP(target, model(), backend="eager")


def test_backend_budget_fails_instead_of_silently_running_eager():
    backend = CompileBackend("eager", max_graphs=0)
    with pytest.raises(RuntimeError, match="graph budget"):
        backend(SimpleNamespace(), [])
