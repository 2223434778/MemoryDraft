from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from memorydraft.backends.regional_compile import RegionalRMSNorm
from scripts import run_asd_compile_probe as common
from scripts.run_asd_execution_probe import selected_records
from scripts.run_asd_rmsnorm_probe import PROFILES, ROLES, capture_inputs, expected_states
from scripts.write_asd_rmsnorm_report import audit, summarize


def model(draft=False):
    return torch.nn.ModuleDict({role: Qwen3RMSNorm(4 if role in {"q_norm", "k_norm"} else 16)
                               for role in (*ROLES, *(("hidden_norm",) if draft else ()))})


def test_rmsnorm_restores_bindings_and_never_accepts_mlp_profile():
    target, draft = model(), model(True)
    adapter = RegionalRMSNorm(target, draft, backend="eager")
    pointers = [p.data_ptr() for m in (target, draft) for p in m.parameters()]
    with pytest.raises(RuntimeError, match="fixture"):
        with adapter.activate(PROFILES[1]):
            assert all("forward" in m.__dict__ for _, m in adapter.regions)
            raise RuntimeError("fixture")
    assert all("forward" not in m.__dict__ for _, m in adapter.regions)
    assert pointers == [p.data_ptr() for m in (target, draft) for p in m.parameters()]
    with pytest.raises(ValueError, match="Invalid"):
        with adapter.activate("mlp-inductor"):
            pass
    assert len(adapter.summary()["inventory"]) == 11


def test_real_norms_cover_hidden_head_and_noncontiguous_input():
    torch.set_num_threads(1)
    adapter = RegionalRMSNorm(model(), model(True), backend="eager")
    with torch.inference_mode():
        for length in (3, 7, 1, 16):
            for _, module in adapter.regions:
                dim = module.weight.numel()
                x = torch.randn(1, length, dim) if dim == 16 else torch.randn(1, 2, length, dim).transpose(1, 2)
                expected = module(x)
                with adapter.activate(PROFILES[1]):
                    actual = module(x)
                assert torch.equal(actual, expected)
    assert 1 <= len(adapter.backend.events) <= 8


def test_capture_all_norm_roles_and_remove_hooks_on_error():
    target, draft = model(), model(True)
    adapter = RegionalRMSNorm(target, draft, backend="eager")
    with pytest.raises(RuntimeError, match="fixture"):
        with capture_inputs(adapter) as states:
            for _ in range(3):
                for owner in (target, draft):
                    for module in owner.values():
                        module(torch.randn(1, 3, module.weight.numel()))
            raise RuntimeError("fixture")
    assert sorted((s["region"].split(".")[0], s["region"].split(".")[-1], s["call"]) for s in states) == expected_states()
    assert all(not m._forward_pre_hooks for _, m in adapter.regions)


def test_incomplete_role_inventory_rejected_without_leaking_hooks():
    target, draft = model(), model(True)
    del draft["hidden_norm"]
    adapter = RegionalRMSNorm(target, draft, backend="eager")
    with pytest.raises(ValueError, match="hidden_norm"):
        with capture_inputs(adapter):
            pass
    assert all(not m._forward_pre_hooks for _, m in adapter.regions)


@pytest.fixture
def matrix():
    """Synthetic timing fields over prior real trajectories, never GPU evidence."""
    previous = common.previous_rows()
    rows, preflight, diagnostics = [], [], []
    for record in selected_records():
        q = record["id"]
        for method in ("AR", *common.METHODS):
            for profile in PROFILES:
                latency = 1. if profile == "eager" else .9
                row = dict(deepcopy(previous[q, method]), profile=profile, stage="preflight", repeat=0,
                           latency_seconds=latency, engine_latency_seconds=latency, policy_setup_seconds=0.,
                           profile_switch_seconds=0., graphs_before=4, graphs_after=4)
                preflight.append(row)
                if method != "AR":
                    for repeat in range(common.REPEATS):
                        rows.append(dict(deepcopy(row), stage="timed", repeat=repeat,
                                         slot=common.timing_order(q, repeat, profiles=PROFILES).index((method, profile))))
        d = deepcopy(previous[q, "DFlash"])
        d.pop("latency_seconds", None)
        d.update(stage="diagnostic", instrumented_latency_seconds=1., rmsnorm_replays=[])
        for owner, role, call in expected_states():
            d["rmsnorm_replays"].append(dict(region=owner+".fixture."+role, call=call,
                numeric=dict(bitwise_equal=False, max_abs=1e-6, rms=1e-7, max_relative_floor_1e_6=1e-3),
                samples_seconds={p: [.001]*common.REPLAY_REPEATS for p in PROFILES},
                median_seconds={p: .001 for p in PROFILES}))
        diagnostics.append(d)
    return rows, preflight, diagnostics, previous


def test_norm_report_complete_matrix_and_independent_profile(matrix):
    rows, preflight, diagnostics, previous = matrix
    result = audit(rows, preflight, diagnostics, previous)
    assert result["rmsnorm_states"] == 192 and result["timed"] == 288
    assert result["compiled_token_matches"] == 60
    assert all(x["ratio"] == pytest.approx(.9) for x in summarize(rows, preflight, diagnostics)["overall"].values())


@pytest.mark.parametrize("change", ["mlp", "role", "missing", "repeat", "ledger"])
def test_norm_audit_rejects_profile_role_or_trajectory_corruption(matrix, change):
    rows, preflight, diagnostics, previous = matrix
    if change == "mlp":
        rows[0]["profile"] = "mlp-inductor"
    elif change == "role":
        diagnostics[0]["rmsnorm_replays"][0]["region"] += "wrong"
    elif change == "missing":
        diagnostics[0]["rmsnorm_replays"].pop()
    elif change == "repeat":
        rows[0]["token_ids"][0] += 1
    else:
        next(r for r in preflight if r["method"] == "ASD" and r["profile"] != "eager")["asd"]["spent"] += .01
    with pytest.raises(ValueError):
        audit(rows, preflight, diagnostics, previous)
