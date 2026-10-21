from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from memorydraft.backends.regional_compile import RegionalMLP
from scripts import run_asd_compile_probe as runner
from scripts.run_asd_execution_probe import selected_records
from scripts.write_asd_compile_report import audit, summarize


def test_profile_diff_preserves_numerical_output_changes_but_rejects_input_changes():
    previous = runner.previous_rows()
    row = previous[selected_records()[0]["id"], "ASD"]
    changed = deepcopy(row)
    changed["token_ids"][0] += 1
    result = runner.trajectory_comparison(row, changed)
    assert not result["token_equal"] and result["first_token_difference"] == 0
    changed["prompt_sha256"] = "changed"
    with pytest.raises(ValueError, match="input"):
        runner.trajectory_comparison(row, changed)


def test_execution_uses_actual_generation_result_memory_field_and_counts_switch(monkeypatch):
    config = Qwen3Config(hidden_size=16, intermediate_size=32, hidden_act="silu")
    model = lambda: torch.nn.ModuleList([Qwen3MLP(config)])
    compiler = RegionalMLP(model(), model(), backend="eager")
    monkeypatch.setattr(runner.suite, "execute", lambda *a, **kw: dict(latency_seconds=1., peak_gpu_bytes=17))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: 23)
    engine = SimpleNamespace(target=SimpleNamespace(device=torch.device("cpu")))
    result = runner.execute(engine, compiler, {}, "AR", "mlp-inductor")
    assert result["peak_gpu_bytes"] == 17 and result["peak_reserved_bytes"] == 23
    assert result["latency_seconds"] == 1.+result["policy_setup_seconds"]+result["profile_switch_seconds"]
    assert not compiler.active


def test_input_hooks_capture_only_selected_calls_and_are_removed():
    config = Qwen3Config(hidden_size=16, intermediate_size=32, hidden_act="silu")
    target, draft = torch.nn.ModuleList([Qwen3MLP(config)]), torch.nn.ModuleList([Qwen3MLP(config)])
    compiler = RegionalMLP(target, draft, backend="eager")
    x = torch.randn(1, 3, 16)
    with runner.capture_inputs(compiler) as states:
        for _ in range(4):
            target[0](x); draft[0](x)
    assert [(s["region"].split(".")[0], s["call"]) for s in states] == [("target", 0), ("draft", 0), ("target", 1)]
    assert all(s["input"].data_ptr() != x.data_ptr() for s in states)
    assert not target[0]._forward_pre_hooks and not draft[0]._forward_pre_hooks


@pytest.fixture
def matrix_fixture():
    """Synthetic performance/profile fields over old trajectories; not GPU data."""
    previous = runner.previous_rows()
    preflight, rows, diagnostics = [], [], []
    for record in selected_records():
        q = record["id"]
        for method in ("AR", *runner.METHODS):
            for profile in runner.PROFILES:
                r = deepcopy(previous[q, method])
                latency = 1. if profile == "eager" else .95
                r.update(profile=profile, stage="preflight", repeat=0, latency_seconds=latency,
                         engine_latency_seconds=latency, policy_setup_seconds=0., profile_switch_seconds=0.,
                         graphs_before=2, graphs_after=2)
                preflight.append(r)
                if method != "AR":
                    for repeat in range(runner.REPEATS):
                        rows.append(dict(deepcopy(r), stage="timed", repeat=repeat,
                                         slot=runner.timing_order(q, repeat).index((method, profile))))
        d = deepcopy(previous[q, "DFlash"])
        d.pop("latency_seconds", None)
        d.update(stage="diagnostic", instrumented_latency_seconds=1., mlp_replays=[])
        for owner, index in (("target", 0), ("target", 1), ("draft", 0)):
            d["mlp_replays"].append(dict(region=owner+".fixture.mlp", call=index,
                numeric=dict(bitwise_equal=False, max_abs=1e-6, rms=1e-7, max_relative_floor_1e_6=1e-3),
                samples_seconds={p: [.001]*runner.REPLAY_REPEATS for p in runner.PROFILES},
                median_seconds={p: .001 for p in runner.PROFILES}))
        diagnostics.append(d)
    return rows, preflight, diagnostics, previous


def test_compiler_report_complete_matrix_and_question_level_estimates(matrix_fixture):
    rows, preflight, diagnostics, previous = matrix_fixture
    result = audit(rows, preflight, diagnostics, previous)
    assert (result["timed"], result["preflight"], result["mlp_states"]) == (288, 120, 36)
    assert result["compiled_token_matches"] == result["compiled_ledger_matches"] == 60
    summary = summarize(rows, preflight, diagnostics)
    assert all(x["ratio"] == pytest.approx(.95) for x in summary["overall"].values())


def test_output_drift_is_retained_and_recompilation_cost_is_not_removed(matrix_fixture):
    rows, preflight, diagnostics, previous = matrix_fixture
    changed = next(r for r in preflight if r["profile"] == "mlp-inductor" and r["method"] == "ASD")
    changed["token_ids"][0] += 1
    selected = [r for r in rows if (r["request_id"], r["method"], r["profile"]) ==
                (changed["request_id"], changed["method"], changed["profile"])]
    for r in selected:
        r["token_ids"][0] += 1
    selected[0]["graphs_after"] += 1
    result = audit(rows, preflight, diagnostics, previous)
    assert result["compiled_token_matches"] == 59 and result["backend_events_during_timing"] == 1


@pytest.mark.parametrize("change", ["missing", "input", "repeat", "ledger", "timing", "states"])
def test_report_rejects_invalid_measurements(matrix_fixture, change):
    rows, preflight, diagnostics, previous = matrix_fixture
    if change == "missing":
        rows.pop()
    elif change == "input":
        preflight[0]["prompt_sha256"] = "bad"
    elif change == "repeat":
        rows[0]["token_ids"][0] += 1
    elif change == "ledger":
        r = next(r for r in preflight if r["method"] == "ASD" and r["profile"] == "mlp-inductor")
        r["asd"]["spent"] += .001
    elif change == "timing":
        rows[0]["profile_switch_seconds"] = 1.
    else:
        diagnostics[0]["mlp_replays"].pop()
    with pytest.raises(ValueError):
        audit(rows, preflight, diagnostics, previous)
