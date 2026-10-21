from types import SimpleNamespace

import pytest
import torch
from torch.autograd import DeviceType

from memorydraft.backends.asd import ASDConfig
from scripts.run_asd_execution_probe import (CaptureVerifier, interval_union, profile_summary,
                                            require_same, selected_records, strict_core)


def event(name, start, end, *, parent=None, device=DeviceType.CPU, kernels=(), annotation=False):
    return SimpleNamespace(name=name, time_range=SimpleNamespace(start=start, end=end),
                           cpu_parent=parent, device_type=device, kernels=kernels,
                           is_async=False, is_user_annotation=annotation,
                           cpu_time_total=end-start, self_cpu_time_total=end-start)


def test_device_annotations_and_inclusive_times_are_not_added_as_kernels():
    request = event("memorydraft::request", 0, 100)
    scope = event("memorydraft::accept_decision", 10, 90, parent=request,
                  kernels=[SimpleNamespace(name="memorydraft::accept_decision", duration=80)])
    op = event("aten::max", 11, 60, parent=scope,
               kernels=[SimpleNamespace(name="real", duration=40)])
    wait = event("cudaStreamSynchronize", 30, 60, parent=op)
    gpu = event("real", 20, 60, device=DeviceType.CUDA)
    duplicate = event("memorydraft::accept_decision", 10, 90, device=DeviceType.CUDA, annotation=True)
    result = profile_summary([request, scope, op, wait, gpu, duplicate])
    assert result["device_event_count"] == result["linked_kernel_count"] == 1
    assert result["device_active_union_us"] == result["device_duration_sum_us"] == 40
    decision = result["scopes"]["accept_decision"]
    assert decision["host_elapsed_us"] == 80
    assert decision["linked_device_us"] == 40
    assert decision["sync_calls"] == 1 and decision["sync_host_us"] == 30
    assert result["scopes"]["request"]["linked_device_us"] == 0


def test_interval_union_preserves_gaps_and_does_not_double_count_overlap():
    assert interval_union([(0, 10), (2, 4), (8, 12), (20, 30)]) == 22
    assert interval_union([]) == 0
    with pytest.raises(ValueError, match="Negative"):
        interval_union([(3, 2)])


def test_missing_device_events_fail_instead_of_reporting_zero_gpu_cost():
    with pytest.raises(ValueError, match="Missing CUPTI"):
        profile_summary([event("memorydraft::request", 0, 10)])


def test_selection_is_first_two_development_tasks_per_family():
    records = selected_records()
    assert len(records) == 12 and len({r["id"] for r in records}) == 12
    assert [r["id"] for r in records[:4]] == ["gsm8k:885", "gsm8k:52", "math500:329", "math500:448"]
    assert all(r["split"] == "development" for r in records)


def test_capture_preserves_actual_decision_and_bounds_tensor_copies():
    policy = CaptureVerifier(ASDConfig(budget=8))
    proposal = torch.tensor([0, 1, 1])
    logits = torch.tensor([[1., 0.], [0., 1.], [1., .9]])
    for _ in range(16):
        result = policy.decide(proposal, logits)
    assert [s["round"] for s in policy.states] == [0, 4, 12]
    assert all(strict_core(s["proposal"], s["logits"]) == s["expected"].strict_accepted for s in policy.states)
    assert policy.states[0]["logits"].data_ptr() != logits.data_ptr()
    assert policy.states[-1]["expected"] == result


def test_identity_check_rejects_budget_changes_even_with_same_tokens():
    from scripts.run_asd_execution_probe import IDENTITY
    row = {k: None for k in IDENTITY}
    row.update(token_ids=[1, 2], asd={"spent": 1.})
    require_same(row, dict(row))
    with pytest.raises(ValueError, match="asd"):
        require_same(row, {**row, "asd": {"spent": 2.}})


@pytest.fixture
def report_fixture():
    """Synthetic timing/profile fields over saved trajectories, never GPU evidence."""
    from copy import deepcopy
    from pathlib import Path
    from memorydraft.evaluation.runtime_migration import read_rows
    from scripts.run_asd_execution_probe import CAPTURE_ROUNDS, METHODS
    root = Path(__file__).resolve().parents[1]
    ids = {r["id"] for r in selected_records()}
    previous = {(r["request_id"], r["method"]): r for i in (0, 1)
                for r in read_rows(root / f"results/asd-block-suite/part-{i}/rows.jsonl")
                if r["stage"] == "prefix" and r["repeat"] == 0 and r["experimental_block_size"] == 16
                and r["request_id"] in ids and r["method"] in METHODS}
    timed, profiles, captures = [], [], []
    for old in previous.values():
        for repeat in range(3):
            timed.append(dict(deepcopy(old), stage="timed", repeat=repeat))
        row = deepcopy(old)
        row["instrumented_latency_seconds"] = row.pop("latency_seconds")
        row.update(stage="profile", profile=dict(
            scopes={name: dict(calls=len(old["rounds"]), linked_device_us=1.) for name in
                    ("draft_forward", "target_verify", "accept_decision", "commit_kv", "publish_stop")},
            device_duration_sum_us=5., device_active_union_us=4., linked_kernel_count=5, device_event_count=5))
        profiles.append(row)
        if old["method"] == "DFlash":
            continue
        row = deepcopy(row)
        row.update(stage="capture", resident_replays=[])
        for index in CAPTURE_ROUNDS:
            if index >= len(old["rounds"]):
                continue
            rd = old["rounds"][index]
            row["resident_replays"].append(dict(round=index, spent_before=sum(r["regret"] for r in old["rounds"][:index]),
                proposals=rd["proposed"], expected=dict(accepted=rd["accepted"], strict_accepted=rd["strict_accepted"],
                relaxed=rd["relaxed_tokens"], regret=rd["regret"]),
                samples_seconds={k: [.001]*30 for k in ("policy", "strict_core", "sync_control")},
                median_seconds={k: .001 for k in ("policy", "strict_core", "sync_control")}))
        captures.append(row)
    return timed, profiles, captures, previous


def test_report_audit_accepts_complete_preserved_matrix(report_fixture):
    from scripts.write_asd_execution_report import audit
    result = audit(*report_fixture)
    assert (result["timed"], result["profile"], result["capture"]) == (144, 48, 36)
    assert result["baseline_trajectory_checks"] == 228


@pytest.mark.parametrize("change", ["missing", "mixed-timing", "duplicate-device", "budget", "replay"])
def test_report_audit_rejects_corruption(report_fixture, change):
    from scripts.write_asd_execution_report import audit
    timed, profiles, captures, previous = report_fixture
    if change == "missing":
        timed.pop()
    elif change == "mixed-timing":
        profiles[0]["latency_seconds"] = 1.
    elif change == "duplicate-device":
        profiles[0]["profile"]["device_duration_sum_us"] *= 2
    elif change == "budget":
        captures[0]["resident_replays"][0]["spent_before"] += 1
    else:
        captures[0]["resident_replays"][0]["samples_seconds"]["policy"].pop()
    with pytest.raises(ValueError):
        audit(timed, profiles, captures, previous)
