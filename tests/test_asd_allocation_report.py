from copy import deepcopy

import pytest

from memorydraft.backends.allocation import fit_stump, predict
from scripts.write_asd_allocation_report import analyze, audit_resources, samples_from


def rows(stage):
    result = []
    actions = ("native", "half", "verify-half", "target" if stage == "branch" else "AR")
    for q in range(8):
        for action in actions:
            for repeat in range(3):
                value = (1. if action == ("native" if q < 4 else "half") else 2.)
                result.append(dict(request_id=f"q{q}", benchmark="left" if q < 4 else "right", action=action,
                                   token_ids=[1, 2], prompt_sha256=f"p{q}", snapshot_id="s", rounds=[], repeat=repeat,
                                   token_match=True, prompt_tokens=10 if q < 4 else 1000,
                                   source_round=0, latency_seconds=value, local_seconds=value,
                                   features=dict(generated=1, prompt_tokens=10 if q < 4 else 1000, recent_advance=1, last_advance=1)))
    return result


def test_oracle_predictor_and_scenario_gates_use_other_requests():
    result = analyze(rows("request"), rows("branch"))
    assert result["continue_selector"]
    for stage in ("request", "branch"):
        overall = result[stage]["overall"]
        assert overall["safe"] and overall["optimistic_oracle_ratio"] < 1
        for q, model in overall["models"].items():
            training = [s for s in result[stage]["samples"] if s["request_id"] != q]
            actions = tuple(result[stage]["samples"][0]["costs"])
            assert model == fit_stump(training, actions)
        for family in result[stage]["families"].values():
            for q in family["choices"]:
                assert q["action"] == predict(overall["models"][q["request_id"]], next(
                    s["features"] for s in result[stage]["samples"] if s["request_id"] == q["request_id"]))


@pytest.mark.parametrize("corrupt", ["duplicate", "missing", "unstable", "zero", "nan"])
def test_incomplete_or_unusable_measurements_rejected(corrupt):
    data = rows("request")
    if corrupt == "duplicate":
        data.append(deepcopy(data[0]))
    elif corrupt == "missing":
        data.pop()
    elif corrupt == "unstable":
        data[0]["token_ids"] = [3]
    elif corrupt == "zero":
        data[0]["latency_seconds"] = 0
    else:
        data[0]["latency_seconds"] = float("nan")
    with pytest.raises(ValueError):
        samples_from(data, "request")


def test_numerical_divergence_remains_reported_and_blocks_admission():
    a, b = rows("request"), rows("branch")
    for data in (a, b):
        for r in data:
            if r["action"] == "half":
                r["token_ids"] = [3]
    result = analyze(a, b)
    assert not result["continue_selector"]
    assert not result["request"]["overall"]["safe"]


def test_fixed_optimum_has_no_adaptation_gate():
    a, b = rows("request"), rows("branch")
    for data in (a, b):
        for r in data:
            r["local_seconds"] = r["latency_seconds"] = 1 if r["action"] == "native" else 2
    result = analyze(a, b)
    assert not result["continue_selector"]
    assert result["branch"]["overall"]["optimistic_oracle_ratio"] == 1


@pytest.mark.parametrize("bad", ["memory", "precision", "graphs", "cache"])
def test_resource_limits_remain_admission_requirements(bad):
    info = dict(cache_bytes=10, compiler=dict(graph_count=4, backend_seconds=2,
                float32_matmul_precision="highest", allow_tf32=False, events=[dict(status="complete")]))
    row = dict(peak_gpu_bytes=100, counters=dict(compile_graphs_before=4, compile_graphs_after=4))
    if bad == "memory":
        row["peak_gpu_bytes"] = 47*2**30
    elif bad == "precision":
        info["compiler"]["allow_tf32"] = True
    elif bad == "graphs":
        info["compiler"]["graph_count"] = 9
    else:
        info["cache_bytes"] = 5*2**30
    with pytest.raises(ValueError):
        audit_resources([info], [row])
