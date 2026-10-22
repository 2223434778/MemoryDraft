from copy import deepcopy
import json

import pytest

from scripts.validate_asd_runtime import check_result


def pair():
    reference = dict(policy="ASD",token_ids=[1,2],text="x",finish_reason="eos",prompt_sha256="p",snapshot_id="s",
                     execution_profile="rmsnorm-inductor",rounds=[{}],asd=dict(budget=8,spent=1,relaxed_tokens=1,blocks=1))
    row = {k:v for k,v in reference.items() if k not in {"policy","rounds"}}
    row.update(approximate=True,draft_adapter_sha256="fixture",counters=dict(rounds=1),asd={**reference["asd"],"gate":.25,"cap":2})
    return row,reference


def test_acceptance_matches_trajectory_budget_and_profile():
    row,reference = pair()
    assert check_result(row,reference,adapter_sha="fixture")


@pytest.mark.parametrize("field",["token_ids","text","finish_reason","prompt_sha256","snapshot_id","execution_profile",
                                  "approximate","draft_adapter_sha256","rounds","budget","spent","gate","cap"])
def test_acceptance_rejects_incorrect_metadata_or_trajectory(field):
    row,reference = pair()
    row = deepcopy(row)
    if field == "rounds":row["counters"]["rounds"] = 2
    elif field in {"budget","spent","gate","cap"}:row["asd"][field] = 99
    elif field == "token_ids":row[field] = [3]
    elif field == "approximate":row[field] = False
    else:row[field] = "wrong"
    with pytest.raises(ValueError):check_result(row,reference,adapter_sha="fixture")


def test_report_rejects_failed_execution_before_reading_results(tmp_path):
    from scripts.write_asd_runtime_report import analyze
    directory = tmp_path/"results/asd-runtime"
    directory.mkdir(parents=True)
    (directory/"execution-verification.json").write_text(json.dumps(dict(status="failed")))
    (directory/"ASD_RUNTIME_SOURCE_MANIFEST.json").write_text("{}")
    with pytest.raises(ValueError,match="execution/source"):
        analyze(tmp_path)


def test_report_distinguishes_cold_costs_from_prior_speedup():
    from scripts.write_asd_runtime_report import render
    variant = dict(model_setup_seconds=20,adapter_setup=dict(setup_seconds=2),cold_requests=[dict(seconds=3)],
                   cli_cold_seconds=4,compiler=dict(graph_count=4),peak_gpu_bytes=35*2**30,
                   http=dict(load=dict(successful=4),disconnect_cancelled=True,recovered=True,metrics=dict(failed=0)))
    text = render(dict(source_commit="fixture",frozen_files=10,variants=dict(position=variant)))
    assert "|position|20.000|2.000|3.000|4.000|4|35.000|" in text
    assert "本轮没有重新估计" in text and "包含适配器合并" in text
