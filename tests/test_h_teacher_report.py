import copy

import pytest

from scripts.write_h_teacher_router_report import METHODS, same_prefix_audit, validate_rows


def test_report_rejects_incomplete_wrong_target_or_snapshot_and_h_off_activation():
    snapshots = dict(a=dict(snapshot_id="teacher"))
    references = dict(a=dict(token_ids=[1, 2, 3], target_prompt_sha256="fixed"))
    data = [dict(request_id="a", repeat=repeat, method=method, correct=True, token_ids=[1, 2, 3],
                 target_prompt_sha256="fixed", snapshot_id="teacher", rounds=[])
            for repeat in range(3) for method in METHODS]
    validate_rows(data, snapshots, references)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_rows(data[:-1], snapshots, references)
    for field, value, message in (("token_ids", [1, 9, 3], "Output"),
                                  ("target_prompt_sha256", "changed", "prompt"),
                                  ("snapshot_id", "manual", "Snapshot")):
        altered = copy.deepcopy(data)
        altered[1][field] = value
        with pytest.raises(ValueError, match=message):
            validate_rows(altered, snapshots, references)
    altered = copy.deepcopy(data)
    altered[1]["rounds"] = [dict(route="H")]
    with pytest.raises(ValueError, match="H-off"):
        validate_rows(altered, snapshots, references)
    shortened = [dict(row, token_ids=row["token_ids"][:2]) for row in data]
    validate_rows(shortened, snapshots, references, token_limit=2)


def test_prefix_audit_scores_only_selected_h_positions_with_the_effective_budget():
    data = [dict(request_id="a", method="RHD_on", token_ids=[1, 2, 3], target_prompt_sha256="fixed",
                 rounds=[dict(route="AR", committed=1), dict(route="H", proposed=1, accepted=1, committed=2)])]
    prefix = [dict(request_id="a", position=1, target_prompt_sha256="fixed", budget=2,
                   reference_ids=[2, 3], lengths=dict(H_teacher=2, D_native=0))]
    assert same_prefix_audit(data, prefix) == dict(rounds=1, proposed=1, online_h=1, selected_h=1, native_d=0)
    altered = copy.deepcopy(prefix)
    altered[0]["reference_ids"] = [9, 3]
    with pytest.raises(ValueError, match="reference differs"):
        same_prefix_audit(data, altered)
