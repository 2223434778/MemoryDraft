import copy

import pytest

from scripts.audit_pilot import audit


def paired_rows():
    base = dict(request_id="r", repeat=0, correct=True, token_ids=[1, 2, 3],
                snapshot_id="s", history_condition="relevant", latency_seconds=3.0)
    return [dict(base, method="AR", rounds=[]), dict(base, method="R", rounds=[
        dict(route="R", proposed=2, accepted=1, committed=2),
        dict(route="AR", proposed=0, accepted=0, committed=1),
    ])]


def test_retrieval_acceptance_is_distinct_from_coverage():
    result = audit(paired_rows())["R"]
    assert result["acceptance_rate"] == 0.5
    assert result["r_proposal_round_fraction"] == 0.5
    assert result["r_accepted_output_fraction"] == 1 / 3


@pytest.mark.parametrize("field,value", [("token_ids", [9, 2, 3]), ("snapshot_id", "future")])
def test_audit_rejects_false_paired_evidence(field, value):
    rows = copy.deepcopy(paired_rows())
    rows[1][field] = value
    with pytest.raises(ValueError, match="mismatch"):
        audit(rows)
