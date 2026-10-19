from copy import deepcopy
from pathlib import Path

import pytest

from memorydraft.evaluation.matrix_probe import METHODS, IDS, audit, inputs, protected, timing_gate
from memorydraft.evaluation.splice_probe import COUNTERS


ROOT = Path(__file__).resolve().parents[1]


def observations():
    tasks, _ = inputs(ROOT, subset=True)
    rows = []
    for t in tasks:
        for method in METHODS:
            c = dict.fromkeys((*COUNTERS, "matrix_rejections", "matrix_solves"), 0)
            c.update(rounds=3, draft_rounds=2, reuse_rounds=1, draft_forward_rounds=2)
            rows.append(dict(request_id=t.request.id, method=method, repeat=0, project_id=t.project_id,
                             group=getattr(t, "group", "development"), token_ids=t.reference["token_ids"],
                             prompt_sha256=t.reference["prompt_sha256"], snapshot_id=t.snapshot_id,
                             correct=True, instrumented_latency_seconds=1., counters=c))
    return tasks, rows


def test_frozen_subset_and_protected_old_experiments():
    tasks, rows = observations()
    assert {t.request.id for t in tasks} == set(IDS)
    assert len(rows) == 64 and not timing_gate(rows)["passed"]
    summary = audit(rows, tasks, 1, instrumented=True)
    assert "seconds" not in summary["all"]["Splice-AllOT"]
    assert any(name.startswith("results/ot-splice-cooldown/") for name in protected(ROOT))
    changed = deepcopy(rows)
    next(r for r in changed if r["method"] == "Splice-PlanOT")["counters"].update(rounds=2, draft_rounds=1)
    assert timing_gate(changed)["passed"]


@pytest.mark.parametrize("change", ["missing", "tokens", "snapshot", "solves", "forwards", "attribution"])
def test_matrix_audit_rejects_invalid_pairs_and_attribution(change):
    tasks, rows = observations()
    r = next(r for r in rows if r["method"] == "Splice-AllOT")
    if change == "missing":
        rows.pop()
    elif change == "tokens":
        r["token_ids"] = [999]
    elif change == "snapshot":
        r["snapshot_id"] = "changed"
    elif change == "solves":
        r["counters"]["matrix_solves"] = 100
    elif change == "forwards":
        r["counters"]["draft_forward_rounds"] = 0
    else:
        r["counters"]["splice_history_accepted_tokens"] = 100
    with pytest.raises(ValueError):
        audit(rows, tasks, 1, instrumented=True)


def test_common_pool_scoring_is_independent_of_the_recorded_edit_solver():
    from types import SimpleNamespace
    from memorydraft.retrieval.transport import AlignmentConfig
    from scripts.probe_matrix_offline import score_pool

    pytest.importorskip("ot")
    pool = [dict(query=list(range(16)), history=list(range(16)), tokens=tuple(range(16, 48)),
                 prefix_length=4, position=16, source="test-only", cheap=1.)]
    matcher = SimpleNamespace(config=AlignmentConfig(), solver=None)
    candidates, decisions = score_pool(pool, matcher, list(range(16, 31)), list(range(16, 50)))
    assert len(candidates) == 1
    assert all(d["selected"] == 0 and d["accepted"] == 32 for d in decisions.values())
