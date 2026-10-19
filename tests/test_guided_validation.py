from dataclasses import replace
import json
from pathlib import Path

import pytest

from memorydraft.evaluation.guided_validation import METHODS, audit, load_inputs
from memorydraft.memory.store import MemoryStore
from scripts.make_guided_validation import authored_projects


ROOT = Path(__file__).resolve().parents[1]


def test_new_cases_match_generator_and_legal_snapshots(tmp_path):
    path = ROOT / "data/research/ot-guided/cases.json"
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(authored_projects()))
    tasks, events = load_inputs(path)
    assert len(tasks) == 16 and len(events) == 7
    with MemoryStore(tmp_path / "history.db") as store:
        store.append(events)
        assert all(store.snapshot(t.request.user_id, t.request.order).snapshot_id == t.snapshot_id for t in tasks)


@pytest.mark.parametrize("change", [None, "missing", "tokens", "nan", "forward_count"])
def test_paired_audit_requires_complete_correct_finite_accounting(change):
    tasks, _ = load_inputs(ROOT / "data/research/ot-guided/cases.json")
    tasks = [replace(t, reference=dict(token_ids=[1, 2], prompt_sha256="frozen")) for t in tasks]
    rows = [dict(request_id=t.request.id, project_id=t.project_id, group=t.group, method=m, repeat=r,
                 token_ids=[1, 2], prompt_sha256="frozen", snapshot_id=t.snapshot_id, correct=True,
                 latency_seconds=1., counters=dict(rounds=2, draft_rounds=1, reuse_rounds=1,
                                                   reuse_accepted_tokens=1, draft_forward_rounds=2))
            for t in tasks for m in METHODS for r in range(2)]
    if change == "missing":
        rows.pop()
    elif change == "tokens":
        rows[0]["token_ids"] = [3]
    elif change == "nan":
        rows[0]["latency_seconds"] = float("nan")
    elif change == "forward_count":
        next(r for r in rows if r["method"] == "Guided-OT")["counters"]["draft_forward_rounds"] = 0
    if change is None:
        assert audit(rows, tasks, METHODS, 2)["all"]["Guided-OT"]["counters"]["draft_forward_rounds"] == 64
    else:
        with pytest.raises(ValueError):
            audit(rows, tasks, METHODS, 2)
