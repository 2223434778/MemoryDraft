from pathlib import Path

from scripts.run_asd_splice_probe import IDS, METHODS, inputs


def test_probe_uses_eight_legal_existing_tasks_including_short_control():
    tasks, events = inputs(Path(__file__).resolve().parents[1])
    assert tuple(t.request.id for t in tasks) == IDS
    assert len(tasks) == 8 and len(METHODS) == 9
    assert len({t.project_id for t in tasks}) == 5
    assert all(t.reference and t.expected_text for t in tasks)
    assert all(e.order < t.request.order for t in tasks for e in events if e.user_id == t.request.user_id)
    assert any(len(t.reference["token_ids"]) < 64 for t in tasks)
