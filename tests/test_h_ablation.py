import json
from types import SimpleNamespace

import pytest

from memorydraft.evaluation.history_alignment import compare_history_sources, freeze_teacher_history, past_requests
from memorydraft.evaluation.replay import read_jsonl
from memorydraft.memory.store import MemoryEvent, MemoryStore, Snapshot
from memorydraft.routing.policy import Router
from tests.test_backend import backend


def test_past_task_construction_ignores_future_versions_and_uses_only_old_facts(tmp_path):
    old = MemoryEvent("u", "u-old", 1, "def read_items(items, limit=10):\n    return [item.strip().lower() for item in items][:limit]\n")
    future = MemoryEvent("u", "u-old", 20, "future current answer uses upper()")
    events = [old, future, MemoryEvent("u", "distractor", 2, "unrelated past text")]
    tasks = past_requests(events)
    assert len(tasks) == 1 and tasks[0]["order"] == 0 and tasks[0]["answer_order"] == 1
    assert "lower()" in tasks[0]["messages"][-1]["content"]
    assert "upper()" not in repr(tasks)

    class PastBackend:
        def generate(self, messages, snapshot, **kwargs):
            assert snapshot.before_order == 0 and not snapshot.records
            assert "lower()" in messages[-1]["content"]
            return SimpleNamespace(text="past Target answer", token_ids=[7], target_prompt_sha256="old", latency_seconds=.1)

    output = tmp_path / "teacher"
    freeze_teacher_history(PastBackend(), events, output)
    with MemoryStore(tmp_path / "memory.db") as store:
        store.import_jsonl(output / "history.jsonl")
        assert {r.record_id: r.text for r in store.snapshot("u", 10).records} == {
            "u-old": "past Target answer", "distractor": "unrelated past text"}
        assert {r.record_id: r.text for r in store.snapshot("u", 21).records}["u-old"] == future.text
        assert store.snapshot("other", 10).records == ()


def test_matched_router_h_switch_keeps_d_ar_cost_logic_and_r_cooldown():
    on = Router("RHD", True)
    off = Router("RHD", True, enable_history=False)
    for _ in range(2):
        assert on.choose(False) == "H"
        on.update("H", 2, 0, 1, 1.)
    for route in ("D", "D", "AR", "AR"):
        assert on.choose(False) == off.choose(False) == route
        for router in (on, off):
            router.update(route, 2 if route == "D" else 0, 0, 1, .2 if route == "D" else .01)
    assert on.choose(False) == off.choose(False) == "AR"
    for router in (on, off):
        assert router.choose(True) == "R"
        router.update("R", 2, 0, 1, .1)
        assert router.choose(True) != "R"


def test_h_off_skips_semantic_retrieval_and_cannot_activate_h(backend, monkeypatch):
    snapshot = Snapshot("u", 10, (MemoryEvent("u", "old", 1, "w3 w4 w5 w6 w7 w8"),))
    messages = [dict(role="user", content="w3 w4")]
    reference = backend.generate(messages, snapshot, method="AR", max_new_tokens=13)

    def fail(*args, **kwargs):
        raise AssertionError("H-off must not run H's semantic retrieval")

    monkeypatch.setattr("memorydraft.backends.transformers.retrieve", fail)
    result = backend.generate(messages, snapshot, method="RHD", enable_history=False, max_new_tokens=13, block_size=2)
    assert result.token_ids == reference.token_ids
    assert not any(e["route"] == "H" for e in result.rounds)
    assert result.target_prompt_sha256 == reference.target_prompt_sha256
    with pytest.raises(ValueError, match="independent"):
        backend.generate(messages, snapshot, method="RHD", enable_history=False, r_scope="shared_pool")


def test_history_source_diagnostic_matches_empty_and_same_history_controls(backend, tmp_path):
    request = dict(id="r", user_id="u", order=10, scenario="controlled", selected_history_ids=["old"],
                   messages=[dict(role="user", content="w3 w4")])
    with MemoryStore(tmp_path / "manual.db") as manual, MemoryStore(tmp_path / "teacher.db") as teacher:
        events = [MemoryEvent("u", "old", 1, "w17 w18 w19")]
        manual.append(events)
        teacher.append(events + [MemoryEvent("other", "old", 1, "w21"), MemoryEvent("u", "future", 20, "w22")])
        summary = compare_history_sources(backend, [request], manual, teacher, tmp_path / "prefix.jsonl", max_new_tokens=7)
    rows = read_jsonl(tmp_path / "prefix.jsonl")
    assert rows and summary["teacher_minus_manual"]["mean_delta"] == 0
    for row in rows:
        assert row["proposal_ids"]["H_manual"] == row["proposal_ids"]["H_teacher"]
        assert row["proposal_ids"]["H_empty"] == row["proposal_ids"]["D_format"]
        assert row["history_payload_tokens"]["H_manual_matched"] == row["history_payload_tokens"]["H_teacher_matched"]
        assert row["sources"] == {"H_manual": ["old"], "H_teacher": ["old"]}


def test_causal_report_weights_projects_and_pairs_router_off_reference():
    from memorydraft.evaluation.history_alignment import paired_delta
    from scripts.write_h_causal_report import relative_time_interval
    quality = [dict(request_id="a", lengths=dict(teacher=2, manual=0), r_match_length=0)]
    quality += [dict(request_id="b", lengths=dict(teacher=0, manual=1), r_match_length=0)] * 3
    assert paired_delta(quality, "teacher", "manual")["mean_delta"] == .5
    data = [dict(request_id="a", repeat=i, method=m, correct=True, token_ids=[1],
                 target_prompt_sha256="fixed", snapshot_id="same", latency_seconds=seconds)
            for i in range(3) for m, seconds in (("off", 2.), ("on", 1.))]
    assert relative_time_interval(data, "off", "on") == (2., 2.)
