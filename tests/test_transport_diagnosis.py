from types import SimpleNamespace

import pytest

pytest.importorskip("ot")

from memorydraft.evaluation.transport_diagnosis import (DecodeRecorder, failure_class,
                                                       first_divergence, inspect_calls, proposal_dict)
from memorydraft.memory.store import MemoryEvent, Snapshot
from memorydraft.retrieval.suffix import SuffixArrayIndex
from memorydraft.retrieval.transport import AlignedReuseMatcher
from scripts.diagnose_transport_online import inputs


class Tokens:
    all_special_ids = [999]

    def encode(self, text, add_special_tokens=False):
        return list(map(int, text.split()))

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


@pytest.mark.parametrize("values,expected", [
    ((0, 0, 0, 0, True), "no_long_raw_span"),
    ((12, 3, 0, 0, True), "ineligible_endpoint"),
    ((12, 12, 3, 0, True), "shortlist_miss"),
    ((12, 12, 12, 0, False), "threshold_rejection"),
    ((12, 12, 12, 0, True), "ranking_miss"),
    ((12, 12, 12, 8, True), "long_but_suboptimal"),
    ((12, 12, 12, 12, True), "long_pool_best"),
])
def test_diagnostic_classes(values, expected):
    assert failure_class(*values) == expected


def test_capture_replays_actual_prefix_and_rejects_future_or_tampered_verification():
    source = SuffixArrayIndex(Snapshot("u", 10, (MemoryEvent("u", "old", 1,
                             " ".join(map(str, range(1, 45)))),)), Tokens())
    matcher = AlignedReuseMatcher(source, method="edit")
    prefix = list(range(1, 25))
    prefix[-2] = 199
    original = matcher.propose

    def decode(*args, **kwargs):
        proposal = kwargs["matcher"].propose(prefix, 8)
        return SimpleNamespace(rounds=[dict(generated_before=24, route="R", lookup="edit", accepted=8)])

    recorder = DecodeRecorder(decode)
    recorder(matcher=matcher)
    assert matcher.propose == original
    reference = prefix + list(range(25, 33))
    rows = inspect_calls(recorder, reference, Tokens(), method="edit")
    assert rows[0]["classification"] == "long_pool_best"
    assert rows[0]["selected_accepted"] == rows[0]["pool_best"] == 8
    assert "confirmed" not in rows[0]
    assert proposal_dict(matcher.propose(prefix, 8)) == rows[0]["proposal"]
    with pytest.raises(ValueError, match="Captured prefix"):
        inspect_calls(recorder, [777] + reference[1:], Tokens(), method="edit")
    recorder.rounds[0]["accepted"] = 7
    with pytest.raises(ValueError, match="actual Target verification"):
        inspect_calls(recorder, reference, Tokens(), method="edit")


def test_capture_restores_matcher_on_failure():
    matcher = SimpleNamespace(index=None, propose=lambda *a: None)
    original = matcher.propose

    def fail(**kwargs):
        raise RuntimeError("simulated decoder failure")

    with pytest.raises(RuntimeError):
        DecodeRecorder(fail)(matcher=matcher)
    assert matcher.propose is original


def test_subset_is_frozen_and_uses_original_time_snapshots(tmp_path):
    from pathlib import Path
    from memorydraft.memory.store import MemoryStore

    tasks, events = inputs(Path(__file__).resolve().parents[1])
    assert len(tasks) == 8
    with MemoryStore(tmp_path / "history.db") as store:
        store.append(events)
        for task in tasks:
            assert store.snapshot(task.request.user_id, task.request.order).snapshot_id == task.snapshot_id


def test_comparison_ignores_lookup_label_and_keeps_first_behavior_change():
    row = dict(route="R", generated_before=8, proposed=32, accepted=0, source="old", lookup="edit")
    assert first_divergence([row], [dict(row, lookup="ot")]) is None
    changed = dict(row, lookup="ot", accepted=32)
    assert first_divergence([row], [changed])["same_prefix"]


@pytest.mark.parametrize("change", [None, "tokens", "prefix", "candidate", "rounds", "comparison"])
def test_report_reaudits_real_compact_evidence(tmp_path, change):
    from pathlib import Path
    import json
    import shutil

    from scripts.write_transport_diagnosis_report import read_json, rebuild

    source = Path(__file__).resolve().parents[1] / "results/ot-diagnosis"
    for name in ("provenance.json", "decisions.json.gz", "comparisons.json", "shortlist-analysis.json"):
        shutil.copyfile(source / name, tmp_path / name)
    rows = read_json(tmp_path / "decisions.json")
    aligned_row = next(r for r in rows if r["counters"].get("alignment_calls", 0))
    call = next(c for c in aligned_row["calls"] if c["alignment_called"])
    if change == "tokens":
        rows[0]["token_ids"] = [123]
    elif change == "prefix":
        call["prefix_sha256"] = "changed"
    elif change == "candidate":
        call["candidates"][0]["accepted"] += 1
    elif change == "rounds":
        rows[0]["counters"]["rounds"] += 1
    elif change == "comparison":
        comparisons = json.loads((tmp_path / "comparisons.json").read_text(encoding="utf-8"))
        comparisons.pop()
        (tmp_path / "comparisons.json").write_text(json.dumps(comparisons), encoding="utf-8")
    (tmp_path / "decisions.json").write_text(json.dumps(rows), encoding="utf-8")
    if change is None:
        rebuild(tmp_path, tmp_path / "report.md")
        assert b"\r" not in (tmp_path / "report.md").read_bytes()
    else:
        with pytest.raises(ValueError):
            rebuild(tmp_path, tmp_path / "report.md")
