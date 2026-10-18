from copy import deepcopy
import json
from pathlib import Path

import pytest

from memorydraft.evaluation.release_triage import (AREAS, answer_event, continuation_map, load_dataset,
    opportunity, oracle_coverage, quality, quality_gate, request_text)
from memorydraft.memory.store import MemoryStore
from scripts.prepare_release_triage import section
from scripts.run_reuse_pilot import digest
from scripts.write_release_triage_report import audit
from memorydraft.evaluation.release_triage import sha
from memorydraft.memory.store import Snapshot


DATA = Path(__file__).resolve().parents[1] / "data/research/release-triage"


def answer(release, gold):
    return dict(version=release["version"], checks=[dict(area=area, status="review" if area in gold else "not_mentioned",
                evidence=gold[area]["lines"][:1] if area in gold else []) for area in AREAS])


def test_all_18_frozen_notes_and_annotations_are_valid_and_not_prompts():
    projects, labels, _ = load_dataset(DATA)
    assert sum(len(p["releases"]) for p in projects) == 18
    for p in projects:
        for turn, release in enumerate(p["releases"]):
            gold = labels[p["id"] + ":" + release["tag"]]
            assert quality(json.dumps(answer(release, gold)), release, gold)["correct"]
            text = request_text(p, turn)
            poisoned = deepcopy(p)
            poisoned["gold"] = "DO NOT LEAK THIS TEST LABEL"
            assert request_text(poisoned, turn) == text
            assert all(other["notes"] not in text for other in p["releases"] if other is not release)
            assert "[L001] " + release["notes"].splitlines()[0] in text


@pytest.mark.parametrize("mutation", ["missing_positive", "unsupported_evidence", "other_release_line", "wrong_version", "unknown_field"])
def test_quality_rejects_wrong_or_ungrounded_answers(mutation):
    p, labels, _ = load_dataset(DATA)
    project, release = p[0], p[0]["releases"][1]
    gold = labels[project["id"] + ":" + release["tag"]]
    value = answer(release, gold)
    if mutation == "missing_positive":
        value["checks"][0].update(status="not_mentioned", evidence=[])
    elif mutation == "unsupported_evidence":
        value["checks"][0]["evidence"] = ["L005"]  # CVE, not PyPy support.
    elif mutation == "other_release_line":
        value["checks"][0]["evidence"] = ["L099"]
    elif mutation == "wrong_version":
        value["version"] = "2.32.3"
    else:
        value["comment"] = "Safe to upgrade"
    assert not quality(json.dumps(value), release, gold)["correct"]


def test_duplicate_json_keys_and_fences_are_not_silently_accepted():
    project = load_dataset(DATA)[0][0]
    release = project["releases"][0]
    assert not quality('{"version":"wrong","version":"2.32.3","checks":[]}', release, {})["schema"]
    assert not quality('```json\n{}\n```', release, {})["schema"]


def test_history_is_project_and_time_isolated_and_keeps_both_past_answers(tmp_path):
    projects, _, _ = load_dataset(DATA)
    with MemoryStore(tmp_path / "history.db") as store:
        store.append([answer_event(projects[0], 0, "past"), answer_event(projects[0], 1, "later"),
                      answer_event(projects[1], 0, "other project")])
        assert [r.text for r in store.snapshot(projects[0]["id"], 20).records] == ["past"]
        assert {r.text for r in store.snapshot(projects[0]["id"], 30).records} == {"past", "later"}


class Tokenizer:
    all_special_ids = [99]

    def encode(self, text, add_special_tokens=False):
        return list(map(int, text.split()))


def test_oracle_measures_extra_resources_and_never_crosses_records_or_special_tokens():
    tokens = list(range(1, 21))
    history = " ".join(map(str, tokens))
    result = opportunity(tokens, "1 2 3 4 5 6 80", [history], Tokenizer())
    assert result["history_only_covered"] == 14 and result["context_covered"] == 0
    assert result["oracle"] and not result["measured_speedup"]
    assert opportunity(tokens, history, [history], Tokenizer())["history_only_covered"] == 0
    for texts in [["1 2 3 4 5 6", "7 8 9 10 11 12 13 14"], ["1 2 3 4 5 6 99 7 8 9 10 11 12 13 14"]]:
        mapping = continuation_map(texts, Tokenizer())
        assert not mapping.get((1, 2, 3, 4, 5, 6))
    covered, _ = oracle_coverage(tokens, continuation_map(["1 2 3 4 5 6 7 8"], Tokenizer()))
    assert not covered  # Two correct proposed tokens do not pass the minimum-eight opportunity rule.


def test_quality_gate_stops_on_critical_error_even_with_17_correct_requests():
    good = dict(schema=True, identity=True, correct=True, areas={a: True for a in AREAS}, critical_misses=[])
    rows = [dict(project_id=str(i // 3), turn=i % 3, quality=deepcopy(good), budget_hit=False, eos=True) for i in range(18)]
    assert quality_gate(rows)["passed"]
    rows[0]["quality"].update(correct=False, critical_misses=["security_fix"])
    rows[0]["quality"]["areas"]["security_fix"] = False
    assert not quality_gate(rows)["passed"]


def test_extract_only_the_named_release_without_past_or_future_sections():
    text = "## 2.0.0\nnew\n## 1.0.0\nold\n"
    assert section(text, "v2.0.0") == "## 2.0.0\nnew\n"
    with pytest.raises(ValueError):
        section(text, "3.0.0")


def test_audit_rejects_future_answer_or_cross_project_history():
    projects, _, _ = load_dataset(DATA)
    rows = []
    for project in projects:
        for turn in range(3):
            past = sorted([answer_event(project, t, f"answer-{t}") for t in range(turn)], key=lambda r: r.record_id)
            snapshot = Snapshot(project["id"], 10 + turn * 10, tuple(past))
            rows.append(dict(project_id=project["id"], turn=turn, order=snapshot.before_order, method="AR",
                text=f"answer-{turn}", token_ids=[1, 2], eos=True, budget=768, budget_hit=False,
                prompt_token_ids=[3], input_tokens=1, prompt_sha256=digest([3]),
                request_sha256=sha(request_text(project, turn).encode()), snapshot_id=snapshot.snapshot_id,
                history_records=[dict(record_id=r.record_id, order=r.order, text_sha256=sha(r.text.encode())) for r in past]))
    audit(projects, rows, [2])
    rows[1]["history_records"].append(dict(record_id="future", order=31, text_sha256=sha(b"future answer")))
    with pytest.raises(ValueError, match="legal history"):
        audit(projects, rows, [2])


def test_frozen_label_change_is_detected(tmp_path):
    import shutil

    shutil.copytree(DATA, tmp_path / "data")
    path = tmp_path / "data/labels.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Frozen data changed"):
        load_dataset(tmp_path / "data")
