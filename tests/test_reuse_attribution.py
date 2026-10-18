from copy import deepcopy
import json

import pytest

from memorydraft.evaluation.reuse_attribution import (EXTRA_TOKEN_BUDGET, candidates, coverage,
                                                       legal_snapshots, parse_request, resource_sets)
from memorydraft.evaluation.reuse_cases import confirmation_projects
from memorydraft.memory.store import Snapshot
from scripts.write_reuse_confirmation_report import expected_config
from tests.test_reuse_report import frozen_development


def test_candidates_use_only_request_facts_and_preserve_explicit_field_order():
    for project in confirmation_projects():
        for turn, request in enumerate(project["currents"]):
            generated = candidates(request)
            facts = parse_request(request)
            compiled = json.loads(generated["Compiled"])
            assert compiled == expected_config(project, turn)
            assert list(compiled) == ["project", facts.collection]
            assert all(list(entry) == list(facts.fields) for entry in compiled[facts.collection])
            for kind in ("Structure", "Static"):
                example = json.loads(generated[kind])
                assert example["project"] == "example"
                assert all(entry[facts.fields[0]].startswith("item_") for entry in example[facts.collection])
                assert facts.project not in generated[kind]
            static = json.loads(generated["Static"])
            assert all(entry["timeout_seconds"] == facts.defaults["timeout_seconds"] for entry in static[facts.collection])
            expanded = json.loads(generated["Expanded"])
            assert [entry[facts.fields[0]] for entry in expanded[facts.collection]] == list(facts.names)
            assert all(entry["timeout_seconds"] == facts.defaults["timeout_seconds"] for entry in expanded[facts.collection])
            assert expanded != compiled
        assert candidates(project["currents"][0])["Static"] == candidates(project["currents"][1])["Static"]
        assert candidates(project["currents"][0])["Expanded"] == candidates(project["currents"][1])["Expanded"]
        assert candidates(project["currents"][0])["Compiled"] != candidates(project["currents"][1])["Compiled"]


@pytest.mark.parametrize("change", ["extra", "unknown_override", "missing_default"])
def test_compiler_rejects_unhandled_instructions_or_inconsistent_facts(change):
    request = confirmation_projects()[0]["currents"][0]
    if change == "extra":
        request += " Also change region."
    elif change == "unknown_override":
        request = request.replace("timeout_seconds for worker_02", "timeout_seconds for worker_99")
    else:
        request = request.replace(', "region": "east"', '')
    with pytest.raises(ValueError):
        candidates(request)


def test_history_reconstruction_uses_latest_artifact_and_rejects_future_or_other_project():
    projects, initial, refs, _ = frozen_development(confirmation_projects())
    snapshots = legal_snapshots(projects, initial, refs)
    identity = projects[0]["id"]
    assert snapshots[identity, 0].records[0].text == "past"
    assert snapshots[identity, 1].records[0].text == "answer-0"
    assert snapshots[identity, 1].records[0].order == 11
    for field, value in (("order", 10), ("project_id", "other"), ("user_id", "other")):
        broken = deepcopy(initial)
        broken[0][field] = value
        with pytest.raises(ValueError):
            legal_snapshots(projects, broken, refs)


def test_schema_and_compiler_resources_are_independent_of_past_answer_text():
    project = confirmation_projects()[0]
    empty = Snapshot(project["id"], 10, ())
    resources = resource_sets(project["currents"][0], empty)
    assert resources["Global"] == resources["Context"] == []
    assert resources["StaticGlobal"] == resources["Static"]
    assert resources["Compiled"] == [candidates(project["currents"][0])["Compiled"]]


def test_coverage_keeps_special_boundaries_and_rejects_silent_budget_truncation():
    class Tokenizer:
        all_special_ids = [0]

        def encode(self, text, add_special_tokens=False):
            return [int(token) for token in text.split()]

    tokens = list(range(1, 16))
    full = " ".join(map(str, tokens))
    broken = "1 2 3 4 5 0 " + " ".join(map(str, tokens[5:]))
    resources = {name: [full] for name in ("Context", "Structure", "Static", "Global", "Expanded", "Compiled")}
    resources["Structure"] = [broken]
    measured, positions = coverage(tokens, "", resources, Tokenizer())
    assert measured["Structure"]["covered"] == 0
    assert measured["Compiled"]["history_outside"] == 0
    assert positions["Global"] == set(range(6, 15))
    resources["Static"] = ["1 " * (EXTRA_TOKEN_BUDGET + 1)]
    with pytest.raises(ValueError, match="budget"):
        coverage(tokens, "", resources, Tokenizer())
