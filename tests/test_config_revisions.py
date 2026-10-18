from copy import deepcopy
import difflib
import json
from pathlib import Path
import shutil

import pytest

from memorydraft.evaluation.config_revisions import (METHODS_BY_MODE, answer_event, load_projects,
                                                     methods_for_split, quality, request_text, working_event)
from memorydraft.memory.store import MemoryStore, Snapshot
from memorydraft.retrieval.suffix import SuffixArrayIndex
from scripts.prepare_config_revisions import CUTOFF, REPOSITORIES
from scripts.write_config_revisions_report import audit


DATA = Path(__file__).resolve().parents[1] / "data/research/config-revisions"


def test_frozen_repositories_and_every_real_patch():
    payload = json.loads((DATA / "projects.json").read_text(encoding="utf-8"))
    assert payload["cutoff"] == CUTOFF and payload["repositories"] == REPOSITORIES
    for split, size in (("development", 6), ("confirmation", 12)):
        projects = load_projects(DATA, split)
        assert len(projects) == size
        for project in projects:
            for turn in range(2):
                assert quality(project["turns"][turn]["patch"], project, turn, "diff")["patch_applies_to_expected"]
                assert quality(project["turns"][turn]["expected"], project, turn, "full")["exact_file_or_patch"]
                assert project["turns"][turn]["before"] in request_text(project, turn)


def test_user_file_is_explicit_and_future_versions_stay_invisible(tmp_path):
    class Tokenizer:
        all_special_ids = []

        def encode(self, text, add_special_tokens=False):
            return list(map(int, text.split()))

    from memorydraft.memory.store import MemoryEvent

    with MemoryStore(tmp_path / "history.db") as store:
        store.append([MemoryEvent("u", "working-file", 1, "1 2 3 4 5", source_role="user"),
                      MemoryEvent("u", "working-file", 11, "1 2 3 8 9", source_role="user")])
        past = store.snapshot("u", 10)
        assert SuffixArrayIndex(past, Tokenizer(), min_match=2).locate([1, 2, 3], 2) is None
        assert SuffixArrayIndex(past, Tokenizer(), min_match=2, include_user_records=True).locate([1, 2, 3], 2).tokens == (4, 5)
        assert not store.snapshot("other", 20).records


def test_frozen_source_cannot_be_silently_replaced(tmp_path):
    target = tmp_path / "source"
    shutil.copytree(DATA, target)
    path = target / "projects.json"
    path.write_bytes(path.read_bytes().replace(b"astral-sh/ruff", b"astral-sh/fake", 1))
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_projects(target, "development")


def test_patch_quality_accepts_other_context_but_never_other_paths():
    project = load_projects(DATA, "development")[0]
    item = project["turns"][0]
    alternate = "".join(difflib.unified_diff(item["before"].splitlines(keepends=True), item["expected"].splitlines(keepends=True),
                                           fromfile="a/" + project["path"], tofile="b/" + project["path"], n=1))
    assert quality(alternate, project, 0, "diff")["patch_applies_to_expected"]
    assert quality(alternate.rstrip("\n"), project, 0, "diff")["patch_applies_to_expected"]
    assert not quality(alternate.rstrip("\n"), project, 0, "diff", version=1)["patch_applies_to_expected"]
    assert not quality(alternate.replace(project["path"], "../outside.yml"), project, 0, "diff")["patch_applies_to_expected"]
    assert not quality("```yaml\n" + item["expected"] + "```", project, 0, "full")["exact_file_or_patch"]


def fixture_rows():
    project = load_projects(DATA, "development")[0]
    references, rows = [], []
    for mode, methods in METHODS_BY_MODE.items():
        previous = None
        for turn in range(2 if mode == "full" else 1):
            text = project["turns"][turn]["expected" if mode == "full" else "patch"]
            events = [working_event(project, mode, turn)]
            if previous:
                events.append(answer_event(project, mode, turn - 1, previous))
            snapshot = Snapshot(project["id"] + ":" + mode, 10 + 10 * turn,
                                tuple(sorted(events, key=lambda event: event.record_id)))
            reference = dict(project_id=project["id"], mode=mode, turn=turn, order=10+10*turn,
                             text=text, token_ids=[turn, len(text)], prompt_sha256="fixed-"+mode+str(turn),
                             snapshot_id=snapshot.snapshot_id, quality=quality(text, project, turn, mode))
            references.append(reference)
            for method in methods:
                row = dict(reference, method=method, repeat=0, correct=True, latency_seconds=1., write_seconds=.01,
                           rounds=[], budget=3072 if mode == "full" else 768)
                rows.append(row)
            previous = text
    return [project], references, rows


@pytest.mark.parametrize("corruption", ["none", "missing", "duplicate", "tokens", "snapshot", "quality"])
def test_report_rebuilds_past_and_rejects_corruption(corruption):
    projects, references, rows = fixture_rows()
    if corruption == "none":
        assert audit(projects, references, rows, repeats=1)["full"]["rows"] == 12
        return
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = deepcopy(rows[-2])
    elif corruption == "tokens":
        rows[-1]["token_ids"] = [999]
    elif corruption == "snapshot":
        references[1]["snapshot_id"] = "future"
    else:
        references[0]["quality"]["exact_file_or_patch"] = False
    with pytest.raises(ValueError):
        audit(projects, references, rows, repeats=1)


def test_confirmation_keeps_strong_controls_and_independent_ar_labels():
    projects, references, rows = fixture_rows()
    methods = methods_for_split("confirmation")
    assert methods["full"] == ("DFlash", "Adapted", "Global", "Context", "Arctic")
    rows = [r for r in rows if r["method"] != "AR"]
    assert audit(projects, references, rows, repeats=1, methods_by_mode=methods)["full"]["rows"] == 10
