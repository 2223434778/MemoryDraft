from copy import deepcopy
from dataclasses import asdict

import pytest

from memorydraft.evaluation.reuse_cases import confirmation_projects, development_projects
from memorydraft.memory.store import MemoryEvent, Snapshot
from scripts.run_reuse_pilot import CONFIRMATION_METHODS, METHODS
from scripts.write_reuse_report import audit


def frozen_development(projects=None, methods=METHODS, repeats=1):
    projects = development_projects() if projects is None else projects
    initial, references, rows = [], [], []
    for project in projects:
        event = MemoryEvent(project["id"], "artifact", 1, "past", project_id=project["id"])
        initial.append(asdict(event))
        for turn in range(len(project["currents"])):
            order = 10 + 10 * turn
            label = dict(project_id=project["id"], turn=turn, order=order, text=f"answer-{turn}", token_ids=[3, 4],
                         prompt_sha256=f"prompt-{project['id']}-{turn}",
                         snapshot_id=Snapshot(project["id"], order, (event,)).snapshot_id)
            references.append(label)
            for repeat in range(repeats):
                for method in methods:
                    rows.append(dict(**label, scenario=project["scenario"], method=method, repeat=repeat, correct=True,
                                     budget=project.get("budget", 384), latency_seconds=1.0, rounds=[]))
            event = MemoryEvent(project["id"], "artifact", order + 1, label["text"], project_id=project["id"])
    return projects, initial, references, rows


def test_frozen_snapshots_and_complete_pairing_are_reconstructed():
    values = frozen_development()
    result = audit(*values)
    assert result["all"]["rows"] == 90
    assert result["all"]["projects"] == 12
    assert result["P2_sequential_config"]["rows"] == 45
    with pytest.raises(ValueError, match="Incomplete"):
        audit(*values[:3], values[3][:-1])


@pytest.mark.parametrize("change", ["tokens", "snapshot", "scenario", "duplicate"])
def test_report_rejects_teacher_leakage_or_changed_scenario(change):
    projects, initial, references, rows = deepcopy(frozen_development())
    if change == "tokens":
        rows[0]["token_ids"] = [8]
    elif change == "snapshot":
        rows[0]["snapshot_id"] = "future-current-answer"
    elif change == "scenario":
        rows[0]["scenario"] = "posthoc-winner"
    else:
        rows.append(rows[0])
    with pytest.raises(ValueError):
        audit(projects, initial, references, rows)


def test_confirmation_projects_and_repeats_remain_disjoint_and_paired():
    projects = confirmation_projects()
    assert not {p["id"] for p in projects} & {p["id"] for p in development_projects()}
    assert len({p["past"] for p in projects}) == 4
    assert all(not p["prompt_contains_old_artifact"] for p in projects)
    values = frozen_development(projects, CONFIRMATION_METHODS, 3)
    summary = audit(*values, expected_projects=projects, methods=CONFIRMATION_METHODS, repeats=3)
    assert summary["all"]["rows"] == 96
    broken = deepcopy(values)
    broken[3][0]["repeat"] = 3
    with pytest.raises(ValueError, match="Unpaired"):
        audit(*broken, expected_projects=projects, methods=CONFIRMATION_METHODS, repeats=3)
