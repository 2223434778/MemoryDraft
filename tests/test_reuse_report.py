from copy import deepcopy
from dataclasses import asdict

import pytest

from memorydraft.evaluation.reuse_cases import development_projects
from memorydraft.memory.store import MemoryEvent, Snapshot
from scripts.run_reuse_pilot import METHODS
from scripts.write_reuse_report import audit


def frozen_development():
    projects = development_projects()
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
            for method in METHODS:
                rows.append(dict(**label, scenario=project["scenario"], method=method, repeat=0, correct=True,
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
