"""Replay eight existing cases once per retrieval arm, then audit actual decisions."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import ot

from memorydraft.evaluation.runtime_migration import check_result, file_hash, input_hashes, read_rows
from memorydraft.evaluation.transport_diagnosis import DecodeRecorder, digest, first_divergence, inspect_calls
from memorydraft.evaluation.transport_probe import probe_inputs
from memorydraft.evaluation.transport_validation import attach_references, load_inputs
from memorydraft.memory.store import MemoryStore
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine


ROOT = Path(__file__).resolve().parents[1]
NEW_IDS = ("parcel-queue:0", "parcel-queue:1", "payment-runbook:0", "payment-runbook:1", "short-settings:1")
OLD_IDS = ("confirm-schema-0:0", "confirm-schema-0:1", "copy")
METHODS = ("Suffix-R", "Edit-R", "OT-R")


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def inputs(root):
    new, new_events = load_inputs(root / "data/research/ot-validation/cases.json")
    new = attach_references(new, read_rows(root / "results/ot-validation/fresh-ar.jsonl"))
    old, old_events = probe_inputs(root)
    tasks = [t for t in new if t.request.id in NEW_IDS] + [t for t in old if t.request.id in OLD_IDS]
    if {t.request.id for t in tasks} != set(NEW_IDS + OLD_IDS) or len(tasks) != 8:
        raise ValueError("Unexpected diagnostic subset")
    events = [e for e in new_events + old_events if e.user_id in {t.request.user_id for t in tasks}]
    return tasks, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks, events = inputs(ROOT)
    protected = ("ot-validation", "ot-r", "reuse-confirmation", "infra-v0.3")
    hashes = {name: input_hashes(ROOT / "results" / name) for name in protected}
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.output.resolve() / "history.db"
    with MemoryStore(database) as store:
        store.append(events)
        for task in tasks:
            if store.snapshot(task.request.user_id, task.request.order).snapshot_id != task.snapshot_id:
                raise ValueError("Diagnostic changed the legal snapshot")
    config = replace(RuntimeConfig.load(args.config), db=str(database), retrieval="suffix", alignment={})
    engine = load_engine(config)
    from memorydraft.runtime import engine as runtime_module

    info = {**provenance(engine), "status": "started", "POT": ot.__version__,
            "scope": "Post-hoc instrumented replay of eight already observed cases, no speed or generalization claim",
            "methods": METHODS, "request_ids": [t.request.id for t in tasks],
            "runner_sha256": file_hash(Path(__file__)), "original_sha256": hashes,
            "protocol_sha256": file_hash(ROOT / "docs/OT_R_DIAGNOSIS_PROTOCOL.md")}
    prior = json.loads((ROOT / "results/ot-validation/provenance.json").read_text(encoding="utf-8"))["source"]["files"]
    if any(info["source"]["files"].get(name) != sha for name, sha in prior.items()):
        raise ValueError("Previously measured package files must remain unchanged")
    info["prior_package_files_unchanged"] = len(prior)
    write(args.output / "provenance.json", info)
    rows = []
    try:
        for task in tasks:
            for method in METHODS:
                engine.config = replace(config, retrieval={"Suffix-R": "suffix", "Edit-R": "edit", "OT-R": "ot"}[method])
                recorder = DecodeRecorder(runtime_module.generate_reuse)
                with patch.object(runtime_module, "generate_reuse", recorder):
                    result = engine.generate(task.request, "R+DFlash")
                check_result(result, task)
                calls = inspect_calls(recorder, task.reference["token_ids"], engine.tokenizer,
                                      method="ot" if method == "OT-R" else "edit")
                expected_calls = result.counters.get("alignment_calls", 0)
                if sum(c["alignment_called"] for c in calls) != expected_calls:
                    raise ValueError("Alignment call accounting differs")
                row = dict(request_id=task.request.id, project_id=task.project_id, method=method,
                           correct=True, prompt_sha256=result.prompt_sha256, snapshot_id=result.snapshot_id,
                           output_sha256=digest(result.token_ids),
                           token_ids=result.token_ids, counters=result.counters, rounds=recorder.rounds, calls=calls)
                rows.append(row)
                write(args.output / "decisions.json", rows)
                print(json.dumps(dict(request=task.request.id, method=method, correct=True,
                                      alignment_calls=expected_calls,
                                      outcomes=[c["classification"] for c in calls if c["alignment_called"]])), flush=True)
        comparisons = []
        for task in tasks:
            keyed = {r["method"]: r for r in rows if r["request_id"] == task.request.id}
            comparisons.append(dict(request_id=task.request.id,
                                    suffix_ot=first_divergence(keyed["Suffix-R"]["rounds"], keyed["OT-R"]["rounds"]),
                                    edit_ot=first_divergence(keyed["Edit-R"]["rounds"], keyed["OT-R"]["rounds"]),
                                    edit_ot_proposals_equal=[c["proposal"]["tokens"] if c["proposal"] else None
                                                             for c in keyed["Edit-R"]["calls"]]
                                    == [c["proposal"]["tokens"] if c["proposal"] else None
                                        for c in keyed["OT-R"]["calls"]]))
        write(args.output / "comparisons.json", comparisons)
        info.update(status="complete", correct_runs=len(rows),
                    original_unchanged=hashes == {name: input_hashes(ROOT / "results" / name) for name in protected})
        if not info["original_unchanged"]:
            raise ValueError("Original evidence changed")
    except Exception as error:
        info.update(status="failed", error=type(error).__name__ + ": " + str(error), correct_runs=len(rows))
        raise
    finally:
        write(args.output / "provenance.json", info)
        engine.close()


if __name__ == "__main__":
    main()
