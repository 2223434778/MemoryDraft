"""Known training-project coverage controls, separate from the 12-request result."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
from unittest.mock import patch

from memorydraft.evaluation.prefix_risk_continuation import protected
from memorydraft.evaluation.prefix_risk_probe import METHODS, RETRIEVAL, audit, audit_decisions
from memorydraft.evaluation.runtime_migration import check_result, file_hash
from memorydraft.evaluation.splice_probe import inputs as old_inputs, input_digest, prefix_acceptance
from memorydraft.evaluation.transport_validation import contract_check, prompt_identity
from memorydraft.memory.store import MemoryStore
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest
from scripts.diagnose_transport_online import write
from scripts.run_prefix_risk_probe import RiskRecorder
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
IDS = ("payment-runbook:1", "shipment-contract:1")


def inputs(root):
    tasks, events = old_inputs(root)
    tasks = [next(t for t in tasks if t.request.id == q) for q in IDS]
    users = {t.request.user_id for t in tasks}
    events = [e for e in events if e.user_id in users]
    calibration = json.loads((root / "data/research/prefix-risk/calibration.json").read_text())
    if not {t.project_id for t in tasks} <= set(calibration["training_projects"]):
        raise ValueError("Coverage controls must be labelled as known training projects")
    return tasks, events, calibration


def primary_hashes(root):
    directory = root / "results/prefix-risk/continuation"
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.relative_to(directory).parts[0] != "controls"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks, events, calibration = inputs(ROOT)
    database = args.output.resolve() / "history.db"
    with MemoryStore(database) as store:
        store.append(events)
        if any(store.snapshot(t.request.user_id, t.request.order).snapshot_id != t.snapshot_id for t in tasks):
            raise ValueError("Changed legal control snapshot")
    config = replace(RuntimeConfig.load(args.config), db=str(database), retrieval="suffix", prefix_risk={})
    engine = load_engine(config)
    hashes, primary = protected(ROOT), primary_hashes(ROOT)
    info = {**provenance(engine), "status": "started", "scope": "Known training-project controls only; selected for mechanism coverage, never generalization evidence",
            "runner_sha256": file_hash(Path(__file__)), "protocol_sha256": file_hash(ROOT / "docs/PREFIX_RISK_CONTROLS_PROTOCOL.md"),
            "calibration_sha256": file_hash(ROOT / "data/research/prefix-risk/calibration.json"),
            "inputs_sha256": input_digest(tasks, events), "original_sha256": hashes, "primary_sha256": primary}
    write(args.output / "provenance.json", info)

    def execute(request, method):
        engine.config = replace(config, retrieval=RETRIEVAL[method], prefix_risk=calibration["models"].get(method, {}))
        return engine.generate(request, "R+DFlash")

    stages = {}
    try:
        for task in tasks:
            identity = prompt_identity(engine.tokenizer, task.request, config.template_options)
            if any(task.reference[k] != identity[k] for k in ("prompt_sha256", "prompt_tokens")):
                raise ValueError("Known control prompt differs from its original reference")
        warm = GenerationRequest("__warmup__", "__warmup__", 0, ({"role": "user", "content": "Say hello."},), 16)
        for method in METHODS:
            execute(warm, method)
        from memorydraft.runtime import engine as runtime_module
        for stage, repeats in (("mechanism", 1), ("timing", 2)):
            directory = args.output / stage
            directory.mkdir()
            rows = []
            for task in tasks:
                for repeat in range(repeats):
                    order = list(METHODS)
                    random.Random(f"risk-controls:{task.request.id}:{repeat}").shuffle(order)
                    for method in order:
                        recorder = RiskRecorder(runtime_module.generate_reuse)
                        if stage == "mechanism":
                            with patch.object(runtime_module, "generate_reuse", recorder):
                                result = execute(task.request, method)
                        else:
                            result = execute(task.request, method)
                        check_result(result, task)
                        row = {**asdict(result), "method": method, "repeat": repeat, "correct": True,
                               "project_id": task.project_id, "group": task.group,
                               "contract": contract_check(result.text, task.expected_text)}
                        row.pop("text")
                        if stage == "mechanism":
                            for call in recorder.calls:
                                remaining = task.reference["token_ids"][call["generated"]:]
                                call["draft_oracle_accepted"] = prefix_acceptance(call["draft_tokens"], remaining)
                                call["oracle_accepted"] = prefix_acceptance(call["proposal"]["tokens"], remaining) if call["proposal"] else None
                            for d in recorder.decisions:
                                remaining = task.reference["token_ids"][d["generated"]:]
                                d["draft_accepted"] = prefix_acceptance(d["draft_tokens"], remaining)
                                for c in d["candidates"]:
                                    c["accepted"] = prefix_acceptance(c["tokens"], remaining)
                            row.update(calls=recorder.calls, decisions=recorder.decisions, rounds=recorder.rounds)
                            row["instrumented_latency_seconds"] = row.pop("latency_seconds")
                            row.pop("ttft_seconds")
                        rows.append(row)
                        append(directory / "rows.jsonl", row)
            write(directory / "summary.json", audit(rows, tasks, repeats, instrumented=stage == "mechanism"))
            if stage == "mechanism":
                audit_decisions(rows, tasks, calibration)
            else:
                expected = {(r["request_id"], r["method"]): r["counters"] for r in stages["mechanism"]}
                if any(v != r["counters"][k] for r in rows for k, v in expected[r["request_id"], r["method"]].items() if k != "alignment_seconds"):
                    raise ValueError("Control timing changed a checked mechanism path")
            stages[stage] = rows
            print("completed", stage, len(rows), flush=True)
        info.update(status="complete", mechanism_rows=len(stages["mechanism"]), timing_rows=len(stages["timing"]))
    except Exception as error:
        info.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        info.update(original_unchanged=hashes == protected(ROOT), primary_unchanged=primary == primary_hashes(ROOT))
        write(args.output / "provenance.json", info)
        engine.close()
    if not info["original_unchanged"] or not info["primary_unchanged"]:
        raise ValueError("Known controls changed original evidence or the 12-request primary result")


if __name__ == "__main__":
    main()
