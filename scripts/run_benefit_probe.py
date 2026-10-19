"""Frozen cost-aware splicing ablations; cross-project calibration, fixed Target."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

from memorydraft.evaluation.benefit_probe import METHODS, RETRIEVAL, audit, calibration, inputs, policy_data, protected, timing_gate
from memorydraft.evaluation.runtime_migration import check_result, file_hash
from memorydraft.evaluation.splice_probe import input_digest, prefix_acceptance
from memorydraft.memory.store import MemoryStore
from memorydraft.runtime.benchmark import provenance, source_digest
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest
from scripts.diagnose_transport_online import write
from scripts.run_runtime_migration import append
from scripts.run_splice_probe import Recorder


ROOT = Path(__file__).resolve().parents[1]


class BenefitRecorder(Recorder):
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        self.decisions = getattr(kwargs["matcher"], "decisions", [])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("mechanism", "timing"), required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks, events = inputs(ROOT, subset=True)
    offline = json.loads((args.offline / "provenance.json").read_text())
    cost_info = json.loads((args.costs.parent / "provenance.json").read_text())
    costs, samples = json.loads(args.costs.read_text()), calibration(ROOT)
    if (offline["status"] != "complete" or offline["source"] != source_digest()
            or offline["costs_sha256"] != file_hash(args.costs) or cost_info["status"] != "complete"
            or cost_info["source"] != source_digest() or not cost_info["original_unchanged"]
            or offline["original_sha256"] != protected(ROOT)):
        raise ValueError("Cost/profile/source gate failed")
    if args.stage == "timing":
        if args.mechanism is None:
            parser.error("Timing requires a passed mechanism directory")
        prior = json.loads((args.mechanism / "provenance.json").read_text())
        if (prior["status"] != "complete" or not prior["timing_gate"]["passed"]
                or prior["source"] != source_digest() or prior["runner_sha256"] != file_hash(Path(__file__))
                or prior["inputs_sha256"] != input_digest(tasks, events)
                or prior["costs_sha256"] != file_hash(args.costs)):
            raise ValueError("Changed checked implementation/calibration")
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.output.resolve() / "history.db"
    with MemoryStore(database) as store:
        store.append(events)
        if any(store.snapshot(t.request.user_id, t.request.order).snapshot_id != t.snapshot_id for t in tasks):
            raise ValueError("Changed legal snapshot")
    config = replace(RuntimeConfig.load(args.config), db=str(database), retrieval="suffix", benefit={})
    if (config.backend != "dflash" or config.threads != 4 or config.template_options != {"enable_thinking": False}
            or config.reuse != {"max_tokens": 32, "min_match": 6, "cooldown_rounds": 1, "max_pending_tokens": 1024}):
        raise ValueError("Changed validated FP32/eager configuration")
    started = time.perf_counter()
    engine = load_engine(config)
    info = {**provenance(engine), "status": "started", "stage": args.stage, "methods": METHODS,
            "scope": "Eight known development requests; nearest-three-project empirical policy excludes current project",
            "instrumented": args.stage == "mechanism", "load_seconds": time.perf_counter() - started,
            "inputs_sha256": input_digest(tasks, events), "runner_sha256": file_hash(Path(__file__)),
            "protocol_sha256": file_hash(ROOT / "docs/OT_BENEFIT_PROTOCOL.md"), "original_sha256": protected(ROOT),
            "costs_sha256": file_hash(args.costs), "samples": samples}
    write(args.output / "provenance.json", info)
    rows, repeats = [], 1 if args.stage == "mechanism" else 2

    def execute(request, method, project):
        benefit = policy_data(samples, costs, project) if method in METHODS[-3:] else {}
        engine.config = replace(config, retrieval=RETRIEVAL[method], benefit=benefit)
        return engine.generate(request, "R+DFlash")

    try:
        warm = GenerationRequest("__warmup__", "__warmup__", 0, ({"role": "user", "content": "Say hello."},), 16)
        for method in METHODS:
            execute(warm, method, "__warmup__")
        from memorydraft.runtime import engine as runtime_module
        for task in tasks:
            for repeat in range(repeats):
                order = list(METHODS)
                random.Random(f"benefit-v1:{task.request.id}:{repeat}").shuffle(order)
                for method in order:
                    recorder = BenefitRecorder(runtime_module.generate_reuse)
                    if args.stage == "mechanism":
                        with patch.object(runtime_module, "generate_reuse", recorder):
                            result = execute(task.request, method, task.project_id)
                    else:
                        result = execute(task.request, method, task.project_id)
                    check_result(result, task)
                    row = {**asdict(result), "method": method, "repeat": repeat, "correct": True,
                           "project_id": task.project_id, "group": getattr(task, "group", "development")}
                    row.pop("text")
                    if args.stage == "mechanism":
                        for call in recorder.calls:
                            remaining = task.reference["token_ids"][call["generated"]:]
                            call["draft_oracle_accepted"] = prefix_acceptance(call["draft_tokens"], remaining)
                            call["oracle_accepted"] = prefix_acceptance(call["proposal"]["tokens"], remaining) if call["proposal"] else None
                        row.update(calls=recorder.calls, rounds=recorder.rounds, decisions=recorder.decisions)
                        if any(task.project_id in d["projects"] for d in recorder.decisions):
                            raise ValueError("Project calibration leak")
                        row["instrumented_latency_seconds"] = row.pop("latency_seconds")
                        row.pop("ttft_seconds")
                    rows.append(row)
                    append(args.output / "rows.jsonl", row)
            print("completed", task.request.id, flush=True)
        write(args.output / "summary.json", audit(rows, tasks, repeats, instrumented=args.stage == "mechanism"))
        if args.stage == "mechanism":
            info["timing_gate"] = timing_gate(rows)
        info.update(status="complete", rows=len(rows), original_unchanged=info["original_sha256"] == protected(ROOT))
        if not info["original_unchanged"]:
            raise ValueError("Protected evidence changed")
    except Exception as error:
        info.update(status="failed", rows=len(rows), error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        write(args.output / "provenance.json", info)
        engine.close()


if __name__ == "__main__":
    main()
