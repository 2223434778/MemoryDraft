"""One frozen independent-project check of prefix-risk-assisted splicing."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import shutil
import time
from unittest.mock import patch

from memorydraft.evaluation.prefix_risk_probe import METHODS, RETRIEVAL, audit, audit_decisions, inputs, protected, timing_gate
from memorydraft.evaluation.runtime_migration import check_result, file_hash, read_rows
from memorydraft.evaluation.splice_probe import prefix_acceptance
from memorydraft.evaluation.transport_validation import attach_references, contract_check, prompt_identity
from memorydraft.memory.store import MemoryStore
from memorydraft.runtime.benchmark import provenance, source_digest
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest
from scripts.diagnose_transport_online import write
from scripts.run_runtime_migration import append
from scripts.run_splice_probe import Recorder

ROOT = Path(__file__).resolve().parents[1]


class RiskRecorder(Recorder):
    def __call__(self, *args, **kwargs):
        matcher = kwargs["matcher"]
        if hasattr(matcher, "record_decisions"):
            matcher.record_decisions = True
        result = super().__call__(*args, **kwargs)
        self.decisions = getattr(matcher, "decisions", [])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("mechanism", "timing"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continue-evaluation", action="store_true",
                        help="User-authorized continuation: all old references, task-quality labels do not stop decoding checks")
    args = parser.parse_args()
    tasks, events, calibration = inputs(ROOT)
    evidence_hashes = protected
    if args.continue_evaluation:
        from memorydraft.evaluation.prefix_risk_continuation import protected as continuation_protected
        evidence_hashes = continuation_protected
    protocol = "docs/PREFIX_RISK_CONTINUATION_PROTOCOL.md" if args.continue_evaluation else "docs/PREFIX_RISK_PROTOCOL.md"
    frozen = {name: file_hash(ROOT / name) for name in (
        "data/research/prefix-risk/cases.json", "data/research/prefix-risk/calibration.json", protocol)}
    if args.stage == "timing":
        if args.mechanism is None:
            parser.error("Timing needs a completed independent mechanism check")
        prior = json.loads((args.mechanism / "provenance.json").read_text())
        if (prior["status"] != "complete" or prior["stage"] != "mechanism" or prior["rows"] != 60
                or not prior["timing_gate"]["passed"] or prior["source"] != source_digest()
                or prior.get("continued_evaluation", False) != args.continue_evaluation
                or prior["frozen_sha256"] != frozen or prior["runner_sha256"] != file_hash(Path(__file__))):
            raise ValueError("Timing requires the identical checked implementation and actual risk switches")
        tasks = attach_references(tasks, read_rows(args.mechanism / "fresh-ar.jsonl"))
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.output.resolve() / "history.db"
    with MemoryStore(database) as store:
        store.append(events)
        if any(store.snapshot(t.request.user_id, t.request.order).snapshot_id != t.snapshot_id for t in tasks):
            raise ValueError("Changed legal memory snapshot")
    config = replace(RuntimeConfig.load(args.config), db=str(database), retrieval="suffix", prefix_risk={})
    if (config.backend != "dflash" or config.threads != 4 or config.template_options != {"enable_thinking": False}
            or config.reuse != {"max_tokens": 32, "min_match": 6, "cooldown_rounds": 1, "max_pending_tokens": 1024}):
        raise ValueError("Use the fixed FP32/eager DFlash profile")
    hashes = evidence_hashes(ROOT)
    started = time.perf_counter()
    engine = load_engine(config)
    info = {**provenance(engine), "status": "started", "stage": args.stage, "methods": METHODS,
            "scope": ("Observed-project continuation; all 12 references including task failures; no parameter tuning or fresh holdout"
                      if args.continue_evaluation else "New authored projects disjoint from old calibration; controlled workload, not production or history-necessity evidence"),
            "continued_evaluation": args.continue_evaluation,
            "load_seconds": time.perf_counter() - started, "frozen_sha256": frozen,
            "runner_sha256": file_hash(Path(__file__)), "original_sha256": hashes}
    write(args.output / "provenance.json", info)
    rows, references = [], []
    repeats = 1 if args.stage == "mechanism" else 2

    def execute(request, method):
        engine.config = replace(config, retrieval=RETRIEVAL[method], prefix_risk=calibration["models"].get(method, {}))
        return engine.generate(request, "R+DFlash")

    try:
        warm = GenerationRequest("__warmup__", "__warmup__", 0, ({"role": "user", "content": "Say hello."},), 16)
        for method in METHODS:
            execute(warm, method)
        if args.stage == "mechanism" and args.continue_evaluation:
            from memorydraft.evaluation.prefix_risk_continuation import check_reference_profile, reference_source
            tasks, metadata, previous_profile = reference_source(ROOT, tasks)
            check_reference_profile(previous_profile, config.as_dict())
            for task in tasks:
                identity = prompt_identity(engine.tokenizer, task.request, config.template_options)
                if any(task.reference[k] != identity[k] for k in ("prompt_sha256", "prompt_tokens")):
                    raise ValueError("Continuation prompt differs from the recorded AR reference")
            shutil.copyfile(ROOT / "results/prefix-risk/mechanism/fresh-ar.jsonl.gz", args.output / "fresh-ar.jsonl.gz")
            info["reference_source"] = metadata
            print("Reused all", len(tasks), "AR references; task contracts correct", metadata["contract_correct"], flush=True)
        elif args.stage == "mechanism":
            for task in tasks:
                identity = prompt_identity(engine.tokenizer, task.request, config.template_options)
                result = engine.generate(task.request, "AR")
                if (result.prompt_sha256 != identity["prompt_sha256"] or result.prompt_tokens != identity["prompt_tokens"]
                        or result.snapshot_id != task.snapshot_id):
                    raise ValueError("AR reference differs from frozen input")
                row = {**asdict(result), "contract": contract_check(result.text, task.expected_text)}
                references.append(row)
                append(args.output / "fresh-ar.jsonl", row)
                print("AR", task.request.id, row["contract"], result.finish_reason, flush=True)
            if any(not r["contract"]["artifact_correct"] or r["finish_reason"] != "eos" for r in references):
                info.update(status="stopped_target_contract", rows=0, references=len(references),
                            error="New Target reference failed a frozen artifact contract; no policy timing")
                return
            tasks = attach_references(tasks, references)
        from memorydraft.runtime import engine as runtime_module
        for task in tasks:
            for repeat in range(repeats):
                order = list(METHODS)
                random.Random(f"prefix-risk-v1:{task.request.id}:{repeat}").shuffle(order)
                for method in order:
                    recorder = RiskRecorder(runtime_module.generate_reuse)
                    if args.stage == "mechanism":
                        with patch.object(runtime_module, "generate_reuse", recorder):
                            result = execute(task.request, method)
                    else:
                        result = execute(task.request, method)
                    check_result(result, task)
                    row = {**asdict(result), "method": method, "repeat": repeat, "correct": True,
                           "project_id": task.project_id, "group": task.group,
                           "contract": contract_check(result.text, task.expected_text)}
                    row.pop("text")
                    if args.stage == "mechanism":
                        for call in recorder.calls:
                            remaining = task.reference["token_ids"][call["generated"]:]
                            call["draft_oracle_accepted"] = prefix_acceptance(call["draft_tokens"], remaining)
                            call["oracle_accepted"] = prefix_acceptance(call["proposal"]["tokens"], remaining) if call["proposal"] else None
                        for decision in recorder.decisions:
                            remaining = task.reference["token_ids"][decision["generated"]:]
                            decision["draft_accepted"] = prefix_acceptance(decision["draft_tokens"], remaining)
                            for c in decision["candidates"]:
                                c["accepted"] = prefix_acceptance(c["tokens"], remaining)
                        row.update(calls=recorder.calls, decisions=recorder.decisions, rounds=recorder.rounds)
                        row["instrumented_latency_seconds"] = row.pop("latency_seconds")
                        row.pop("ttft_seconds")
                    rows.append(row)
                    append(args.output / "rows.jsonl", row)
            print("completed", task.request.id, flush=True)
        write(args.output / "summary.json", audit(rows, tasks, repeats, instrumented=args.stage == "mechanism"))
        if args.stage == "mechanism":
            audit_decisions(rows, tasks, calibration)
            info["timing_gate"] = timing_gate(rows)
            if args.continue_evaluation:
                info["timing_gate"].update(passed=True, policy="Measure all completed strict pairs, including unchanged paths")
        else:
            expected = {(r["request_id"], r["method"]): r["counters"] for r in read_rows(args.mechanism / "rows.jsonl")}
            for row in rows:
                if any(value != row["counters"][key] for key, value in expected[row["request_id"], row["method"]].items()
                       if key != "alignment_seconds"):
                    raise ValueError("Timing changed the checked mechanism path")
            info["paired_paths"] = len(rows)
        info.update(status="complete", rows=len(rows), original_unchanged=hashes == evidence_hashes(ROOT))
        if not info["original_unchanged"] or frozen != {name: file_hash(ROOT / name) for name in frozen}:
            raise ValueError("Protected evidence or frozen protocol changed")
    except Exception as error:
        info.update(status="failed", rows=len(rows), error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        info["original_unchanged"] = hashes == evidence_hashes(ROOT)
        write(args.output / "provenance.json", info)
        engine.close()


if __name__ == "__main__":
    main()
