"""Bounded development replay and frozen new-request validation of draft-guided R."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time

import ot

from memorydraft.evaluation.guided_validation import METHODS, RETRIEVAL, audit, load_inputs
from memorydraft.evaluation.runtime_migration import check_result, file_hash, input_hashes
from memorydraft.evaluation.transport_validation import attach_references, contract_check, prompt_identity
from memorydraft.memory.store import MemoryStore
from memorydraft.runtime.benchmark import provenance, source_digest
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest
from scripts.diagnose_transport_online import inputs as development_inputs, write
from scripts.run_runtime_migration import append


ROOT = Path(__file__).resolve().parents[1]
PROTECTED = ("ot-r", "ot-validation", "ot-diagnosis", "reuse-confirmation", "infra-v0.3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "holdout"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = ROOT / "data/research/ot-guided/cases.json"
    if args.stage == "development":
        tasks, events = development_inputs(ROOT)
        methods, repeats = METHODS[1:], 1
    else:
        tasks, events = load_inputs(cases)
        methods, repeats = METHODS, 2
        if args.development is None:
            parser.error("Holdout requires the completed development directory")
        prior = json.loads((args.development / "provenance.json").read_text(encoding="utf-8"))
        if (prior["status"] != "complete" or prior["stage"] != "development" or prior["rows"] != 32
                or prior["source"] != source_digest() or prior["runner_sha256"] != file_hash(Path(__file__))):
            raise ValueError("Holdout requires the identical successfully checked development implementation")
    hashes = {name: input_hashes(ROOT / "results" / name) for name in PROTECTED}
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.output.resolve() / "history.db"
    with MemoryStore(database) as store:
        store.append(events)
        for task in tasks:
            if store.snapshot(task.request.user_id, task.request.order).snapshot_id != task.snapshot_id:
                raise ValueError("Wrong legal snapshot")
    config = replace(RuntimeConfig.load(args.config), db=str(database), retrieval="suffix", alignment={})
    if (config.backend != "dflash" or config.threads != 4
            or config.template_options != {"enable_thinking": False}
            or config.reuse != {"max_tokens": 32, "min_match": 6, "cooldown_rounds": 1, "max_pending_tokens": 1024}):
        raise ValueError("Use frozen DFlash/Target settings")
    started = time.perf_counter()
    engine = load_engine(config)
    info = {**provenance(engine), "status": "started", "stage": args.stage, "POT": ot.__version__,
            "methods": methods, "repeats": repeats, "load_seconds": time.perf_counter() - started,
            "original_sha256": hashes, "cases_sha256": file_hash(cases),
            "runner_sha256": file_hash(Path(__file__)), "protocol_sha256": file_hash(ROOT / "docs/OT_GUIDED_PROTOCOL.md"),
            "scope": "Authored controlled tasks; development replay separated from previously unseen Target outputs"}
    write(args.output / "provenance.json", info)

    def execute(request, method):
        engine.config = replace(config, retrieval=RETRIEVAL[method])
        return engine.generate(request, "DFlash" if method == "DFlash" else "R+DFlash")

    rows, references = [], []
    try:
        warm = GenerationRequest("__warmup__", "__warmup__", 0, ({"role": "user", "content": "Say hello."},), 16)
        for method in methods:
            execute(warm, method)
        if args.stage == "holdout":
            identities = {t.request.id: prompt_identity(engine.tokenizer, t.request, config.template_options) for t in tasks}
            for task in tasks:
                result = engine.generate(task.request, "AR")
                identity = identities[task.request.id]
                if (result.prompt_sha256 != identity["prompt_sha256"] or result.prompt_tokens != identity["prompt_tokens"]
                        or result.snapshot_id != task.snapshot_id):
                    raise ValueError("AR input differs from the frozen input")
                row = {**asdict(result), "contract": contract_check(result.text, task.expected_text)}
                references.append(row)
                append(args.output / "fresh-ar.jsonl", row)
                print("AR", task.request.id, row["contract"], result.finish_reason, flush=True)
            tasks = attach_references(tasks, references)
            info["fresh_ar_correct"] = len(tasks)
            if any(not r["contract"]["artifact_correct"] or r["finish_reason"] != "eos" for r in references):
                info.update(status="stopped_target_contract", rows=0,
                            error="Frozen Target failed an artifact contract or exhausted the token budget")
                return
        for task in tasks:
            for repeat in range(repeats):
                order = list(methods)
                random.Random(f"guided-v1:{task.request.id}:{repeat}").shuffle(order)
                for method in order:
                    result = execute(task.request, method)
                    check_result(result, task)
                    row = {**asdict(result), "method": method, "repeat": repeat, "correct": True,
                           "project_id": task.project_id, "group": getattr(task, "group", "development")}
                    if hasattr(task, "expected_text"):
                        row["contract"] = contract_check(result.text, task.expected_text)
                    row.pop("text")
                    rows.append(row)
                    append(args.output / "rows.jsonl", row)
            print("completed", task.request.id, flush=True)
        write(args.output / "summary.json", audit(rows, tasks, methods, repeats))
        info.update(status="complete", rows=len(rows),
                    original_unchanged=hashes == {name: input_hashes(ROOT / "results" / name) for name in PROTECTED})
        if not info["original_unchanged"] or info["cases_sha256"] != file_hash(cases):
            raise ValueError("Old evidence or frozen cases changed")
    except Exception as error:
        info.update(status="failed", rows=len(rows), error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        engine.close()
        write(args.output / "provenance.json", info)


if __name__ == "__main__":
    main()
