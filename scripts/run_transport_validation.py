"""Frozen four-arm validation on 24 new authored controlled requests."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time

import ot

from memorydraft.evaluation.runtime_migration import file_hash, input_hashes
from memorydraft.evaluation.transport_probe import METHODS, REPEATS
from memorydraft.evaluation.transport_validation import (attach_references, audit_rows, check_generation,
                                                       comparisons, contract_check, load_inputs, prompt_identity)
from memorydraft.memory.store import MemoryStore
from memorydraft.retrieval.transport import AlignmentConfig
from memorydraft.runtime.benchmark import provenance, write_json
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest
from scripts.run_runtime_migration import append


ROOT = Path(__file__).resolve().parents[1]
PROTECTED = ("ot-r", "reuse-confirmation", "infra-v0.3", "runtime-migration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/research/ot-validation/cases.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks, events = load_inputs(args.cases)
    original = {name: input_hashes(ROOT / "results" / name) for name in PROTECTED}
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.output.resolve() / "history.db"
    with MemoryStore(database) as store:
        store.append(events)
        for task in tasks:
            if store.snapshot(task.request.user_id, task.request.order).snapshot_id != task.snapshot_id:
                raise ValueError("Wrong legal history snapshot")
    config = replace(RuntimeConfig.load(args.config), db=str(database), retrieval="suffix", alignment={})
    if (config.backend != "dflash" or config.template_options != {"enable_thinking": False}
            or config.reuse != {"max_tokens": 32, "min_match": 6, "cooldown_rounds": 1, "max_pending_tokens": 1024}):
        raise ValueError("Use the frozen DFlash settings")
    started = time.perf_counter()
    engine = load_engine(config)
    identities = {t.request.id: {**prompt_identity(engine.tokenizer, t.request, config.template_options),
                                "snapshot_id": t.snapshot_id} for t in tasks}
    info = {**provenance(engine), "status": "started", "methods": METHODS, "repeats": REPEATS,
            "load_and_identity_seconds": time.perf_counter() - started, "POT": ot.__version__,
            "original_sha256": original, "cases_sha256": file_hash(args.cases),
            "runner_sha256": file_hash(Path(__file__)),
            "protocol_sha256": file_hash(ROOT / "docs/OT_R_VALIDATION_PROTOCOL.md"),
            "alignment": asdict(AlignmentConfig()), "identities": identities,
            "fixture_origin": "new authored controlled projects; not natural production traffic",
            "config_per_arm": {m: asdict(replace(config, retrieval={"Edit-R": "edit", "OT-R": "ot"}.get(m, "suffix"))) for m in METHODS}}
    prior = json.loads((ROOT / "results/ot-r/provenance.json").read_text(encoding="utf-8"))["source"]["files"]
    if any(info["source"]["files"].get(name) != digest for name, digest in prior.items()):
        raise ValueError("The prior OT package must remain unchanged for this validation")
    info["prior_package_files_unchanged"] = len(prior)
    write_json(args.output / "provenance.json", info)

    def execute(request, method):
        engine.config = replace(config, retrieval={"Edit-R": "edit", "OT-R": "ot"}.get(method, "suffix"))
        return engine.generate(request, "DFlash" if method == "DFlash" else "R+DFlash")

    rows, references = [], []
    try:
        warm = GenerationRequest("__warmup__", "__warmup__", 0, ({"role": "user", "content": "Say hello."},), 16)
        for method in METHODS:
            execute(warm, method)
        # Generate every AR reference before inspecting any treatment output. Never write them to history.
        for task in tasks:
            result = engine.generate(task.request, "AR")
            identity = identities[task.request.id]
            if (result.prompt_sha256 != identity["prompt_sha256"] or result.prompt_tokens != identity["prompt_tokens"]
                    or result.snapshot_id != identity["snapshot_id"]):
                raise ValueError("AR input differs from pre-generation identity")
            row = {**asdict(result), "contract": contract_check(result.text, task.expected_text)}
            references.append(row)
            append(args.output / "fresh-ar.jsonl", row)
            print("AR checked", task.request.id, len(result.token_ids), row["contract"], flush=True)
        tasks = attach_references(tasks, references)
        info["fresh_ar_correct"] = len(tasks)
        info["fresh_ar_artifact_correct"] = sum(r["contract"]["artifact_correct"] for r in references)
        write_json(args.output / "provenance.json", info)
        for task in tasks:
            for repeat in range(REPEATS):
                methods = list(METHODS)
                random.Random(f"ot-r-validation-v1:{task.request.id}:{repeat}").shuffle(methods)
                for method in methods:
                    result = execute(task.request, method)
                    try:
                        check_generation(result, task)
                    except ValueError:
                        write_json(args.output / "failure.json", {"method": method, "repeat": repeat,
                                                                 "result": asdict(result), "reference": task.reference})
                        raise
                    row = {**asdict(result), "method": method, "correct": True, "repeat": repeat,
                           "project_id": task.project_id, "group": task.group, "budget": task.request.max_new_tokens,
                           "contract": contract_check(result.text, task.expected_text)}
                    row.pop("text")
                    rows.append(row)
                    append(args.output / "rows.jsonl", row)
            print("completed", task.request.id, flush=True)
        if (original != {name: input_hashes(ROOT / "results" / name) for name in PROTECTED}
                or info["cases_sha256"] != file_hash(args.cases)):
            raise ValueError("Frozen input or old evidence changed")
        summary = audit_rows(rows, tasks)
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "comparisons.json", comparisons(summary, tasks))
        info.update(status="complete", original_unchanged=True, rows=len(rows))
        print(json.dumps(summary["groups"], indent=2), flush=True)
    except Exception as error:
        info.update(status="failed", error_type=type(error).__name__, error=str(error), rows=len(rows))
        raise
    finally:
        engine.close()
        write_json(args.output / "provenance.json", info)


if __name__ == "__main__":
    main()
