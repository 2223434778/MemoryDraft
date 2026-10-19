"""Small direct ASD/retrieval combination probe over eight already observed tasks."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
from unittest.mock import patch

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.evaluation.runtime_migration import file_hash
from memorydraft.evaluation.splice_probe import inputs as old_inputs, input_digest
from memorydraft.evaluation.transport_validation import contract_check, prompt_identity
from memorydraft.memory.store import MemoryStore
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest
from scripts.diagnose_transport_online import write
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "9426ce7b3b8cafb2d34f675a5bb9db2ebf0259f6"
IDS = ("payment-runbook:0", "payment-runbook:1", "shipment-contract:0", "shipment-contract:1",
       "inventory-query:0", "inventory-query:1", "edge-router:1", "short-settings:1")
METHODS = ("DFlash", "DFlash-ASD", "Suffix", "Suffix-ASD", "Guided", "Guided-ASD",
           "Splice", "Splice-ASD", "Splice-ASD-Run")
RETRIEVAL = {"DFlash": "suffix", "Suffix": "suffix", "Guided": "edit-guided", "Splice": "edit-splice"}


def inputs(root):
    tasks, events = old_inputs(root)
    keyed = {t.request.id: t for t in tasks}
    selected = [keyed[name] for name in IDS]
    users = {t.request.user_id for t in selected}
    return selected, [e for e in events if e.user_id in users]


class DecodeHook:
    def __init__(self, generate, method, *, mechanism=False, fixed=False):
        self.generate, self.method = generate, method
        self.mechanism, self.fixed = mechanism, fixed
        self.policy, self.detail = None, None

    def __call__(self, *args, **kwargs):
        approximate = "ASD" in self.method
        if approximate or self.mechanism:
            self.policy = ASDVerifier(ASDConfig(budget=2. if approximate else 0.,
                                                value="exact-run" if self.method.endswith("-Run") else "remaining"),
                                      record=self.mechanism)
            kwargs["verifier"] = self.policy
        if self.fixed:
            kwargs["stop_token_ids"] = ()
        self.detail = self.generate(*args, **kwargs)
        if self.mechanism:
            for event, row in zip(self.policy.events, self.detail.rounds):
                event.update(generated_before=row["generated_before"], route=row["route"], lookup=row["lookup"],
                             draft_prefix=row.get("draft_prefix"))
        return self.detail


def summarize(rows):
    table = {}
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        if not selected:
            continue
        seconds = sum(r.get("latency_seconds", 0.) for r in selected)
        table[method] = dict(rows=len(selected), seconds=seconds,
                             tokens=sum(len(r["token_ids"]) for r in selected),
                             strict_identity=sum(r["strict_identity"] for r in selected),
                             contract_correct=sum(r["contract"]["artifact_correct"] for r in selected),
                             rounds=sum(r["counters"]["rounds"] for r in selected),
                             relaxed_tokens=sum(r["asd"]["relaxed_tokens"] for r in selected),
                             regret=sum(r["asd"]["spent"] for r in selected),
                             length_stops=sum(r["finish_reason"] == "length" for r in selected))
        table[method]["tps"] = table[method]["tokens"] / seconds if seconds else None
    return table


def published_hashes(root):
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root / "results").rglob("*"))
            if p.is_file() and p.relative_to(root / "results").parts[0] not in {"raw", "asd-splice"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks, events = inputs(ROOT)
    database = args.output.resolve() / "history.db"
    with MemoryStore(database) as store:
        store.append(events)
        if any(store.snapshot(t.request.user_id, t.request.order).snapshot_id != t.snapshot_id for t in tasks):
            raise ValueError("Changed legal snapshot")
    config = replace(RuntimeConfig.load(args.config), db=str(database), retrieval="suffix", prefix_risk={})
    engine = load_engine(config)
    original = published_hashes(ROOT)
    info = {**provenance(engine), "status": "started", "upstream_asd_commit": UPSTREAM,
            "scope": "Eight previously observed development requests; approximate decoding, not lossless or holdout",
            "inputs_sha256": input_digest(tasks, events), "original_sha256": original,
            "asd": asdict(ASDConfig()), "methods": METHODS, "ids": IDS,
            "fixed_work": "64 generated tokens, EOS ignored for throughput only; natural EOS quality is separate"}
    write(args.output / "provenance.json", info)
    write(args.output / "references.json", {t.request.id: t.reference for t in tasks})
    from memorydraft.runtime import engine as runtime_module
    original_generate = runtime_module.generate_reuse

    def execute(request, method, *, mechanism=False, fixed=False):
        base = method.split("-")[0]
        engine.config = replace(config, retrieval=RETRIEVAL[base])
        hook = DecodeHook(original_generate, method, mechanism=mechanism, fixed=fixed)
        with patch.object(runtime_module, "generate_reuse", hook):
            result = engine.generate(request, "DFlash" if base == "DFlash" else "R+DFlash")
        return result, hook

    try:
        for task in tasks:
            identity = prompt_identity(engine.tokenizer, task.request, config.template_options)
            if any(task.reference[k] != identity[k] for k in ("prompt_sha256", "prompt_tokens")):
                raise ValueError("Prompt differs from its original AR reference")
        warm = GenerationRequest("__warmup__", "__warmup__", 0, ({"role": "user", "content": "Say hello."},), 32)
        for method in METHODS:
            execute(warm, method)
        mechanism_outputs = {}
        fixed_outputs = {}
        for stage, repeats in (("mechanism", 1), ("natural", 2), ("fixed", 2)):
            rows = []
            directory = args.output / stage
            directory.mkdir()
            for task in tasks:
                for repeat in range(repeats):
                    order = list(METHODS)
                    random.Random(f"asd-splice:{stage}:{task.request.id}:{repeat}").shuffle(order)
                    for method in order:
                        request = replace(task.request, max_new_tokens=64) if stage == "fixed" else task.request
                        result, hook = execute(request, method, mechanism=stage == "mechanism", fixed=stage == "fixed")
                        if result.prompt_sha256 != task.reference["prompt_sha256"] or result.snapshot_id != task.snapshot_id:
                            raise ValueError("Changed Target input or history")
                        identity = result.token_ids == task.reference["token_ids"]
                        if stage != "fixed" and "ASD" not in method and not identity:
                            raise ValueError("Strict or zero-budget output differs from original AR")
                        key = (task.request.id, method)
                        if stage == "mechanism":
                            mechanism_outputs[key] = result.token_ids
                        elif stage == "natural" and result.token_ids != mechanism_outputs[key]:
                            raise ValueError("Instrumented and timed outputs differ")
                        if stage == "fixed":
                            if len(result.token_ids) != 64:
                                raise ValueError("Unequal fixed output work")
                            if key in fixed_outputs and result.token_ids != fixed_outputs[key]:
                                raise ValueError("Fixed-work repeats diverge")
                            fixed_outputs[key] = result.token_ids
                        state = hook.policy.summary() if hook.policy else dict(budget=0., spent=0., relaxed_tokens=0,
                                                                              blocks=len(hook.detail.rounds), value="strict", approximate=False)
                        row = {**asdict(result), "method": method, "repeat": repeat, "project_id": task.project_id,
                               "strict_identity": identity, "contract": contract_check(result.text, task.expected_text),
                               "asd": state, "fixed_work": stage == "fixed"}
                        if stage == "mechanism":
                            row.update(events=hook.policy.events, rounds=hook.detail.rounds)
                            row["instrumented_latency_seconds"] = row.pop("latency_seconds")
                        rows.append(row)
                        append(directory / "rows.jsonl", row)
                print("completed", stage, task.request.id, flush=True)
            write(directory / "summary.json", summarize(rows))
            info[stage + "_rows"] = len(rows)
            write(args.output / "provenance.json", info)
        for task in tasks:
            strict = [fixed_outputs[(task.request.id, m)] for m in METHODS if "ASD" not in m]
            if any(tokens != strict[0] for tokens in strict):
                raise ValueError("Strict fixed-work methods disagree")
        info["status"] = "complete"
    except Exception as error:
        info.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        info["original_unchanged"] = original == published_hashes(ROOT)
        write(args.output / "provenance.json", info)
        engine.close()
    if not info["original_unchanged"]:
        raise ValueError("Original published evidence changed")


if __name__ == "__main__":
    main()
