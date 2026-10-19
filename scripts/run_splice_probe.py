"""Strict post-hoc mechanism replay; optional, gated uninstrumented development timing."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import ot

from memorydraft.evaluation.runtime_migration import check_result, file_hash
from memorydraft.evaluation.splice_probe import METHODS, RETRIEVAL, audit, input_digest, inputs, mechanism_gate
from memorydraft.evaluation.splice_probe import COOLDOWN_IDS, prefix_acceptance, protected_hashes
from memorydraft.memory.store import MemoryStore
from memorydraft.runtime.benchmark import provenance, source_digest
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest
from scripts.diagnose_transport_online import write
from scripts.run_runtime_migration import append


ROOT = Path(__file__).resolve().parents[1]


class Recorder:
    def __init__(self, generate):
        self.generate, self.calls, self.rounds = generate, [], []

    def __call__(self, *args, **kwargs):
        matcher = kwargs["matcher"]
        guided = getattr(matcher, "propose_with_draft", None)
        if guided is not None:
            observe = matcher.observe

            def propose(confirmed, draft_tokens, max_tokens):
                proposal = guided(confirmed, draft_tokens, max_tokens)
                self.calls.append(dict(generated=len(confirmed), draft_tokens=list(draft_tokens),
                                       proposal={**asdict(proposal), "draft_prefix": getattr(proposal, "draft_prefix", 0)}
                                       if proposal else None, accepted=None))
                return proposal

            def accept(proposal, accepted, confirmed):
                if self.calls and self.calls[-1]["proposal"] is not None and self.calls[-1]["accepted"] is None:
                    self.calls[-1]["accepted"] = accepted
                return observe(proposal, accepted, confirmed)

            matcher.propose_with_draft, matcher.observe = propose, accept
        result = self.generate(*args, **kwargs)
        self.rounds = result.rounds
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("mechanism", "timing"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path)
    parser.add_argument("--cooldown-check", action="store_true", help="Post-hoc five-request feedback repair check")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks, events = inputs(ROOT)
    if args.cooldown_check:
        tasks = [t for t in tasks if t.request.id in COOLDOWN_IDS]
        events = [e for e in events if e.user_id in {t.request.user_id for t in tasks}]
        if {t.request.id for t in tasks} != set(COOLDOWN_IDS):
            raise ValueError("Changed five-request repair scope")
    if args.stage == "timing":
        if args.mechanism is None:
            parser.error("Timing requires the completed mechanism directory")
        prior = json.loads((args.mechanism / "provenance.json").read_text())
        if (prior["status"] != "complete" or prior["stage"] != "mechanism" or prior["rows"] != len(tasks) * len(METHODS)
                or prior.get("cooldown_check", False) != args.cooldown_check
                or not prior["mechanism_gate"]["passed"] or prior["source"] != source_digest()
                or prior["runner_sha256"] != file_hash(Path(__file__)) or prior["inputs_sha256"] != input_digest(tasks, events)):
            raise ValueError("Timing requires the identical checked implementation and a passed mechanism gate")
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.output.resolve() / "history.db"
    with MemoryStore(database) as store:
        store.append(events)
        if any(store.snapshot(t.request.user_id, t.request.order).snapshot_id != t.snapshot_id for t in tasks):
            raise ValueError("Changed legal snapshot")
    config = replace(RuntimeConfig.load(args.config), db=str(database), retrieval="suffix", alignment={})
    if (config.backend != "dflash" or config.threads != 4 or config.template_options != {"enable_thinking": False}
            or config.reuse != {"max_tokens": 32, "min_match": 6, "cooldown_rounds": 1, "max_pending_tokens": 1024}):
        raise ValueError("Use the frozen FP32/eager DFlash profile")
    hashes = protected_hashes(ROOT, include_splice=args.cooldown_check)
    started = time.perf_counter()
    engine = load_engine(config)
    info = {**provenance(engine), "status": "started", "stage": args.stage, "POT": ot.__version__,
            "scope": "Post-hoc development replay only; all Target answers were previously observed",
            "instrumented": args.stage == "mechanism", "methods": METHODS, "cooldown_check": args.cooldown_check,
            "load_seconds": time.perf_counter() - started, "inputs_sha256": input_digest(tasks, events),
            "runner_sha256": file_hash(Path(__file__)),
            "protocol_sha256": file_hash(ROOT / "docs" / ("OT_SPLICE_COOLDOWN_PROTOCOL.md" if args.cooldown_check else "OT_SPLICE_PROTOCOL.md")),
            "original_sha256": hashes}
    write(args.output / "provenance.json", info)
    rows = []
    repeats = 1 if args.stage == "mechanism" else 2

    def execute(request, method):
        engine.config = replace(config, retrieval=RETRIEVAL[method])
        return engine.generate(request, "R+DFlash")

    try:
        warm = GenerationRequest("__warmup__", "__warmup__", 0, ({"role": "user", "content": "Say hello."},), 16)
        for method in METHODS:
            execute(warm, method)
        from memorydraft.runtime import engine as runtime_module
        for task in tasks:
            for repeat in range(repeats):
                order = list(METHODS)
                random.Random(f"splice-v1:{task.request.id}:{repeat}").shuffle(order)
                for method in order:
                    recorder = Recorder(runtime_module.generate_reuse)
                    if args.stage == "mechanism":
                        with patch.object(runtime_module, "generate_reuse", recorder):
                            result = execute(task.request, method)
                    else:
                        result = execute(task.request, method)
                    check_result(result, task)
                    row = {**asdict(result), "method": method, "repeat": repeat, "correct": True,
                           "project_id": task.project_id, "group": getattr(task, "group", "development")}
                    row.pop("text")
                    if args.stage == "mechanism":
                        for call in recorder.calls:
                            remaining = task.reference["token_ids"][call["generated"]:]
                            call["draft_oracle_accepted"] = prefix_acceptance(call["draft_tokens"], remaining)
                            call["oracle_accepted"] = (prefix_acceptance(call["proposal"]["tokens"], remaining)
                                                       if call["proposal"] else None)
                        row.update(calls=recorder.calls, rounds=recorder.rounds)
                        row["instrumented_latency_seconds"] = row.pop("latency_seconds")
                        row.pop("ttft_seconds")
                    rows.append(row)
                    append(args.output / "rows.jsonl", row)
            print("completed", task.request.id, flush=True)
        summary = audit(rows, tasks, repeats, instrumented=args.stage == "mechanism")
        if args.stage == "mechanism":
            info["mechanism_gate"] = mechanism_gate(rows)
        write(args.output / "summary.json", summary)
        info.update(status="complete", rows=len(rows), original_unchanged=hashes == protected_hashes(ROOT, include_splice=args.cooldown_check))
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
