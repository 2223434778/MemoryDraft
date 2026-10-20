"""Natural-EOS math probe of measured-prefix utility under a persistent ASD budget."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import random
from unittest.mock import patch

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.evaluation.asd_budget import load_tasks, request_for, score
from memorydraft.evaluation.runtime_migration import file_hash
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest
from scripts.diagnose_transport_online import write
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {"DFlash": ASDConfig(budget=0), "ASD": ASDConfig(budget=8),
            "ASD-Run": ASDConfig(budget=8, value="exact-run"),
            "ASD-Utility": ASDConfig(budget=8, selection="utility", regret_price=2)}


class BudgetHook:
    def __init__(self, generate, method, mechanism):
        self.generate, self.method, self.mechanism = generate, method, mechanism
        self.policy, self.detail = None, None

    def __call__(self, *args, **kwargs):
        if self.method != "DFlash" or self.mechanism:
            self.policy = ASDVerifier(POLICIES[self.method], record=self.mechanism)
            kwargs["verifier"] = self.policy
        self.detail = self.generate(*args, **kwargs)
        if self.mechanism:
            if len(self.policy.events) != len(self.detail.rounds):
                raise ValueError("Missing verifier events")
            for event, row in zip(self.policy.events, self.detail.rounds):
                event["generated_before"] = row["generated_before"]
        return self.detail


def protected(root):
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root / "results").rglob("*"))
            if p.is_file() and p.relative_to(root / "results").parts[0] not in {"raw", "asd-budget"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("gsm8k", "math500"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = ROOT / "data/research/asd-budget/records.json"
    records = [r for r in load_tasks(data) if r["benchmark"] == args.benchmark]
    config = replace(RuntimeConfig.load(args.config), db=None, retrieval="suffix", prefix_risk={})
    engine = load_engine(config)
    original = protected(ROOT)
    info = {**provenance(engine), "status": "started", "benchmark": args.benchmark,
            "runner_sha256": file_hash(Path(__file__)), "data_sha256": file_hash(data),
            "protocol_sha256": file_hash(ROOT / "docs/ASD_BUDGET_PROTOCOL.md"),
            "original_sha256": original, "policies": {k: asdict(v) for k,v in POLICIES.items()},
            "selection": "Frozen new slices; price=2 chosen before these generations, no threshold search",
            "quality": "GSM8K numeric; MATH-500 normalized boxed exact match, no symbolic equivalence",
            "memory": "All requests isolated, no retrieval or generated answers indexed",
            "eos": "Natural EOS, maximum 512 tokens; no forced continuation stage"}
    write(args.output / "provenance.json", info)
    from memorydraft.runtime import engine as runtime_module
    native = runtime_module.generate_reuse

    def execute(request, method, mechanism=False):
        hook = BudgetHook(native, method, mechanism)
        with patch.object(runtime_module, "generate_reuse", hook):
            result = engine.generate(request, "DFlash")
        return result, hook

    references, outputs = {}, {}
    try:
        warm = GenerationRequest("__warmup__", "__warmup__", 0,
                                 ({"role": "user", "content": "What is 3 plus 4?"},), 32)
        for method in POLICIES:
            execute(warm, method)
        for stage, repeats in (("mechanism", 1), ("natural", 2)):
            directory = args.output / stage
            directory.mkdir()
            count = 0
            for record in records:
                for repeat in range(repeats):
                    order = list(POLICIES)
                    random.Random(f"asd-budget:{stage}:{record['id']}:{repeat}").shuffle(order)
                    if stage == "mechanism":
                        order = ["DFlash", *(m for m in order if m != "DFlash")]
                    for method in order:
                        result, hook = execute(request_for(record), method, stage == "mechanism")
                        if stage == "mechanism" and method == "DFlash":
                            references[record["id"]] = asdict(result)
                        ref = references[record["id"]]
                        if (result.prompt_sha256, result.snapshot_id) != (ref["prompt_sha256"], ref["snapshot_id"]):
                            raise ValueError("Changed Target prompt or isolated history")
                        key = record["id"], method
                        if stage == "mechanism":
                            outputs[key] = result.token_ids
                        elif result.token_ids != outputs[key]:
                            raise ValueError("Instrumented and timed outputs differ")
                        state = hook.policy.summary() if hook.policy else dict(budget=0., spent=0., relaxed_tokens=0,
                                    blocks=result.counters["rounds"], value="strict", selection="strict", regret_price=0.,
                                    shortened_blocks=0, foregone_proposal_tokens=0, approximate=False)
                        row = {**asdict(result), "method": method, "benchmark": args.benchmark, "repeat": repeat,
                               "strict_identity": result.token_ids == ref["token_ids"], "asd": state,
                               "quality": score(record, result.text)}
                        if stage == "mechanism":
                            row.update(events=hook.policy.events, rounds=hook.detail.rounds)
                            row["instrumented_latency_seconds"] = row.pop("latency_seconds")
                        append(directory / "rows.jsonl", row)
                        count += 1
                print("completed", stage, record["id"], flush=True)
            info[stage + "_rows"] = count
            write(args.output / "references.json", references)
            write(args.output / "provenance.json", info)
        info["status"] = "complete"
    except Exception as error:
        info.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        info["original_unchanged"] = original == protected(ROOT)
        write(args.output / "provenance.json", info)
        engine.close()
    if not info["original_unchanged"]:
        raise ValueError("Original published evidence changed")


if __name__ == "__main__":
    main()
