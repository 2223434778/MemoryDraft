"""Natural-EOS math probe of measured-prefix utility under a persistent ASD budget."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
from unittest.mock import patch

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.evaluation.asd_budget import load_tasks, request_for, score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
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
    def __init__(self, generate, method, mechanism, policies=None):
        self.generate, self.method, self.mechanism = generate, method, mechanism
        self.policies = POLICIES if policies is None else policies
        self.policy, self.detail = None, None

    def __call__(self, *args, **kwargs):
        if self.method != "DFlash" or self.mechanism:
            self.policy = ASDVerifier(self.policies[self.method], record=self.mechanism)
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
            if p.is_file() and p.relative_to(root / "results").parts[0] != "raw"}


def select_records(records, benchmark, task_ids=None):
    selected = [r for r in records if r["benchmark"] == benchmark]
    if task_ids is not None:
        available = {r["id"] for r in selected}
        if not task_ids or len(set(task_ids)) != len(task_ids) or set(task_ids) - available:
            raise ValueError("Task IDs must be nonempty, unique and belong to the selected benchmark")
        selected = [r for r in selected if r["id"] in task_ids]
    return selected


def compare_previous(result, previous):
    if (result.prompt_sha256, result.snapshot_id) != (previous["prompt_sha256"], previous["snapshot_id"]):
        raise ValueError("Previous probe has a different Target prompt or history")
    old, new = previous["token_ids"], result.token_ids
    common = next((i for i,(a,b) in enumerate(zip(old,new)) if a != b), min(len(old),len(new)))
    normal = max(r["proposed"] for r in previous["rounds"])
    # A shorter final verification block changes ASD's remaining-length gate.
    # Only the prefix preceding that block must be identical across limits.
    affected = min((r["generated_before"] for r in previous["rounds"] if r["proposed"] < normal), default=len(old))
    required = affected if previous["finish_reason"] == "length" else len(old)
    if common < required:
        raise ValueError("Trajectory changed before the previous length-dependent block")
    return dict(previous_tokens=len(old), previous_finish_reason=previous["finish_reason"],
                common_prefix_tokens=common, required_prefix_tokens=required, exact_previous_prefix=common==len(old))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("gsm8k", "math500"), required=True)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--policies", type=Path)
    parser.add_argument("--prefix-tokens", type=int)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/ASD_BUDGET_PROTOCOL.md")
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 4096:
        parser.error("max-new-tokens must be in [1, 4096]")
    if args.prefix_tokens is not None and not 16 <= args.prefix_tokens < args.max_new_tokens:
        parser.error("prefix-tokens must be at least 16 and below max-new-tokens")
    policies = POLICIES
    if args.policies:
        from scripts.prepare_asd_attribution import load_policies
        policies = load_policies(args.policies)
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.data or ROOT / "data/research/asd-budget/records.json"
    if args.data:
        from scripts.prepare_asd_attribution import load_tasks as load_attribution_tasks
        records = load_attribution_tasks(data)
    else:
        records = load_tasks(data)
    records = select_records(records, args.benchmark, args.task_ids)
    previous, previous_info, previous_hash = {}, None, None
    if args.previous:
        directory = args.previous / args.benchmark
        previous = {(r["request_id"],r["method"]):r for r in read_rows(directory / "mechanism/rows.jsonl")}
        if any((r["id"],m) not in previous for r in records for m in policies):
            raise ValueError("Missing previous mechanism references")
        previous_info = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        if previous_info["status"] != "complete" or previous_info["policies"] != {m:asdict(c) for m,c in policies.items()}:
            raise ValueError("Incomplete previous probe or changed policies")
        path = directory / "mechanism/rows.jsonl"
        previous_hash = file_hash(path if path.exists() else path.with_suffix(".jsonl.gz"))
    base_config = RuntimeConfig.load(args.config)
    config = replace(base_config, db=None, retrieval="suffix", prefix_risk={},
                     max_new_tokens=max(base_config.max_new_tokens,args.max_new_tokens))
    engine = load_engine(config)
    original = protected(ROOT)
    info = {**provenance(engine), "status": "started", "benchmark": args.benchmark,
            "runner_sha256": file_hash(Path(__file__)), "data_sha256": file_hash(data),
            "protocol_sha256": file_hash(args.protocol),
            "original_sha256": original, "policies": {k: asdict(v) for k,v in policies.items()},
            "policies_sha256": file_hash(args.policies) if args.policies else None,
            "prefix_tokens": args.prefix_tokens,
            "selection": "Frozen task IDs and controls; no threshold search",
            "task_ids": [r["id"] for r in records], "max_new_tokens": args.max_new_tokens,
            "previous_rows_sha256": previous_hash,
            "quality": "GSM8K numeric; MATH-500 normalized boxed exact match, no symbolic equivalence",
            "memory": "All requests isolated, no retrieval or generated answers indexed",
            "eos": f"Natural EOS, maximum {args.max_new_tokens} tokens; no forced continuation stage"}
    if previous_info and (previous_info["source"]["sha256"] != info["source"]["sha256"]
                          or previous_info["data_sha256"] != info["data_sha256"]):
        engine.close()
        raise ValueError("Previous runtime package or task data changed")
    write(args.output / "provenance.json", info)
    from memorydraft.runtime import engine as runtime_module
    native = runtime_module.generate_reuse

    def execute(request, method, mechanism=False):
        hook = BudgetHook(native, method, mechanism, policies)
        with patch.object(runtime_module, "generate_reuse", hook):
            result = engine.generate(request, "DFlash")
        return result, hook

    references, outputs = {}, {}
    try:
        warm = GenerationRequest("__warmup__", "__warmup__", 0,
                                 ({"role": "user", "content": "What is 3 plus 4?"},), 32)
        for method in policies:
            execute(warm, method)
        stages = [("mechanism", 1), ("natural", 2)]
        if args.prefix_tokens:
            stages.append(("prefix", 2))
        prefix_outputs = {}
        for stage, repeats in stages:
            directory = args.output / stage
            directory.mkdir()
            count = 0
            for record in records:
                for repeat in range(repeats):
                    order = list(policies)
                    random.Random(f"asd-budget:{stage}:{record['id']}:{repeat}").shuffle(order)
                    if stage == "mechanism":
                        order = ["DFlash", *(m for m in order if m != "DFlash")]
                    for method in order:
                        limit = args.prefix_tokens if stage == "prefix" else args.max_new_tokens
                        request = replace(request_for(record), max_new_tokens=limit)
                        result, hook = execute(request, method, stage == "mechanism")
                        if stage == "mechanism" and method == "DFlash":
                            references[record["id"]] = asdict(result)
                        ref = references[record["id"]]
                        if (result.prompt_sha256, result.snapshot_id) != (ref["prompt_sha256"], ref["snapshot_id"]):
                            raise ValueError("Changed Target prompt or isolated history")
                        key = record["id"], method
                        if stage == "mechanism":
                            outputs[key] = asdict(result)
                        elif stage == "natural" and result.token_ids != outputs[key]["token_ids"]:
                            raise ValueError("Instrumented and timed outputs differ")
                        elif stage == "prefix":
                            if key in prefix_outputs and prefix_outputs[key] != result.token_ids:
                                raise ValueError("Repeated prefix trajectories differ")
                            prefix_outputs[key] = result.token_ids
                        state = hook.policy.summary() if hook.policy else dict(budget=0., spent=0., relaxed_tokens=0,
                                    blocks=result.counters["rounds"], value="strict", selection="strict", regret_price=0.,
                                    shortened_blocks=0, foregone_proposal_tokens=0, approximate=False)
                        row = {**asdict(result), "method": method, "benchmark": args.benchmark, "repeat": repeat,
                               "strict_identity": result.token_ids == ref["token_ids"], "asd": state,
                               "quality": score(record, result.text)}
                        if stage == "prefix":
                            from types import SimpleNamespace
                            row["rounds"] = hook.detail.rounds
                            row["full_prefix"] = compare_previous(SimpleNamespace(**outputs[key]), row)
                            # The reference has a different length cap in this stage.
                            row["strict_identity"] = None
                        if previous:
                            row["previous_prefix"] = compare_previous(result, previous[key])
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
