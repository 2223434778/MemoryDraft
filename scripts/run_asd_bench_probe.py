"""Direct GSM8K/MBPP ASD probe, with prompt-only retrieval as a separate arm."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
from unittest.mock import patch

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.evaluation.asd_bench import load_tasks, request_for, score
from memorydraft.evaluation.runtime_migration import file_hash
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest
from scripts.diagnose_transport_online import write
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("DFlash", "DFlash-ASD", "Splice", "Splice-ASD")


class BenchHook:
    def __init__(self, generate, method, *, mechanism=False, fixed=False):
        self.generate, self.method, self.mechanism, self.fixed = generate, method, mechanism, fixed
        self.policy, self.detail = None, None

    def __call__(self, *args, **kwargs):
        if "ASD" in self.method or self.mechanism:
            self.policy = ASDVerifier(ASDConfig(budget=8. if "ASD" in self.method else 0.), record=self.mechanism)
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
    output = {}
    for group in ("all", "gsm8k", "mbpp"):
        table = {}
        for method in METHODS:
            selected = [r for r in rows if r["method"] == method and (group == "all" or r["benchmark"] == group)]
            seconds = sum(r.get("latency_seconds", 0.) for r in selected)
            table[method] = dict(rows=len(selected), seconds=seconds, tokens=sum(len(r["token_ids"]) for r in selected),
                                 quality_correct=sum(r["quality"]["correct"] for r in selected),
                                 strict_identity=sum(r["strict_identity"] for r in selected),
                                 rounds=sum(r["counters"]["rounds"] for r in selected),
                                 reuse_rounds=sum(r["counters"]["reuse_rounds"] for r in selected),
                                 relaxed_tokens=sum(r["asd"]["relaxed_tokens"] for r in selected),
                                 regret=sum(r["asd"]["spent"] for r in selected),
                                 length_stops=sum(r["finish_reason"] == "length" for r in selected),
                                 splice_proposals=sum(r["counters"].get("splice_proposals", 0) for r in selected),
                                 splice_history_committed=sum(r["counters"].get("splice_history_accepted_tokens", 0) for r in selected))
            table[method]["tps"] = table[method]["tokens"] / seconds if seconds else None
        output[group] = table
    return output


def protected(root):
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root / "results").rglob("*"))
            if p.is_file() and p.relative_to(root / "results").parts[0] not in {"raw", "asd-bench"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = ROOT / "data/research/asd-bench/records.json"
    records = load_tasks(data)
    config = replace(RuntimeConfig.load(args.config), db=None, retrieval="suffix", prefix_risk={})
    engine = load_engine(config)
    original = protected(ROOT)
    info = {**provenance(engine), "status": "started", "runner_sha256": file_hash(Path(__file__)),
            "data_sha256": file_hash(data), "protocol_sha256": file_hash(ROOT / "docs/ASD_BENCH_PROTOCOL.md"),
            "original_sha256": original, "methods": METHODS, "asd": asdict(ASDConfig(budget=8.)),
            "scope": "GSM8K first 8 test records and MBPP test IDs11..18; Qwen3-8B/DFlash, not DSpark reproduction",
            "memory": "Each problem isolated; prompt-only retrieval; no reference solutions or other completions indexed",
            "quality": "GSM8K final numeric answer; MBPP provided public tests, not hidden tests",
            "fixed_work": "128 tokens with EOS ignored; quality is evaluated only under natural EOS"}
    write(args.output / "provenance.json", info)
    from memorydraft.runtime import engine as runtime_module
    native = runtime_module.generate_reuse

    def execute(request, method, *, mechanism=False, fixed=False):
        engine.config = replace(config, retrieval="edit-splice" if method.startswith("Splice") else "suffix")
        hook = BenchHook(native, method, mechanism=mechanism, fixed=fixed)
        with patch.object(runtime_module, "generate_reuse", hook):
            result = engine.generate(request, "R+DFlash" if method.startswith("Splice") else "DFlash")
        return result, hook

    references, outputs, fixed_outputs = {}, {}, {}
    try:
        warm = GenerationRequest("__warmup__", "__warmup__", 0, ({"role": "user", "content": "What is 3 plus 4?"},), 32)
        for method in METHODS:
            execute(warm, method)
        for stage, repeats in (("mechanism", 1), ("natural", 2), ("fixed", 2)):
            directory = args.output / stage
            directory.mkdir()
            rows = []
            for record in records:
                for repeat in range(repeats):
                    order = list(METHODS)
                    random.Random(f"asd-bench:{stage}:{record['id']}:{repeat}").shuffle(order)
                    if stage == "mechanism":
                        order = ["DFlash", *(m for m in order if m != "DFlash")]
                    for method in order:
                        request = request_for(record, max_new_tokens=128 if stage == "fixed" else 512)
                        result, hook = execute(request, method, mechanism=stage == "mechanism", fixed=stage == "fixed")
                        if stage == "mechanism" and method == "DFlash":
                            references[record["id"]] = asdict(result)
                        ref = references[record["id"]]
                        if (result.prompt_sha256, result.snapshot_id) != (ref["prompt_sha256"], ref["snapshot_id"]):
                            raise ValueError("Changed prompt or isolated history")
                        identity = result.token_ids == ref["token_ids"]
                        key = record["id"], method
                        if stage != "fixed" and "ASD" not in method and not identity:
                            raise ValueError("Strict retrieval output differs from strict DFlash")
                        if stage == "mechanism":
                            outputs[key] = result.token_ids
                        elif stage == "natural" and result.token_ids != outputs[key]:
                            raise ValueError("Instrumented and timed outputs differ")
                        if stage == "fixed":
                            if len(result.token_ids) != 128 or (key in fixed_outputs and fixed_outputs[key] != result.token_ids):
                                raise ValueError("Fixed-work length or repeat mismatch")
                            fixed_outputs[key] = result.token_ids
                        state = hook.policy.summary() if hook.policy else dict(budget=0., spent=0., relaxed_tokens=0,
                                                                              blocks=len(hook.detail.rounds), value="strict", approximate=False)
                        row = {**asdict(result), "method": method, "benchmark": record["benchmark"], "repeat": repeat,
                               "strict_identity": identity, "asd": state, "quality": score(record, result.text) if stage != "fixed" else {"correct": False, "not_evaluated": True}}
                        if stage == "mechanism":
                            row.update(events=hook.policy.events, rounds=hook.detail.rounds)
                            row["instrumented_latency_seconds"] = row.pop("latency_seconds")
                        rows.append(row)
                        append(directory / "rows.jsonl", row)
                print("completed", stage, record["id"], flush=True)
            write(directory / "summary.json", summarize(rows))
            info[stage + "_rows"] = len(rows)
            write(args.output / "references.json", references)
            write(args.output / "provenance.json", info)
        if any(fixed_outputs[r["id"], "DFlash"] != fixed_outputs[r["id"], "Splice"] for r in records):
            raise ValueError("Strict fixed-work methods differ")
        info["status"] = "complete"
    except Exception as error:
        info.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        info["original_unchanged"] = original == protected(ROOT)
        write(args.output / "provenance.json", info)
        engine.close()
    if not info["original_unchanged"]:
        raise ValueError("Original published results changed")


if __name__ == "__main__":
    main()
