"""Collect matched strict/ASD deployment prefixes and fresh teacher supervision."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.evaluation.adaptation import SOURCE_ROUNDS, digest, prefix_digest, teacher_example
from memorydraft.evaluation.asd_suite import request_for
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import protected
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]


def questions():
    base = ROOT/"data/research/asd-adaptation"
    manifest = json.loads((base/"manifest.json").read_text(encoding="utf-8"))
    raw = (base/"questions.json").read_bytes()
    if digest(raw) != manifest["questions_sha256"]:
        raise ValueError("Changed frozen adaptation questions")
    rows = json.loads(raw)
    if len(rows) != 40 or any(set(r) != {"id", "benchmark", "source_index", "question", "split"} for r in rows):
        raise ValueError("Training source must contain only question metadata")
    return rows


def collect_source(engine, record, arm):
    from memorydraft.runtime import engine as module
    native, detail, captured = module.generate_reuse, None, {}
    policy = ASDVerifier(ASDConfig(budget=8, gate=.25, cap=2), record=True)
    def hook(*args, **kwargs):
        nonlocal detail
        prompt, step = args[2][0].tolist(), [0]
        def observe(confirmed, tokens, logits, budget):
            index = step[0]
            if index in SOURCE_ROUNDS:
                captured[index] = dict(prefix_ids=[*prompt, *confirmed], generated_before=len(confirmed),
                                       source_round=index, past_spent=policy.spent, past_relaxed=policy.relaxed)
            step[0] += 1
        kwargs.update(on_draft=observe, incremental_stop=True)
        if arm == "asd-prefix":
            kwargs["verifier"] = policy
        detail = native(*args, **kwargs)
        return detail
    with patch.object(module, "generate_reuse", hook):
        result = engine.generate(request_for(record, 256), "DFlash")
    row = dict(asdict(result), source_policy=arm, split=record["split"], rounds=detail.rounds,
               asd=policy.summary() if arm == "asd-prefix" else None, events=policy.events)
    for index, point in captured.items():
        if point["generated_before"] != detail.rounds[index]["generated_before"]:
            raise ValueError("Captured prefix missed actual source frontier")
    return row, captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    out = ROOT/f"results/raw/asd-adaptation/collect-{args.shard}"
    cache = ROOT/f"artifacts/adaptation-features-{args.shard}"
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    before, started = protected(ROOT), time.perf_counter()
    info = dict(status="started", shard=args.shard, original_sha256=before, source_limit=256, state_rounds=SOURCE_ROUNDS)
    write(out/"provenance.json", info)
    engine = None
    try:
        engine = load_engine(replace(RuntimeConfig.load(ROOT/"configs/runtime-dflash.json"), db=None, max_new_tokens=4096))
        info["runtime"] = provenance(engine)
        available, total_bytes, examples, max_peak = {}, 0, 0, 0
        for ordinal, record in enumerate(questions()):
            if ordinal % 2 != args.shard:
                continue
            captures = {}
            for arm in ("ordinary", "asd-prefix"):
                row, captures[arm] = collect_source(engine, record, arm)
                append(out/"sources.jsonl", row)
                max_peak = max(max_peak, row["peak_gpu_bytes"])
            for index in SOURCE_ROUNDS:
                if any(index not in captures[a] for a in captures):
                    append(out/"unavailable.jsonl", dict(request_id=record["id"], source_round=index,
                           available_arms=[a for a in captures if index in captures[a]], reason="source ended before fixed round"))
                    continue
                for arm in captures:
                    point = captures[arm][index]
                    prefix = point["prefix_ids"]
                    if len(prefix) > 1024:
                        raise ValueError("Training prefix exceeds frozen context budget")
                    key = prefix_digest(prefix)
                    path = cache/(key+".pt")
                    if key not in available:
                        sample = teacher_example(engine.target, torch.tensor([prefix], device=engine.target.device),
                                                 engine.draft.target_layer_ids, 16, engine.stops)
                        torch.save(sample, path)
                        total_bytes += path.stat().st_size
                        if total_bytes > 8*2**30:
                            raise ValueError("Feature cache exceeds per-card 8GiB budget")
                        available[key] = dict(feature_path=path.relative_to(ROOT).as_posix(), feature_sha256=digest(path.read_bytes()),
                                              teacher_tokens=sample["labels"][0].tolist(), context_tokens=sample["context"].shape[1])
                    entry = dict(request_id=record["id"], split=record["split"], source_policy=arm,
                                 prefix_sha256=key, **point, **available[key])
                    append(out/"examples.jsonl", entry)
                    examples += 1
                    max_peak = max(max_peak, torch.cuda.max_memory_allocated())
            info.update(examples=examples, cache_bytes=total_bytes, peak_gpu_bytes=max_peak)
            write(out/"provenance.json", info)
            print("collected", record["id"], record["split"], examples, flush=True)
        if not 0 < max_peak <= 46*2**30:
            raise ValueError("Teacher collection exceeded GPU budget")
        info.update(status="complete", seconds=time.perf_counter()-started)
    except Exception as error:
        info.update(status="failed", error=type(error).__name__+": "+str(error))
        raise
    finally:
        info["original_unchanged"] = before == protected(ROOT)
        write(out/"provenance.json", info)
        if engine is not None:
            engine.close()
        if not info["original_unchanged"]:
            raise ValueError("Previous evidence changed")


if __name__ == "__main__":
    main()
