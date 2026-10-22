"""Fresh teacher features at strict prefixes matched to the old ASD positions."""
import argparse
from dataclasses import replace
from pathlib import Path
import time

import torch

from memorydraft.evaluation.adaptation import digest, parameter_guard, teacher_example
from memorydraft.evaluation.adaptation_followup import STUDY, aligned_controls, check_previous_publication, previous_rows
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import protected
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    out = ROOT/f"results/raw/{STUDY}/collect-{args.shard}"
    cache = ROOT/f"artifacts/followup-features-{args.shard}"
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    before, started = protected(ROOT), time.perf_counter()
    info = dict(status="started", shard=args.shard, previous_files_verified=check_previous_publication(ROOT))
    engine = None
    try:
        rows = aligned_controls(previous_rows(ROOT, "examples"), previous_rows(ROOT, "sources"))
        if len(rows) != 80:
            raise ValueError("Changed complete matched state coverage")
        engine = load_engine(replace(RuntimeConfig.load(ROOT/"configs/runtime-dflash.json"), db=None))
        engine.target.requires_grad_(False)
        guard = parameter_guard(engine.target)
        info["runtime"] = provenance(engine)
        count, total_bytes = 0, 0
        for i, row in enumerate(rows):
            if i % 2 != args.shard:
                continue
            if len(row["prefix_ids"]) > 1024:
                raise ValueError("Matched prefix exceeds fixed context budget")
            sample = teacher_example(engine.target, torch.tensor([row["prefix_ids"]], device=engine.target.device),
                                     engine.draft.target_layer_ids, 16, engine.stops)
            path = cache/(row["prefix_sha256"]+".pt")
            if path.exists():
                raise ValueError("Unexpected duplicate feature prefix")
            torch.save(sample, path)
            total_bytes += path.stat().st_size
            if total_bytes > 8*2**30:
                raise ValueError("Feature cache exceeds per-card budget")
            append(out/"examples.jsonl", dict(row, feature_path=path.relative_to(ROOT).as_posix(),
                   feature_sha256=digest(path.read_bytes()), teacher_tokens=sample["labels"][0].tolist(), context_tokens=sample["context"].shape[1]))
            count += 1
        if parameter_guard(engine.target) != guard:
            raise ValueError("Teacher changed during collection")
        peak = torch.cuda.max_memory_allocated()
        if not 0 < peak <= 46*2**30:
            raise ValueError("Collection exceeded GPU budget")
        info.update(status="complete", examples=count, cache_bytes=total_bytes, peak_gpu_bytes=peak,
                    seconds=time.perf_counter()-started, teacher_parameter_versions_unchanged=True)
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
