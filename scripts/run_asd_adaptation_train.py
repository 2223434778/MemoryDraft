"""Equal-update LoRA controls, selected by common internal teacher calibration."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import time

import torch

from memorydraft.evaluation.adaptation import (ACCUMULATE, ARMS, SEED, UPDATES, attach_lora, digest,
                                               evaluate_blocks, parameter_guard, sample_loss)
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import protected
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]


def example_index():
    rows = [json.loads(line) for i in (0, 1) for line in
            (ROOT/f"results/raw/asd-adaptation/collect-{i}/examples.jsonl").read_text(encoding="utf-8").splitlines()]
    pairs = {}
    for row in rows:
        key = row["request_id"], row["source_round"]
        arms = pairs.setdefault(key, {})
        if row["source_policy"] in arms:
            raise ValueError("Duplicate training state")
        arms[row["source_policy"]] = row
    if any(set(v) != set(ARMS) for v in pairs.values()):
        raise ValueError("Unpaired training states")
    counts = {s: len({r["request_id"] for r in rows if r["split"] == s}) for s in ("train", "calibration")}
    if counts["train"] < 16 or counts["calibration"] < 4:
        raise ValueError("Insufficient fixed-prefix coverage")
    return rows


def load_examples(rows):
    cache = {}
    result = []
    for row in rows:
        path = ROOT/row["feature_path"]
        if row["feature_path"] not in cache:
            if digest(path.read_bytes()) != row["feature_sha256"]:
                raise ValueError("Changed teacher features")
            cache[row["feature_path"]] = torch.load(path, map_location="cpu", weights_only=True)
        sample = cache[row["feature_path"]]
        if sample["labels"][0].tolist() != row["teacher_tokens"] or sample["prefix_ids"][0].tolist() != row["prefix_ids"]:
            raise ValueError("Teacher sample metadata mismatch")
        result.append((row, sample))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    out = ROOT/f"results/raw/asd-adaptation/train-{args.arm}"
    assets = ROOT/f"artifacts/adaptation-adapters/{args.arm}"
    out.mkdir(parents=True, exist_ok=False)
    assets.mkdir(parents=True, exist_ok=False)
    before = protected(ROOT)
    info = dict(status="started", arm=args.arm, original_sha256=before, updates=UPDATES, accumulation=ACCUMULATE)
    write(out/"provenance.json", info)
    engine = None
    try:
        rows = example_index()
        training = load_examples(sorted([r for r in rows if r["split"] == "train" and r["source_policy"] == args.arm],
                                        key=lambda r: (r["request_id"], r["source_round"])))
        calibration = load_examples([r for r in rows if r["split"] == "calibration"])
        engine = load_engine(replace(RuntimeConfig.load(ROOT/"configs/runtime-dflash.json"), db=None))
        model = attach_lora(engine.target, engine.draft)
        guard = parameter_guard(engine.target)
        params = [p for p in model.parameters() if p.requires_grad]
        frozen = tuple((n, p.data_ptr(), p._version) for n, p in model.named_parameters() if not p.requires_grad)
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
        info.update(train_states=len(training), train_requests=len({r["request_id"] for r, _ in training}),
                    calibration_states=len(calibration), trainable_parameters=sum(p.numel() for p in params),
                    train_order_keys=[(r["request_id"], r["source_round"]) for r, _ in training])
        torch.cuda.reset_peak_memory_stats()
        engine.synchronize()
        started = time.perf_counter()
        initial = evaluate_blocks(engine.target, model, calibration)
        info["initial_calibration"] = initial
        append(out/"metrics.jsonl", dict(update=0, calibration=initial))
        best, best_step, cycle, order, position = float("inf"), None, 0, [], 0
        losses, first_grad_norm = [], None
        for update in range(1, UPDATES+1):
            if time.perf_counter()-started > 900:
                raise RuntimeError("Training exceeded frozen 900-second budget")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for _ in range(ACCUMULATE):
                if position == len(order):
                    order = list(range(len(training)))
                    random.Random(SEED+cycle).shuffle(order)
                    cycle += 1
                    position = 0
                _, sample = training[order[position]]
                position += 1
                loss = sample_loss(engine.target, model, sample)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite adaptation loss")
                total += float(loss.detach())/ACCUMULATE
                (loss/ACCUMULATE).backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite adapter gradient")
            if first_grad_norm is None:
                first_grad_norm = float(norm)
                if first_grad_norm <= 0:
                    raise ValueError("Adapter receives no gradient")
            optimizer.step()
            losses.append(total)
            if update % 32 == 0:
                measured = evaluate_blocks(engine.target, model, calibration)
                metrics = dict(update=update, mean_training_loss=sum(losses[-32:])/32, calibration=measured,
                               seconds=time.perf_counter()-started)
                append(out/"metrics.jsonl", metrics)
                state = {n: p.detach().cpu().contiguous().clone() for n, p in get_peft_model_state_dict(model).items()}
                if measured["loss"] < best:
                    best, best_step = measured["loss"], update
                    save_file(state, str(assets/"selected.safetensors"))
                if update == UPDATES:
                    save_file(state, str(assets/"final.safetensors"))
                print(args.arm, update, measured, flush=True)
        engine.synchronize()
        if parameter_guard(engine.target) != guard or any(p.grad is not None for p in engine.target.parameters()):
            raise ValueError("Frozen teacher changed during adaptation")
        if frozen != tuple((n, p.data_ptr(), p._version) for n, p in model.named_parameters() if not p.requires_grad):
            raise ValueError("Frozen draft base parameters changed")
        peak = torch.cuda.max_memory_allocated()
        if not 0 < peak <= 46*2**30:
            raise ValueError("Training exceeded GPU memory budget")
        tensors = get_peft_model_state_dict(model)
        info.update(status="complete", seconds=time.perf_counter()-started, selected_step=best_step,
                    selected_calibration_loss=best, first_grad_norm=first_grad_norm, peak_gpu_bytes=peak,
                    teacher_parameter_versions_unchanged=True, teacher_gradients_absent=True, draft_base_versions_unchanged=True,
                    adapter_path=(assets/"selected.safetensors").relative_to(ROOT).as_posix(),
                    adapter_sha256=digest((assets/"selected.safetensors").read_bytes()),
                    final_sha256=digest((assets/"final.safetensors").read_bytes()),
                    tensor_shapes={n:list(p.shape) for n, p in tensors.items()})
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
