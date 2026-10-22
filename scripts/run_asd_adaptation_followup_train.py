"""One equal-budget strict-position control; old adapters remain read-only."""
from dataclasses import replace
import json
from pathlib import Path
import random
import time

import torch

from memorydraft.evaluation.adaptation import ACCUMULATE, SEED, UPDATES, attach_lora, digest, evaluate_blocks, parameter_guard, sample_loss
from memorydraft.evaluation.adaptation_followup import STUDY, load_examples, previous_rows
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import protected
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]


def main():
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    out, assets = ROOT/f"results/raw/{STUDY}/train", ROOT/"artifacts/followup-adapter"
    out.mkdir(parents=True, exist_ok=False)
    assets.mkdir(parents=True, exist_ok=False)
    before = protected(ROOT)
    info = dict(status="started", arm="ordinary-position", updates=UPDATES, accumulation=ACCUMULATE, initialization_seed=SEED)
    engine = None
    try:
        rows = [r for i in (0, 1) for r in read_rows(ROOT/f"results/raw/{STUDY}/collect-{i}/examples.jsonl")]
        training_rows = sorted([r for r in rows if r["split"] == "train"], key=lambda r:(r["request_id"], r["source_round"]))
        calibration_rows = [r for r in previous_rows(ROOT, "examples") if r["split"] == "calibration"]
        training = load_examples(ROOT, training_rows, previous=False)
        calibration = load_examples(ROOT, calibration_rows, previous=True)
        old = json.loads((ROOT/"results/asd-adaptation/train-asd-prefix/provenance.json").read_text(encoding="utf-8"))
        keys = [[r["request_id"], r["source_round"]] for r in training_rows]
        if keys != old["train_order_keys"] or len(training) != 64 or len(calibration) != 32:
            raise ValueError("Matched control changed old training/shuffle/calibration coverage")
        engine = load_engine(replace(RuntimeConfig.load(ROOT/"configs/runtime-dflash.json"), db=None))
        model = attach_lora(engine.target, engine.draft)
        guard = parameter_guard(engine.target)
        params = [p for p in model.parameters() if p.requires_grad]
        frozen = tuple((n, p.data_ptr(), p._version) for n, p in model.named_parameters() if not p.requires_grad)
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
        info.update(train_states=len(training), train_requests=len({r["request_id"] for r, _ in training}),
                    calibration_states=len(calibration), trainable_parameters=sum(p.numel() for p in params), train_order_keys=keys,
                    calibration_identity_sha256=digest(json.dumps(calibration_rows, sort_keys=True).encode()))
        torch.cuda.reset_peak_memory_stats()
        engine.synchronize()
        started = time.perf_counter()
        initial = evaluate_blocks(engine.target, model, calibration)
        info["initial_calibration"] = initial
        if abs(initial["loss"]-old["initial_calibration"]["loss"]) > 1e-6:
            raise ValueError("Common calibration did not reproduce frozen initialization")
        append(out/"metrics.jsonl", dict(update=0, calibration=initial))
        best, best_step, cycle, order, position = float("inf"), None, 0, [], 0
        losses, first_grad_norm = [], None
        for update in range(1, UPDATES+1):
            if time.perf_counter()-started > 900:
                raise RuntimeError("Training exceeded the frozen budget")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for _ in range(ACCUMULATE):
                if position == len(order):
                    order = list(range(len(training)))
                    random.Random(SEED+cycle).shuffle(order)
                    cycle, position = cycle+1, 0
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
                append(out/"metrics.jsonl", dict(update=update, mean_training_loss=sum(losses[-32:])/32,
                       calibration=measured, seconds=time.perf_counter()-started))
                state = {n:p.detach().cpu().contiguous().clone() for n, p in get_peft_model_state_dict(model).items()}
                if measured["loss"] < best:
                    best, best_step = measured["loss"], update
                    save_file(state, str(assets/"selected.safetensors"))
                if update == UPDATES:
                    save_file(state, str(assets/"final.safetensors"))
                print("ordinary-position", update, measured, flush=True)
        engine.synchronize()
        if parameter_guard(engine.target) != guard or any(p.grad is not None for p in engine.target.parameters()):
            raise ValueError("Frozen teacher changed")
        if frozen != tuple((n, p.data_ptr(), p._version) for n, p in model.named_parameters() if not p.requires_grad):
            raise ValueError("Frozen draft base changed")
        peak = torch.cuda.max_memory_allocated()
        if not 0 < peak <= 46*2**30:
            raise ValueError("Training exceeded GPU budget")
        info.update(status="complete", seconds=time.perf_counter()-started, selected_step=best_step,
                    initial_calibration=initial, selected_calibration_loss=best, first_grad_norm=first_grad_norm,
                    teacher_parameter_versions_unchanged=True, teacher_gradients_absent=True, draft_base_versions_unchanged=True,
                    peak_gpu_bytes=peak, adapter_path=(assets/"selected.safetensors").relative_to(ROOT).as_posix(),
                    adapter_sha256=digest((assets/"selected.safetensors").read_bytes()), final_sha256=digest((assets/"final.safetensors").read_bytes()),
                    tensor_shapes={n:list(p.shape) for n, p in get_peft_model_state_dict(model).items()})
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
