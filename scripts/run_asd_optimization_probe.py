"""Bounded, explicit execution/length/drafter checks; no change to strict defaults."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
from unittest.mock import patch

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.backends.dflash_reuse import generate_reuse
from memorydraft.evaluation.asd_budget import request_for, score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.runtime.config import RuntimeConfig
from memorydraft.runtime.engine import InferenceEngine, verify_dflash_assets
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import protected
from scripts.run_asd_compute_probe import selected_states, sources
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = ("math500:1", "math500:4", "math500:7", "gsm8k:21")
TRANSFER_IDS = ("math500:8", "math500:9", "gsm8k:24", "gsm8k:25")
PROFILES = {"fp32-eager": (torch.float32, "eager"), "fp32-sdpa": (torch.float32, "sdpa"),
            "bf16-sdpa": (torch.bfloat16, "sdpa"), "bf16-eager": (torch.bfloat16, "eager")}
# Tuple: verification policy, smaller draft block (including anchor), length rule, incremental EOS.
ARMS = {"D-old": ("strict", None, "full", False), "D-fast": ("strict", None, "full", True),
        "U-old": ("utility", None, "full", False), "U-fast": ("utility", None, "full", True),
        "ASD-fast": ("longest", None, "full", True),
        "D-verify7": ("strict", None, "short", True), "U-verify7": ("utility", None, "short", True),
        "D-draft8": ("strict", 8, "full", True), "U-draft8": ("utility", 8, "full", True),
        "U-recent": ("utility", None, "recent", True), "U-budget": ("utility", None, "budget", True)}


def records():
    indexed = {}
    for name in ("asd-budget", "asd-attribution"):
        data = json.loads((ROOT / "data/research" / name / "records.json").read_text(encoding="utf-8"))
        indexed.update((r["id"], r) for r in data["records"])
    return indexed


class LengthPolicy:
    """Only previous decisions and the current remaining ledger inform length."""
    def __init__(self, kind, rule):
        self.kind, self.rule = kind, rule
        self.policy = ASDVerifier(ASDConfig(budget=0 if kind == "strict" else 8,
                                           selection="utility" if kind == "utility" else "longest"))
        self.advances, self.selected = [], []

    def choose(self, maximum):
        short = self.rule == "short"
        if self.rule in {"recent", "budget"} and len(self.advances) >= 4:
            short = sum(self.advances[-4:]) / 4 <= 4
        if self.rule == "budget" and self.policy.spent >= 6:
            short = True
        k = min(maximum, 7) if short else maximum
        self.selected.append(k)
        return k

    def decide(self, tokens, logits, *, stop_token_ids=()):
        decision = self.policy.decide(tokens, logits, stop_token_ids=stop_token_ids)
        self.advances.append(decision.accepted + 1)
        return decision


def load_profile(config, profile):
    """Experimental loader: the production RuntimeConfig contract stays FP32/eager."""
    verify_dflash_assets(Path(config.draft))
    dtype, attention = PROFILES[profile]
    torch.set_num_threads(config.threads)
    torch.set_float32_matmul_precision("highest")
    tokenizer = AutoTokenizer.from_pretrained(config.target, local_files_only=True, trust_remote_code=False)
    target = AutoModelForCausalLM.from_pretrained(config.target, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=dtype, attn_implementation=attention).to(config.device).eval()
    draft = AutoModel.from_pretrained(config.draft, local_files_only=True, trust_remote_code=True,
                                     torch_dtype=dtype, attn_implementation=attention).to(config.device).eval()
    return InferenceEngine(target, tokenizer, config, draft)


def execute(engine, record, arm, limit, repeat=0, *, profile=False):
    kind, block_size, rule, fast = ARMS[arm]
    policy = LengthPolicy(kind, rule)
    detail = None
    from memorydraft.runtime import engine as module
    native = module.generate_reuse

    def hook(*args, **kwargs):
        nonlocal detail
        kwargs.update(draft_block_size=block_size, incremental_stop=fast)
        if kind != "strict" or rule in {"recent", "budget"}:
            kwargs["verifier"] = policy
        if rule != "full":
            kwargs["verify_length"] = policy.choose
        detail = native(*args, **kwargs)
        return detail

    with patch.object(module, "generate_reuse", hook):
        result = engine.generate(replace(request_for(record), max_new_tokens=limit), "DFlash", profile=profile)
    return {**asdict(result), "arm": arm, "repeat": repeat, "max_new_tokens": limit,
            "quality": score(record, result.text), "rounds": detail.rounds,
            "asd": policy.policy.summary(), "selected_lengths": policy.selected}


@torch.inference_mode()
def draft_pair(engine, record, source, state):
    """Single forced token, fresh caches per arm; diagnostic, not online timing."""
    prompt = engine.tokenizer.apply_chat_template(list(request_for(record).messages), add_generation_prompt=True,
                                                 return_tensors="pt", **engine.config.template_options)[0].tolist()
    common = [*prompt, *source["token_ids"][:state["position"]]]
    rows = []
    for name, token in (("proposal", state["candidate"]), ("recovery", state["recovery"])):
        inputs = torch.tensor([[*common, token]], device=engine.target.device)
        result = generate_reuse(engine.target, engine.draft, inputs, max_new_tokens=128,
                                stop_token_ids=engine.stops, config=engine.reuse, incremental_stop=True)
        tokens = result.sequences[0, inputs.shape[1]:].tolist()
        first = result.rounds[0] if result.rounds else None
        rows.append(dict(request_id=record["id"], position=state["position"], arm=name, forced_token=token,
                         common_sha256=hashlib.sha256(json.dumps(common).encode()).hexdigest(),
                         first_round=first, new_tokens=len(tokens), rounds=len(result.rounds),
                         finish_reason="eos" if tokens and tokens[-1] in engine.stops else "length",
                         token_ids=tokens, timing="Fresh-prefix strict diagnostic; excluded from online speed"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    parser.add_argument("--attribution", action="store_true")
    args = parser.parse_args()
    allowed = TRANSFER_IDS if args.transfer else TASK_IDS
    if not args.task_ids or len(set(args.task_ids)) != len(args.task_ids) or set(args.task_ids) - set(allowed):
        parser.error("Unknown, duplicate or empty task IDs")
    if args.transfer and args.attribution:
        parser.error("Transfer and attribution are separate stages")
    if args.arms and (not (args.transfer or args.attribution) or len(set(args.arms)) != len(args.arms)):
        parser.error("Explicit unique arms are only allowed for the frozen transfer rule")
    args.output.mkdir(parents=True, exist_ok=False)
    before = protected(ROOT)
    info = dict(status="started", profile=args.profile, task_ids=args.task_ids, transfer=args.transfer,
                attribution=args.attribution,
                runner_sha256=file_hash(Path(__file__)), protocol_sha256=file_hash(ROOT / "docs/ASD_OPTIMIZATION_PROTOCOL.md"),
                original_sha256=before, loader="Experimental profile, production config remains FP32/eager")
    extension = ROOT / "docs/ASD_OPTIMIZATION_EXTENSION_PROTOCOL.md"
    if extension.exists():
        info["extension_protocol_sha256"] = file_hash(extension)
    write(args.output / "provenance.json", info)
    try:
        config = replace(RuntimeConfig.load(args.config), db=None, max_new_tokens=2048)
        engine = load_profile(config, args.profile)
        info.update(actual_dtype=str(engine.target.dtype), target_attention=engine.target.config._attn_implementation,
                    draft_attention=engine.draft.config._attn_implementation, stop_token_ids=engine.stops,
                    draft_block_size=engine.draft.block_size, torch=torch.__version__)
        if engine.draft.block_size != 16:
            raise ValueError("Frozen arms require native block size 16")
        write(args.output / "provenance.json", info)
        data = records()
        arms = args.arms or (list(ARMS) if args.profile == "fp32-eager" and not args.transfer else ["D-fast", "ASD-fast", "U-fast"])
        warm = dict(data[args.task_ids[0]])
        for arm in arms:
            execute(engine, warm, arm, 16)
        references = {}
        for q in args.task_ids:
            # Strict AR is a numeric compatibility control, not an assumed reference
            # across precision profiles. Its actual first mismatch is retained.
            ar = engine.generate(replace(request_for(data[q]), max_new_tokens=128), "AR")
            append(args.output / "ar.jsonl", asdict(ar))
            jobs = [(2048, a, 0) for a in arms]
            jobs += [(128, a, rep) for rep in range(2) for a in arms]
            random.Random(f"asd-optimization:{args.profile}:{q}").shuffle(jobs)
            for limit, arm, repeat in jobs:
                row = execute(engine, data[q], arm, limit, repeat)
                key = limit, arm
                if key in references and (references[key]["token_ids"], references[key]["asd"]) != (row["token_ids"], row["asd"]):
                    raise ValueError("Timed repeat changed output or ledger")
                references[key] = row
                append(args.output / "rows.jsonl", row)
            for a, b in (("D-old", "D-fast"), ("U-old", "U-fast")):
                if a in arms:
                    for limit in (128, 2048):
                        if (references[limit, a]["token_ids"], references[limit, a]["asd"]) != (references[limit, b]["token_ids"], references[limit, b]["asd"]):
                            raise ValueError("Incremental EOS changed the trajectory")
            print("completed", args.profile, q, len(jobs), flush=True)
            references.clear()
        if args.profile == "fp32-eager" and not args.transfer and not args.attribution:
            old = sources()
            for q in args.task_ids:
                for state in selected_states(old[q]):
                    for row in draft_pair(engine, data[q], old[q], state):
                        append(args.output / "draft-pairs.jsonl", row)
                print("draft-pairs", q, flush=True)
        if not args.transfer and not args.attribution:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                execute(engine, data[args.task_ids[0]], "U-fast", 128, profile=True)
            stats = [dict(key=e.key, cpu_us=e.cpu_time_total, device_us=getattr(e, "device_time_total", 0), count=e.count)
                     for e in prof.key_averages() if e.key.startswith("memorydraft::")]
            write(args.output / "profile.json", dict(instrumented=True, events=stats))
        info.update(status="complete", arms=arms)
    except Exception as error:
        info.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        info["original_unchanged"] = before == protected(ROOT)
        write(args.output / "provenance.json", info)
        if not info["original_unchanged"]:
            raise ValueError("Previous evidence changed")
    print("COMPLETE", args.profile, flush=True)


if __name__ == "__main__":
    main()
