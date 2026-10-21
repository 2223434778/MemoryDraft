"""Regional MLP compilation feasibility; numerical profiles stay separate."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import statistics
import time
from unittest.mock import patch

import torch

from memorydraft.backends.asd_cpu import CPUASDVerifier
from memorydraft.backends.regional_compile import RegionalMLP
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts import run_asd_suite as suite
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import POLICIES, protected
from scripts.run_asd_execution_probe import require_same, selected_records
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
METHODS = tuple(POLICIES)
PROFILES = ("eager", "mlp-inductor")
LIMIT = 128
REPEATS = 3
REPLAY_REPEATS = 20


def previous_rows():
    result = {(r["request_id"], r["method"]): r for i in (0, 1)
              for r in read_rows(ROOT/f"results/asd-cpu/part-{i}/rows.jsonl")
              if r["repeat"] == 0 and r["implementation"] in {"cpu", "native"}}
    result.update({(r["request_id"], "AR"): r for i in (0, 1)
                   for r in read_rows(ROOT/f"results/asd-block-suite/part-{i}/rows.jsonl")
                   if r["stage"] == "prefix" and r["repeat"] == 0 and r["method"] == "AR"})
    return result


def timing_order(q, repeat, methods=METHODS, profiles=PROFILES):
    rng = random.Random(f"asd-compile:{q}:{repeat}")
    methods = list(methods)
    rng.shuffle(methods)
    order = []
    for method in methods:
        pair = list(profiles)
        rng.shuffle(pair)
        order.extend((method, p) for p in pair)
    return order


def trajectory_comparison(eager, compiled):
    for k in ("request_id", "method", "prompt_sha256", "prompt_tokens", "snapshot_id"):
        if eager[k] != compiled[k]:
            raise ValueError("Compiler profile changed the Target input: "+k)
    a, b = eager["token_ids"], compiled["token_ids"]
    common = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return dict(token_equal=a == b, first_token_difference=None if a == b else common,
                round_equal=eager["rounds"] == compiled["rounds"], ledger_equal=eager["asd"] == compiled["asd"],
                length_equal=len(a) == len(b), eos_equal=eager["finish_reason"] == compiled["finish_reason"])


def execute(engine, compiler, record, method, profile, limit=LIMIT):
    instances = []
    setup = 0.

    def factory(*a, **kw):
        nonlocal setup
        started = time.perf_counter()
        policy = CPUASDVerifier(*a, **kw)
        setup += time.perf_counter()-started
        instances.append(policy)
        return policy

    switch_before = compiler.switch_seconds
    graphs_before = len(compiler.backend.events)
    with compiler.activate(profile), patch.object(suite, "ASDVerifier", factory):
        row = suite.execute(engine, record, method, limit)
    switch = compiler.switch_seconds-switch_before
    row.update(profile=profile, engine_latency_seconds=row["latency_seconds"],
               policy_setup_seconds=setup, profile_switch_seconds=switch,
               graphs_before=graphs_before, graphs_after=len(compiler.backend.events),
               peak_reserved_bytes=torch.cuda.max_memory_reserved(engine.target.device))
    row["latency_seconds"] = row["engine_latency_seconds"]+setup+switch
    if method not in {"AR", "DFlash"}:
        row["cpu_execution"] = instances[0].execution_summary()
    if row["peak_gpu_bytes"] > 46*2**30:
        raise RuntimeError("Compiled profile exceeded the 46 GiB allocation budget")
    return row


@contextmanager
def capture_inputs(compiler):
    selected = [next((n, m) for n, m in compiler.regions if n.startswith(owner+".")) for owner in ("target", "draft")]
    states = []
    handles = []
    for name, module in selected:
        calls = [0]

        def hook(m, args, *, name=name, calls=calls):
            index = calls[0]
            calls[0] += 1
            maximum = 2 if name.startswith("target.") else 1
            if index < maximum:
                x = args[0]
                states.append(dict(region=name, call=index, module=m, input=x.detach().clone(),
                                   original_stride=list(x.stride())))

        handles.append(module.register_forward_pre_hook(hook))
    try:
        yield states
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def replay_mlp(compiler, state, profiles=PROFILES):
    x, module = state["input"], state["module"]
    eager = type(module).forward
    compiled = compiler.compiled[type(module)]
    a, b = eager(module, x), compiled(module, x)
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    if not finite:
        raise ValueError("Non-finite compiled region output")
    delta = (a-b).abs()
    numeric = dict(bitwise_equal=torch.equal(a, b), max_abs=float(delta.max()),
                   rms=float((delta.double().square().mean()).sqrt()),
                   max_relative_floor_1e_6=float((delta/a.abs().clamp_min(1e-6)).max()))
    samples = {p: [] for p in profiles}
    for repeat in range(REPLAY_REPEATS+5):
        order = list(profiles)
        random.Random(f"mlp-replay:{repeat}").shuffle(order)
        for profile in order:
            fn = eager if profile == "eager" else compiled
            torch.cuda.synchronize(x.device)
            started = time.perf_counter()
            fn(module, x)
            torch.cuda.synchronize(x.device)
            elapsed = time.perf_counter()-started
            if repeat >= 5:
                samples[profile].append(elapsed)
    return dict(region=state["region"], call=state["call"], input_shape=list(x.shape),
                original_stride=state["original_stride"], replay_stride=list(x.stride()), numeric=numeric,
                samples_seconds=samples, median_seconds={p: statistics.median(v) for p, v in samples.items()})


def main(*, compiler_type=RegionalMLP, profiles=PROFILES, capture=capture_inputs,
         replay_key="mlp_replays", state_key="mlp_states"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    records = selected_records()[args.shard::2]
    previous = previous_rows()
    before = protected(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    info = dict(status="started", shard=args.shard, task_ids=[r["id"] for r in records], original_sha256=before,
                repeats=REPEATS, max_new_tokens=LIMIT, backend_graph_budget=8, backend_seconds_budget=300,
                cache_dirs={k: os.environ[k] for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")})
    write(args.output/"provenance.json", info)
    counts = dict(warmup=0, preflight=0, timed=0, diagnostics=0, **{state_key: 0})
    compiler = None
    try:
        engine = load_engine(replace(RuntimeConfig.load(args.config), db=None, max_new_tokens=4096))
        if engine.draft.block_size != 16 or engine.target.dtype != torch.float32:
            raise ValueError("Expected frozen native16 FP32 models")
        info.update(runtime=provenance(engine), stop_token_ids=engine.stops)
        started = time.perf_counter()
        compiler = compiler_type(engine.target, engine.draft)
        info["controller_setup_seconds"] = time.perf_counter()-started
        warmup = dict(benchmark="gsm8k", id="warmup", question="What is 3 plus 4?", answer="#### 7")
        started = time.perf_counter()
        for profile in profiles:
            for method in ("AR", *METHODS):
                row = execute(engine, compiler, warmup, method, profile, 32)
                row.update(stage="warmup", repeat=0)
                append(args.output/"warmup.jsonl", row)
                counts["warmup"] += 1
        seen = {}
        for record in records:
            for method, profile in timing_order(record["id"], -1, ("AR", *METHODS), profiles):
                row = execute(engine, compiler, record, method, profile)
                if profile == "eager":
                    require_same(row, previous[record["id"], method])
                row.update(stage="preflight", repeat=0)
                seen[record["id"], method, profile] = row
                append(args.output/"preflight.jsonl", row)
                counts["preflight"] += 1
            print("prepared", record["id"], len(compiler.backend.events), flush=True)
        info["cold_preparation_wall_seconds"] = time.perf_counter()-started
        cache_bytes = sum(p.stat().st_size for folder in info["cache_dirs"].values() for p in Path(folder).rglob("*") if p.is_file())
        if cache_bytes > 4*2**30:
            raise RuntimeError("Compiler caches exceeded the 4 GiB preparation budget")
        info.update(cache_bytes_after_preparation=cache_bytes, compiler_after_preparation=compiler.summary())
        write(args.output/"provenance.json", info)
        for record in records:
            for repeat in range(REPEATS):
                for slot, (method, profile) in enumerate(timing_order(record["id"], repeat, profiles=profiles)):
                    row = execute(engine, compiler, record, method, profile)
                    require_same(row, seen[record["id"], method, profile])
                    row.update(stage="timed", repeat=repeat, slot=slot)
                    append(args.output/"rows.jsonl", row)
                    counts["timed"] += 1
            print("timed", record["id"], json.dumps(counts), flush=True)
        # Hooks and same-input arithmetic checks are separate from all timings.
        for record in records:
            with capture(compiler) as states:
                row = execute(engine, compiler, record, "DFlash", "eager")
            require_same(row, seen[record["id"], "DFlash", "eager"])
            row["instrumented_latency_seconds"] = row.pop("latency_seconds")
            row.update(stage="diagnostic", **{replay_key: [replay_mlp(compiler, s, profiles) for s in states]})
            append(args.output/"diagnostics.jsonl", row)
            counts["diagnostics"] += 1
            counts[state_key] += len(states)
            del row, states
        info.update(status="complete", counts=counts, compiler=compiler.summary())
    except Exception as error:
        info.update(status="failed", counts=counts, error=type(error).__name__+": "+str(error))
        if compiler is not None:
            info["compiler"] = compiler.summary()
        raise
    finally:
        info["original_unchanged"] = before == protected(ROOT)
        write(args.output/"provenance.json", info)
        if not info["original_unchanged"]:
            raise ValueError("Previous evidence changed")


if __name__ == "__main__":
    main()
