"""Bounded execution-cost diagnosis; profiler and resident replay are not timings."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
import statistics
import time
from unittest.mock import patch

import torch

from memorydraft.backends.asd import ASDVerifier
from memorydraft.evaluation.asd_suite import BENCHMARKS, load_suite
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import POLICIES, protected
from scripts.run_asd_suite import execute
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
METHODS = tuple(POLICIES)
LIMIT = 128
REPEATS = 3
CAPTURE_ROUNDS = (0, 4, 12)
REPLAY_REPEATS = 30
IDENTITY = ("request_id", "method", "token_ids", "text", "prompt_sha256", "prompt_tokens",
            "snapshot_id", "asd", "rounds", "finish_reason")


def selected_records():
    records = load_suite(ROOT / "data/research/asd-suite", "development")
    return [r for b in BENCHMARKS for r in [r for r in records if r["benchmark"] == b][:2]]


def require_same(row, reference):
    changed = [k for k in IDENTITY if row[k] != reference[k]]
    if changed:
        raise ValueError("Execution diagnosis changed trajectory/input: " + ",".join(changed))


def interval_union(intervals):
    end = None
    total = 0.
    for lo, hi in sorted(intervals):
        if hi < lo:
            raise ValueError("Negative event interval")
        total += max(0., hi - max(lo, end if end is not None else lo))
        end = max(end, hi) if end is not None else hi
    return total


def scope_of(event):
    while event is not None:
        if event.name.startswith("memorydraft::"):
            return event.name.split("::", 1)[1]
        event = event.cpu_parent
    return "unscoped"


def profile_summary(events):
    """Use each CPU event's directly linked kernels once, never inclusive sums.

    CPU scope elapsed includes children and waits. GPU kernel duration sums can
    overlap; the device timeline union is separately reported. CUDA annotations
    are NOT kernels. CPU waits and GPU work must not be added together.
    """
    from torch.autograd import DeviceType

    events = list(events)
    annotation_names = {e.name for e in events if e.device_type == DeviceType.CUDA and
                        (e.name.startswith("memorydraft::") or getattr(e, "is_user_annotation", False))}
    scopes = defaultdict(lambda: dict(calls=0, host_elapsed_us=0., linked_device_us=0.,
                                      linked_kernels=0, sync_calls=0, sync_host_us=0.))
    operators = defaultdict(lambda: dict(calls=0, self_cpu_us=0., linked_device_us=0.))
    device_intervals = []
    linked = 0
    for event in events:
        annotation = event.name.startswith("memorydraft::") or getattr(event, "is_user_annotation", False)
        if event.device_type == DeviceType.CUDA:
            if not annotation:
                device_intervals.append((event.time_range.start, event.time_range.end))
            continue
        if event.device_type != DeviceType.CPU or event.is_async:
            continue
        group = scopes[scope_of(event)]
        if event.name.startswith("memorydraft::"):
            group["calls"] += 1
            group["host_elapsed_us"] += event.cpu_time_total
        kernels = [k for k in event.kernels if k.name not in annotation_names]
        duration = sum(k.duration for k in kernels)
        group["linked_device_us"] += duration
        group["linked_kernels"] += len(kernels)
        linked += len(kernels)
        if event.name.startswith(("cuda", "cu")) and "Synchronize" in event.name:
            group["sync_calls"] += 1
            group["sync_host_us"] += event.cpu_time_total
        op = operators[event.name]
        op["calls"] += 1
        op["self_cpu_us"] += event.self_cpu_time_total
        op["linked_device_us"] += duration
    if not device_intervals or not linked:
        raise ValueError("Missing CUPTI device events/kernel links; do not fabricate GPU attribution")
    return dict(scopes=dict(scopes), device_event_count=len(device_intervals), linked_kernel_count=linked,
                device_active_union_us=interval_union(device_intervals),
                device_duration_sum_us=sum(hi-lo for lo, hi in device_intervals),
                device_span_us=max(hi for _, hi in device_intervals)-min(lo for lo, _ in device_intervals),
                top_cpu_operators=sorted((dict(name=k, **v) for k, v in operators.items()),
                                         key=lambda v: v["self_cpu_us"], reverse=True)[:20],
                top_device_operators=sorted((dict(name=k, **v) for k, v in operators.items()),
                                            key=lambda v: v["linked_device_us"], reverse=True)[:20])


class CaptureVerifier(ASDVerifier):
    """Copies at most three real inputs in a separate, untimed capture run."""
    def __init__(self, config, *, record=False):
        super().__init__(config, record=record)
        self.states = []

    def decide(self, proposal, logits, *, stop_token_ids=()):
        state = None
        if self.blocks in CAPTURE_ROUNDS:
            state = dict(round=self.blocks, spent_before=self.spent,
                         proposal=proposal.detach().clone(), logits=logits.detach().clone(),
                         stops=tuple(stop_token_ids))
        result = super().decide(proposal, logits, stop_token_ids=stop_token_ids)
        if state is not None:
            state["expected"] = result
            self.states.append(state)
        return result


def strict_core(proposal, logits):
    # Receives K logits, excluding the bonus position. This is a strict-prefix
    # core reference, not a replacement for the complete native K+1 decision.
    top = logits.argmax(-1)
    return int((proposal == top).cumprod(0).sum().item())


@torch.inference_mode()
def resident_replay(state, config):
    proposal, logits = state["proposal"], state["logits"]
    device = logits.device
    samples = {"policy": [], "strict_core": [], "sync_control": []}
    for repeat in range(REPLAY_REPEATS + 5):
        order = list(samples)
        random.Random(f"replay:{repeat}").shuffle(order)
        for kind in order:
            policy = ASDVerifier(config)
            policy.spent = state["spent_before"]
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            if kind == "policy":
                result = policy.decide(proposal, logits, stop_token_ids=state["stops"])
            elif kind == "strict_core":
                result = strict_core(proposal, logits)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if kind == "policy" and result != state["expected"]:
                raise ValueError("Resident policy replay changed the captured decision")
            if kind == "strict_core" and result != state["expected"].strict_accepted:
                raise ValueError("Resident strict replay changed the captured prefix")
            if repeat >= 5:
                samples[kind].append(elapsed)
    return dict(round=state["round"], spent_before=state["spent_before"],
                proposals=proposal.numel(), expected=vars(state["expected"]),
                median_seconds={k: statistics.median(v) for k, v in samples.items()}, samples_seconds=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    records = selected_records()[args.shard::2]
    old = {(r["request_id"], r["method"]): r for shard in (0, 1)
           for r in read_rows(ROOT / f"results/asd-block-suite/part-{shard}/rows.jsonl")
           if r["stage"] == "prefix" and r["repeat"] == 0 and r["experimental_block_size"] == 16}
    before = protected(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    info = dict(status="started", task_ids=[r["id"] for r in records], shard=args.shard,
                max_new_tokens=LIMIT, repeats=REPEATS, capture_rounds=CAPTURE_ROUNDS,
                replay_repeats=REPLAY_REPEATS, original_sha256=before)
    write(args.output / "provenance.json", info)
    counts = dict(timed=0, profile=0, capture=0, replay_states=0)
    try:
        engine = load_engine(replace(RuntimeConfig.load(args.config), db=None, max_new_tokens=4096))
        if engine.draft.block_size != 16 or engine.target.device.type != "cuda":
            raise ValueError("Requires native b16 checkpoint and CUDA")
        info.update(runtime=provenance(engine), stops=engine.stops)
        write(args.output / "provenance.json", info)
        for method in METHODS:
            execute(engine, dict(benchmark="gsm8k", id="warmup", question="What is 3 plus 4?", answer="#### 7"), method, 32)
        for record in records:
            seen = {}
            # Timings precede ALL profiler/capture work for this request.
            for repeat in range(REPEATS):
                order = list(METHODS)
                random.Random(f"execution:{record['id']}:{repeat}").shuffle(order)
                for method in order:
                    row = execute(engine, record, method, LIMIT)
                    require_same(row, old[record["id"], method])
                    if method in seen:
                        require_same(row, seen[method])
                    seen[method] = row
                    row.update(stage="timed", repeat=repeat)
                    append(args.output / "rows.jsonl", row)
                    counts["timed"] += 1
            for method in METHODS:
                engine.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    with torch.profiler.record_function("memorydraft::request"):
                        row = execute(engine, record, method, LIMIT, profile=True)
                require_same(row, seen[method])
                row["instrumented_latency_seconds"] = row.pop("latency_seconds")
                row.update(stage="profile", profile=profile_summary(prof.events()))
                append(args.output / "profiles.jsonl", row)
                counts["profile"] += 1
                del prof, row
            from scripts import run_asd_suite as suite
            for method in METHODS[1:]:
                captured = []

                def factory(*a, **kw):
                    policy = CaptureVerifier(*a, **kw)
                    captured.append(policy)
                    return policy

                with patch.object(suite, "ASDVerifier", factory):
                    row = execute(engine, record, method, LIMIT)
                require_same(row, seen[method])
                policy = captured[0]
                row["instrumented_latency_seconds"] = row.pop("latency_seconds")
                row.update(stage="capture", resident_replays=[resident_replay(s, POLICIES[method]) for s in policy.states])
                append(args.output / "replays.jsonl", row)
                counts["capture"] += 1
                counts["replay_states"] += len(policy.states)
                del captured, policy, row
            print("completed", record["benchmark"], record["id"], json.dumps(counts), flush=True)
            info.update(counts=counts)
            write(args.output / "provenance.json", info)
        info.update(status="complete", counts=counts)
    except Exception as error:
        info.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        info["original_unchanged"] = before == protected(ROOT)
        write(args.output / "provenance.json", info)
        if not info["original_unchanged"]:
            raise ValueError("Previous evidence changed")


if __name__ == "__main__":
    main()
