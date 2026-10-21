"""Paired original/compact-CPU ASD execution; no change to Target or decode."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import statistics
import time
from unittest.mock import patch

import torch

from memorydraft.backends.asd import ASDVerifier
from memorydraft.backends.asd_cpu import CPUASDVerifier
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts import run_asd_suite as suite
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import POLICIES, protected
from scripts.run_asd_execution_probe import CaptureVerifier, require_same, selected_records
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
METHODS = tuple(POLICIES)[1:]
REPEATS = 5
LIMIT = 128
IMPLEMENTATIONS = ("tensor", "cpu")
ARMS = (("DFlash", "native"),) + tuple((m, impl) for m in METHODS for impl in IMPLEMENTATIONS)
REPLAY_REPEATS = 30


def timing_order(request_id, repeat):
    rng = random.Random(f"asd-cpu:{request_id}:{repeat}")
    methods = list(METHODS)
    rng.shuffle(methods)
    groups = []
    for method in methods:
        implementations = list(IMPLEMENTATIONS)
        rng.shuffle(implementations)
        groups.append([(method, impl) for impl in implementations])
    groups.insert(rng.randrange(len(groups)+1), [("DFlash", "native")])
    return [arm for group in groups for arm in group]


def execute(engine, record, method, implementation):
    if (method, implementation) not in ARMS:
        raise ValueError("Unknown execution arm")
    instances = []
    setup_seconds = 0.

    def factory(*args, **kwargs):
        nonlocal setup_seconds
        started = time.perf_counter()
        policy = (CPUASDVerifier if implementation == "cpu" else ASDVerifier)(*args, **kwargs)
        setup_seconds += time.perf_counter()-started
        instances.append(policy)
        return policy

    with patch.object(suite, "ASDVerifier", factory):
        row = suite.execute(engine, record, method, LIMIT)
    row.update(implementation=implementation)
    row["engine_latency_seconds"] = row["latency_seconds"]
    row["policy_setup_seconds"] = setup_seconds
    row["latency_seconds"] += setup_seconds
    if implementation == "cpu":
        row["cpu_execution"] = instances[0].execution_summary()
    return row


class AuditVerifier(CaptureVerifier):
    def __init__(self, config, *, record=False):
        super().__init__(config, record=True)
        self.shadow = CPUASDVerifier(config)

    def decide(self, tokens, logits, *, stop_token_ids=()):
        result = super().decide(tokens, logits, stop_token_ids=stop_token_ids)
        actual = self.shadow.decide(tokens, logits, stop_token_ids=stop_token_ids)
        if result != actual or self.summary() != self.shadow.summary():
            raise ValueError("CPU shadow changed actual decision or persistent ledger")
        self.events[-1].update(proposal_token_ids=tokens.tolist(), target_top_token_ids=logits.argmax(-1).tolist())
        return result


@torch.inference_mode()
def resident_replay(state, config):
    samples = {impl: [] for impl in IMPLEMENTATIONS}
    for repeat in range(REPLAY_REPEATS+5):
        order = list(IMPLEMENTATIONS)
        random.Random(f"asd-cpu-replay:{repeat}").shuffle(order)
        for impl in order:
            verifier = (CPUASDVerifier if impl == "cpu" else ASDVerifier)(config)
            verifier.spent = state["spent_before"]
            torch.cuda.synchronize(state["logits"].device)
            started = time.perf_counter()
            result = verifier.decide(state["proposal"], state["logits"], stop_token_ids=state["stops"])
            torch.cuda.synchronize(state["logits"].device)
            elapsed = time.perf_counter()-started
            if result != state["expected"]:
                raise ValueError("Resident replay changed original captured decision")
            if repeat >= 5:
                samples[impl].append(elapsed)
    return dict(round=state["round"], spent_before=state["spent_before"], expected=asdict(state["expected"]),
                proposals=state["proposal"].numel(), samples_seconds=samples,
                median_seconds={k: statistics.median(v) for k, v in samples.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    records = selected_records()[args.shard::2]
    previous = {(r["request_id"], r["method"]): r for i in (0, 1)
                for r in read_rows(ROOT/f"results/asd-execution/part-{i}/rows.jsonl") if r["repeat"] == 0}
    before = protected(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    info = dict(status="started", task_ids=[r["id"] for r in records], shard=args.shard,
                repeats=REPEATS, max_new_tokens=LIMIT, arms=ARMS, original_sha256=before)
    write(args.output/"provenance.json", info)
    counts = dict(timed=0, capture=0, shadow_blocks=0, replay_states=0)
    try:
        engine = load_engine(replace(RuntimeConfig.load(args.config), db=None, max_new_tokens=4096))
        if engine.draft.block_size != 16 or engine.target.dtype != torch.float32:
            raise ValueError("Requires the frozen FP32 native b16 checkpoint")
        info.update(runtime=provenance(engine), stop_token_ids=engine.stops)
        write(args.output/"provenance.json", info)
        warmup = dict(benchmark="gsm8k", id="warmup", question="What is 3 plus 4?", answer="#### 7")
        for method, impl in ARMS:
            execute(engine, warmup, method, impl)
        for record in records:
            for repeat in range(REPEATS):
                for slot, (method, impl) in enumerate(timing_order(record["id"], repeat)):
                    row = execute(engine, record, method, impl)
                    require_same(row, previous[record["id"], method])
                    row.update(stage="timed", repeat=repeat, slot=slot)
                    append(args.output/"rows.jsonl", row)
                    counts["timed"] += 1
            # No shadow comparison or replay enters timed engine execution.
            for method in METHODS:
                instances = []

                def factory(*a, **kw):
                    verifier = AuditVerifier(*a, **kw)
                    instances.append(verifier)
                    return verifier

                with patch.object(suite, "ASDVerifier", factory):
                    row = suite.execute(engine, record, method, LIMIT)
                require_same(row, previous[record["id"], method])
                policy = instances[0]
                row["instrumented_latency_seconds"] = row.pop("latency_seconds")
                row.update(stage="capture", events=policy.events, cpu_execution=policy.shadow.execution_summary(),
                           resident_replays=[resident_replay(s, POLICIES[method]) for s in policy.states])
                append(args.output/"mechanism.jsonl", row)
                counts["capture"] += 1
                counts["shadow_blocks"] += len(policy.events)
                counts["replay_states"] += len(policy.states)
                del policy, instances, row
            info.update(counts=counts)
            write(args.output/"provenance.json", info)
            print("completed", record["id"], json.dumps(counts), flush=True)
        info.update(status="complete", counts=counts)
    except Exception as error:
        info.update(status="failed", error=type(error).__name__+": "+str(error))
        raise
    finally:
        info["original_unchanged"] = before == protected(ROOT)
        write(args.output/"provenance.json", info)
        if not info["original_unchanged"]:
            raise ValueError("Previous evidence changed")


if __name__ == "__main__":
    main()
