"""Development-only interventions on the actual Utility failure trajectory."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import torch

from memorydraft.backends.asd import ASDDecision, ASDVerifier
from memorydraft.backends.dflash_reuse import generate_reuse
from memorydraft.evaluation.asd_budget import load_tasks, request_for, score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import POLICIES, protected
from scripts.run_asd_confidence_probe import ObservedVerifier, branch_prefix
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = ("gsm8k:21", "math500:7")
CONFIG = POLICIES["ASD-Utility"]
MAX_TOKENS, PREFIX_TOKENS = 2048, 128


def sources():
    path = ROOT / "results/asd-confidence/mixed-b/mechanism/rows.jsonl"
    rows = {r["request_id"]: r for r in read_rows(path) if r["method"] == "ASD-Utility"}
    if not set(TASK_IDS) <= rows.keys():
        raise ValueError("Missing frozen Utility references")
    return {q: rows[q] for q in TASK_IDS}


def exception_states(row):
    """Include later accepted mismatches in a block, using the realized prefix."""
    states = []
    spent = 0.
    for block, event in enumerate(row["events"]):
        within = 0.
        for i in range(event["accepted"]):
            if not event["mismatch"][i]:
                continue
            regret = event["regrets"][i]
            position = event["generated_before"] + i
            run = next((j - i for j in range(i + 1, len(event["mismatch"])) if event["mismatch"][j]),
                       len(event["mismatch"]) - i)
            if i + 1 not in event["proposal_eos"]:
                if row["token_ids"][position] != event["proposal_ids"][i]:
                    raise ValueError("Exception is not on the committed trajectory")
                states.append(dict(block=block, index=i, position=position,
                                   candidate=event["proposal_ids"][i], recovery=event["target_ids"][i],
                                   regret=regret, run=run, spent_before_block=spent,
                                   spent_before_token=spent + within,
                                   top_probability=event["top_probability"][i],
                                   proposal_probability=event["proposal_probability"][i],
                                   pair_mass=event["pair_mass"][i], target_rank=event["target_rank"][i],
                                   draft_probability=event["draft_probability"][i]))
            within += regret
        if abs(within - event["regret"]) > 1e-8 or abs(spent + within - event["spent"]) > 1e-8:
            raise ValueError("Source ledger does not match accepted exceptions")
        spent += within
    return states


def selected_states(row):
    states = exception_states(row)
    if row["request_id"] == "gsm8k:21":
        return states
    if row["request_id"] != "math500:7" or not states:
        raise ValueError("Unexpected development control")
    chosen = (states[0], max(states, key=lambda s: (s["regret"], -s["position"])), states[-1])
    return sorted({s["position"]: s for s in chosen}.values(), key=lambda s: s["position"])


class RepairUtilityVerifier(ASDVerifier):
    """Reject one recorded exception in place; retain all earlier budget charges."""
    def __init__(self, state, source_event):
        super().__init__(CONFIG, record=True)
        self.state, self.source_event = state, source_event
        self.intervention = None

    def decide(self, tokens, logits, *, stop_token_ids=()):
        block, before = self.blocks, self.spent
        decision = super().decide(tokens, logits, stop_token_ids=stop_token_ids)
        if block != self.state["block"]:
            return decision
        event, i = self.events[-1], self.state["index"]
        source = self.source_event
        if (tokens.tolist() != source["proposal_ids"] or logits.argmax(-1).tolist() != source["target_ids"]
                or abs(before - self.state["spent_before_block"]) > 1e-8
                or decision.accepted != source["accepted"] or i >= decision.accepted
                or not event["mismatch"][i]
                or max(abs(a - b) for a, b in zip(event["regrets"], source["regrets"])) > 1e-3):
            raise ValueError("Repair did not reach the exact source decision")
        charge = sum(r for r, m in zip(event["regrets"][:i], event["mismatch"][:i]) if m)
        relaxed = sum(event["mismatch"][:i])
        self.spent = before + charge
        self.relaxed += relaxed - decision.relaxed
        self.shortened += int(i < event["feasible_accepted"]) - int(decision.accepted < event["feasible_accepted"])
        self.foregone += decision.accepted - i
        self.intervention = dict(block=block, original_accepted=decision.accepted, accepted=i,
                                 original_charge=decision.regret, charge=charge, spent=self.spent)
        event.update(accepted=i, relaxed=relaxed, regret=charge, spent=self.spent,
                     repair=dict(self.intervention))
        return ASDDecision(i, decision.strict_accepted, relaxed, charge)


def validate_repair(tokens, source_tokens, state):
    p = state["position"]
    if tokens[:p] != source_tokens[:p] or len(tokens) <= p or tokens[p] != state["recovery"]:
        raise ValueError("Repair changed the prefix or failed to emit recovery")


def execute_policy(engine, record, source, state=None):
    policy = ASDVerifier(CONFIG, record=True) if state is None else RepairUtilityVerifier(
        state, source["events"][state["block"]])
    observer = ObservedVerifier(policy)
    from memorydraft.runtime import engine as module
    native, detail = module.generate_reuse, None

    def hook(*args, **kwargs):
        nonlocal detail
        detail = native(*args, **kwargs, verifier=observer, on_draft=observer.on_draft)
        return detail

    with patch.object(module, "generate_reuse", hook):
        result = engine.generate(replace(request_for(record), max_new_tokens=MAX_TOKENS), "DFlash")
    for event, rd in zip(policy.events, detail.rounds):
        event["generated_before"] = rd["generated_before"]
    if (result.prompt_sha256, result.snapshot_id) != (source["prompt_sha256"], source["snapshot_id"]):
        raise ValueError("Target prompt or history changed")
    if state is None:
        if result.token_ids != source["token_ids"] or policy.summary() != source["asd"]:
            raise ValueError("Utility baseline did not reproduce")
    else:
        validate_repair(result.token_ids, source["token_ids"], state)
        if policy.intervention is None:
            raise ValueError("Repair was not triggered")
    row = {**asdict(result), "state": state, "quality": score(record, result.text),
           "asd": policy.summary(), "events": policy.events, "rounds": detail.rounds,
           "intervention": getattr(policy, "intervention", None)}
    row["instrumented_seconds"] = row.pop("latency_seconds")
    return row


@torch.inference_mode()
def strict_branches(engine, record, source, state):
    request = request_for(record)
    prompt = engine.tokenizer.apply_chat_template(list(request.messages), add_generation_prompt=True,
                                                 return_tensors="pt", **engine.config.template_options)[0].tolist()
    common = prompt + source["token_ids"][:state["position"]]
    logits = engine.target(torch.tensor([common], device=engine.target.device),
                           use_cache=False, logits_to_keep=1).logits[0, -1]
    gap = float(logits.max().double() - logits[state["candidate"]].double())
    if int(logits.argmax()) != state["recovery"] or abs(gap - state["regret"]) > 1e-3:
        raise ValueError("Same-prefix independent Target scores are unstable")
    probabilities = logits.float().softmax(-1)
    entropy = float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())
    jobs = [(h, a, t) for h in ("natural", "prefix")
            for a, t in (("proposal", state["candidate"]), ("recovery", state["recovery"]))]
    random.Random(f"utility-causal:{record['id']}:{state['position']}").shuffle(jobs)
    rows = []
    for horizon, arm, forced in jobs:
        inputs = branch_prefix(prompt, source["token_ids"], state, forced)
        prefix = inputs[len(prompt):]
        limit = MAX_TOKENS - len(prefix) if horizon == "natural" else min(PREFIX_TOKENS, MAX_TOKENS - len(prefix))
        engine.synchronize()
        started = time.perf_counter()
        if forced in engine.stops or limit == 0:
            continuation, rounds = [], []
        else:
            detail = generate_reuse(engine.target, engine.draft,
                                    torch.tensor([inputs], device=engine.target.device), max_new_tokens=limit,
                                    stop_token_ids=engine.stops, config=engine.reuse)
            continuation, rounds = detail.sequences[0, len(inputs):].tolist(), detail.rounds
        engine.synchronize()
        elapsed = time.perf_counter() - started
        tokens = prefix + continuation
        text = engine.tokenizer.decode(tokens, skip_special_tokens=True)
        rows.append(dict(request_id=record["id"], state=state, horizon=horizon, arm=arm,
                         forced_token=forced, common_prefix_sha256=hashlib.sha256(json.dumps(common).encode()).hexdigest(),
                         replay_argmax=int(logits.argmax()), replay_regret=gap, target_entropy=entropy,
                         token_ids=tokens, text=text, quality=score(record, text), continuation_limit=limit,
                         continuation_tokens=len(continuation), rounds=rounds, continuation_rounds=len(rounds),
                         finish_reason="eos" if tokens and tokens[-1] in engine.stops else "length",
                         recompute_seconds=elapsed,
                         timing="Diagnostic re-prefill plus strict continuation; one repeat, not online policy speed"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    args = parser.parse_args()
    references = sources()
    source = references[args.task_id]
    states = selected_states(source)
    record = next(r for r in load_tasks(ROOT / "data/research/asd-budget/records.json") if r["id"] == args.task_id)
    original = protected(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    engine = load_engine(replace(RuntimeConfig.load(args.config), max_new_tokens=MAX_TOKENS, db=None,
                                 retrieval="suffix", prefix_risk={}))
    info = {**provenance(engine), "status": "started", "task_id": args.task_id,
            "config": asdict(CONFIG), "states": states, "original_sha256": original,
            "runner_sha256": file_hash(Path(__file__)), "stop_token_ids": sorted(engine.stops),
            "protocol_sha256": file_hash(ROOT / "docs/ASD_UTILITY_CAUSAL_PROTOCOL.md"),
            "data_sha256": file_hash(ROOT / "data/research/asd-budget/records.json")}
    # The generic engine timing description is superseded by each diagnostic row.
    info["timing"] = "Instrumented whole-policy runs and diagnostic branch re-prefills; no online speed claim"
    write(args.output / "provenance.json", info)
    try:
        baseline = execute_policy(engine, record, source)
        append(args.output / "policies.jsonl", baseline)
        print("reproduced baseline", args.task_id, baseline["quality"], flush=True)
        for state in states:
            repaired = execute_policy(engine, record, source, state)
            append(args.output / "policies.jsonl", repaired)
            for row in strict_branches(engine, record, source, state):
                append(args.output / "branches.jsonl", row)
            print("completed", args.task_id, state["position"], repaired["quality"], flush=True)
        info.update(status="complete", policy_rows=1 + len(states), branch_rows=4 * len(states))
    except Exception as error:
        info.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        info["original_unchanged"] = original == protected(ROOT)
        write(args.output / "provenance.json", info)
        engine.close()
    if not info["original_unchanged"]:
        raise ValueError("Published evidence changed")


if __name__ == "__main__":
    main()
