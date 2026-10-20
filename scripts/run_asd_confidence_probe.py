"""Frozen four-task confidence pricing and single-exception intervention diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.backends.asd_confidence import ConfidenceUtilityVerifier, token_confidence
from memorydraft.backends.dflash_reuse import generate_reuse
from memorydraft.evaluation.asd_budget import load_tasks, request_for, score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import POLICIES as OLD_POLICIES, compare_previous, protected
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = ("math500:1", "math500:4", "math500:7", "gsm8k:21")
POLICIES = {**OLD_POLICIES,
            "Utility-Price1": ASDConfig(budget=8, selection="utility", regret_price=1),
            "Utility-Price3": ASDConfig(budget=8, selection="utility", regret_price=3),
            "Confidence-Utility": ASDConfig(budget=8, selection="utility", regret_price=2)}
ALPHA = .5


def make_policy(method, *, record=False):
    if method == "Confidence-Utility":
        return ConfidenceUtilityVerifier(POLICIES[method], alpha=ALPHA, record=record)
    return ASDVerifier(POLICIES[method], record=record)


class ObservedVerifier:
    """Mechanism-only observer; timed policies do not compute these diagnostics."""
    def __init__(self, policy):
        self.policy = policy
        self.proposal = self.draft_probability = None

    def on_draft(self, confirmed, proposed, logits, limit):
        self.proposal = proposed
        tokens = torch.tensor(proposed, device=logits.device)
        self.draft_probability = token_confidence(tokens, logits)["proposal_probability"].tolist()

    def decide(self, tokens, logits, *, stop_token_ids=()):
        if self.proposal != tokens.tolist():
            raise ValueError("Observed draft and verified proposal differ")
        decision = self.policy.decide(tokens, logits, stop_token_ids=stop_token_ids)
        probabilities = {k: v.tolist() for k, v in token_confidence(tokens, logits).items()}
        top_ids = logits.argmax(-1).tolist()
        proposed = logits.gather(1, tokens[:, None]).squeeze(1)
        ranks = (logits > proposed[:, None]).sum(-1).add(1).tolist()
        event = self.policy.events[-1]
        event.update(proposal_ids=tokens.tolist(), target_ids=top_ids,
                     proposal_eos=[i + 1 for i, t in enumerate(self.proposal) if t in stop_token_ids],
                     draft_probability=self.draft_probability, target_rank=ranks, **probabilities)
        self.proposal = self.draft_probability = None
        return decision


def candidate_states(row):
    states = []
    for block, event in enumerate(row["events"]):
        first = next((i for i, mismatch in enumerate(event["mismatch"]) if mismatch), None)
        if first is None or first >= event["accepted"]:
            continue
        position = event["generated_before"] + first
        if position >= 512 or first + 1 in event["proposal_eos"]:
            continue
        run = next((j - first for j in range(first + 1, len(event["mismatch"])) if event["mismatch"][j]),
                   len(event["mismatch"]) - first)
        states.append(dict(block=block, index=first, position=position,
                           pair_mass=event["pair_mass"][first], regret=event["regrets"][first], run=run,
                           candidate=event["proposal_ids"][first], recovery=event["target_ids"][first]))
    return states


def select_states(row):
    states = candidate_states(row)
    if not states:
        return dict(matched=False, candidates=0, states=[])
    pairs = []
    for i, a in enumerate(states):
        for b in states[i + 1:]:
            gap = abs(a["pair_mass"] - b["pair_mass"])
            if abs(a["regret"] - b["regret"]) <= .25 and abs(a["run"] - b["run"]) <= 2 and gap >= .1:
                pairs.append((a, b))
    if pairs:
        a, b = min(pairs, key=lambda p: (-abs(p[0]["pair_mass"] - p[1]["pair_mass"]),
                                        p[0]["position"], p[1]["position"]))
        chosen, matched = sorted((a, b), key=lambda s: s["pair_mass"]), True
    else:
        ordered = sorted(states, key=lambda s: (s["pair_mass"], s["position"]))
        chosen, matched = [ordered[0]], False
        if len(ordered) > 1:
            chosen.append(ordered[-1])
    return dict(matched=matched, candidates=len(states), states=chosen)


def branch_prefix(prompt, source, state, forced):
    position = state["position"]
    if not 0 <= position < len(source) or source[position] != state["candidate"]:
        raise ValueError("Intervention is not on the recorded committed trajectory")
    return [*prompt, *source[:position], forced]


@torch.inference_mode()
def intervene(engine, record, row, state):
    request = request_for(record)
    prompt = engine.tokenizer.apply_chat_template(list(request.messages), add_generation_prompt=True,
                                                 return_tensors="pt", **engine.config.template_options)
    prompt_ids = prompt[0].tolist()
    common = [*prompt_ids, *row["token_ids"][:state["position"]]]
    common_tensor = torch.tensor([common], device=engine.target.device)
    logits = engine.target(common_tensor, use_cache=False, logits_to_keep=1).logits[0, -1]
    top = int(logits.argmax())
    gap = float(logits.max().double() - logits[state["candidate"]].double())
    stable = top == state["recovery"] and abs(gap - state["regret"]) <= 1e-3
    rows = []
    for arm, forced in (("proposal", state["candidate"]), ("recovery", state["recovery"])):
        inputs = branch_prefix(prompt_ids, row["token_ids"], state, forced)
        prefix = inputs[len(prompt_ids):]
        remaining = 2048 - len(prefix)
        engine.synchronize()
        started = time.perf_counter()
        if forced in engine.stops or remaining == 0:
            continuation, rounds = [], []
        else:
            detail = generate_reuse(engine.target, engine.draft,
                                    torch.tensor([inputs], device=engine.target.device),
                                    max_new_tokens=remaining, stop_token_ids=engine.stops, config=engine.reuse)
            continuation = detail.sequences[0, len(inputs):].tolist()
            rounds = detail.rounds
        engine.synchronize()
        elapsed = time.perf_counter() - started
        tokens = prefix + continuation
        text = engine.tokenizer.decode(tokens, skip_special_tokens=True)
        rows.append(dict(request_id=record["id"], state=state, arm=arm, forced_token=forced,
                         common_prefix_sha256=hashlib.sha256(json.dumps(common).encode()).hexdigest(),
                         replay_argmax=top, replay_regret=gap, same_state_stable=stable,
                         token_ids=tokens, text=text, quality=score(record, text),
                         continuation_tokens=len(continuation), continuation_rounds=len(rounds),
                         finish_reason="eos" if tokens and tokens[-1] in engine.stops else "length",
                         recompute_seconds=elapsed,
                         timing="Diagnostic full re-prefill plus strict continuation; not online policy latency"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+", required=True)
    args = parser.parse_args()
    if (not args.task_ids or len(set(args.task_ids)) != len(args.task_ids)
            or set(args.task_ids) - set(TASK_IDS)):
        parser.error("Only unique frozen development IDs are allowed")
    records = {r["id"]: r for r in load_tasks(ROOT / "data/research/asd-budget/records.json")}
    selected = [records[i] for i in TASK_IDS if i in args.task_ids]
    args.output.mkdir(parents=True, exist_ok=False)
    config = replace(RuntimeConfig.load(args.config), max_new_tokens=2048, db=None,
                     retrieval="suffix", prefix_risk={})
    engine = load_engine(config)
    original = protected(ROOT)
    info = {**provenance(engine), "status": "started", "task_ids": [r["id"] for r in selected],
            "policies": {m: asdict(c) for m, c in POLICIES.items()}, "alpha": ALPHA,
            "runner_sha256": file_hash(Path(__file__)),
            "data_sha256": file_hash(ROOT / "data/research/asd-budget/records.json"),
            "protocol_sha256": file_hash(ROOT / "docs/ASD_CONFIDENCE_PROTOCOL.md"),
            "original_sha256": original, "max_new_tokens": 2048, "prefix_tokens": 128}
    write(args.output / "provenance.json", info)
    from memorydraft.runtime import engine as module
    native = module.generate_reuse
    references, prefix_outputs = {}, {}

    def execute(request, method, mechanism):
        policy = make_policy(method, record=mechanism)
        observer = ObservedVerifier(policy) if mechanism else None
        detail = None

        def hook(*args, **kwargs):
            nonlocal detail
            if observer:
                kwargs.update(verifier=observer, on_draft=observer.on_draft)
            elif method != "DFlash":
                kwargs["verifier"] = policy
            detail = native(*args, **kwargs)
            return detail

        with patch.object(module, "generate_reuse", hook):
            result = engine.generate(request, "DFlash")
        if not mechanism and method == "DFlash":
            policy.blocks = len(detail.rounds)
        if mechanism:
            if len(policy.events) != len(detail.rounds):
                raise ValueError("Missing confidence observations")
            for event, rd in zip(policy.events, detail.rounds):
                event["generated_before"] = rd["generated_before"]
        return result, policy, detail

    try:
        warm = GenerationRequest("warmup", "warmup", 0, ({"role": "user", "content": "What is 3 plus 4?"},), 32)
        for method in POLICIES:
            execute(warm, method, False)
        for stage, repeats, limit in (("mechanism", 1, 2048), ("natural", 2, 2048), ("prefix", 2, 128)):
            (args.output / stage).mkdir()
            count = 0
            for record in selected:
                for repeat in range(repeats):
                    order = list(POLICIES)
                    random.Random(f"asd-confidence:{stage}:{record['id']}:{repeat}").shuffle(order)
                    for method in order:
                        result, policy, detail = execute(replace(request_for(record), max_new_tokens=limit),
                                                         method, stage == "mechanism")
                        row = {**asdict(result), "method": method, "repeat": repeat,
                               "asd": policy.summary(), "quality": score(record, result.text)}
                        key = record["id"], method
                        if stage == "mechanism":
                            row.update(events=policy.events, rounds=detail.rounds)
                            row["instrumented_latency_seconds"] = row.pop("latency_seconds")
                            if method in OLD_POLICIES:
                                previous = (ROOT / "results/asd-budget/gsm8k" if record["benchmark"] == "gsm8k"
                                            else ROOT / "results/asd-completion/math500")
                                old = next(r for r in read_rows(previous / "mechanism/rows.jsonl")
                                           if (r["request_id"], r["method"]) == key)
                                row["previous_prefix"] = compare_previous(result, old)
                            references[key] = row
                        else:
                            ref = references[key]
                            if (result.prompt_sha256, result.snapshot_id) != (ref["prompt_sha256"], ref["snapshot_id"]):
                                raise ValueError("Target prompt or isolated history changed")
                            if stage == "natural" and (result.token_ids != ref["token_ids"]
                                                       or policy.summary() != ref["asd"]):
                                raise ValueError("Mechanism and timed trajectories/ledgers differ")
                            if stage == "prefix":
                                row["rounds"] = detail.rounds
                                row["full_prefix"] = compare_previous(SimpleNamespace(**ref), row)
                                if key in prefix_outputs and prefix_outputs[key] != result.token_ids:
                                    raise ValueError("Prefix repeats differ")
                                prefix_outputs[key] = result.token_ids
                        append(args.output / stage / "rows.jsonl", row)
                        count += 1
                print("completed", stage, record["id"], flush=True)
            info[stage + "_rows"] = count
            write(args.output / "provenance.json", info)
        selections = {}
        count = 0
        for record in selected:
            source = references[record["id"], "ASD"]
            selections[record["id"]] = select_states(source)
            for state in selections[record["id"]]["states"]:
                for row in intervene(engine, record, source, state):
                    append(args.output / "branches.jsonl", row)
                    count += 1
            print("completed branches", record["id"], flush=True)
        write(args.output / "selections.json", selections)
        info.update(status="complete", branch_rows=count)
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
