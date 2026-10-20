"""Fixed-work, live-KV two-action compute diagnostic; never an online oracle."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import random
import time

import torch

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.backends.dflash_reuse import generate_reuse
from memorydraft.evaluation.asd_budget import load_tasks, request_for, score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_asd_budget_probe import protected
from scripts.run_asd_utility_causal_probe import CONFIG, RepairUtilityVerifier, exception_states, execute_policy
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = ("math500:1", "math500:4", "math500:7", "gsm8k:21")
WINDOW = 128
FEATURES = ("local_gain", "regret_delta", "run", "recent_advance")


def sources():
    rows = [r for part in ("math-a", "mixed-b") for r in read_rows(
        ROOT / "results/asd-confidence" / part / "mechanism/rows.jsonl") if r["method"] == "ASD-Utility"]
    indexed = {r["request_id"]: r for r in rows}
    if set(indexed) != set(TASK_IDS):
        raise ValueError("Missing frozen compute references")
    return indexed


def selected_states(row):
    # One comparison per source decision block; never duplicate the same accept action.
    first = {}
    for s in exception_states(row):
        first.setdefault(s["block"], s)
    eligible = [s for s in first.values() if s["position"] < 512 and s["position"] + WINDOW < len(row["token_ids"])]
    if len(eligible) < 4:
        raise ValueError("Fewer than four eligible source blocks")
    selected = [dict(eligible[i * (len(eligible) - 1) // 3]) for i in range(4)]
    for s in selected:
        e = row["events"][s["block"]]
        before = row["rounds"][max(0, s["block"] - 4):s["block"]]
        s.update(boundary=e["generated_before"], local_gain=e["accepted"] - s["index"],
                 regret_delta=e["regret"], recent_advance=sum(r["advance"] for r in before) / len(before) if before else 1.)
        if s["index"] != e["strict_accepted"]:
            raise ValueError("Reject action must be the strict prefix at the source block")
    return selected


class ComputeFork:
    """Keep the live decode caches: Utility before fork, chosen action, strict after."""
    def __init__(self, source, state, arm, synchronize=lambda: None, now=time.perf_counter):
        if arm not in {"accept", "reject"}:
            raise ValueError("Unknown compute action")
        self.source, self.state, self.arm = source, state, arm
        self.prefix = (ASDVerifier(CONFIG, record=True) if arm == "accept" else
                       RepairUtilityVerifier(state, source["events"][state["block"]]))
        self.strict = ASDVerifier(ASDConfig(budget=0))
        self.synchronize, self.now = synchronize, now
        self.started, self.action = None, None

    def decide(self, tokens, logits, *, stop_token_ids=()):
        if self.started is not None:
            return self.strict.decide(tokens, logits, stop_token_ids=stop_token_ids)
        block, before = self.prefix.blocks, self.prefix.spent
        decision = self.prefix.decide(tokens, logits, stop_token_ids=stop_token_ids)
        if block == self.state["block"]:
            e = self.source["events"][block]
            r = self.prefix.events[-1]["regrets"]
            expected = e["accepted"] if self.arm == "accept" else self.state["index"]
            if (tokens.tolist() != e["proposal_ids"] or logits.argmax(-1).tolist() != e["target_ids"]
                    or max(abs(a - b) for a, b in zip(r, e["regrets"])) > 1e-3
                    or abs(before - self.state["spent_before_block"]) > 1e-8 or decision.accepted != expected):
                raise ValueError("Compute fork did not reach the exact source state")
            self.action = dict(accepted=decision.accepted, strict_accepted=decision.strict_accepted,
                               charge=decision.regret, spent=self.prefix.spent, original_spent=before,
                               proposed=len(tokens), source_block=block)
            # The common verification is already paid. Selector/source-check cost is
            # excluded for this optimistic diagnostic; no predictor is deployed.
            self.synchronize()
            self.started = self.now()
        return decision


@torch.inference_mode()
def execute(engine, record, source, state, arm, horizon, repeat):
    request = request_for(record)
    prompt = engine.tokenizer.apply_chat_template(list(request.messages), add_generation_prompt=True,
                                                 return_tensors="pt", **engine.config.template_options)
    inputs = prompt.to(engine.target.device)
    limit = 2048 if horizon == "natural" else state["boundary"] + WINDOW
    fork = ComputeFork(source, state, arm, engine.synchronize)
    detail = generate_reuse(engine.target, engine.draft, inputs, max_new_tokens=limit,
                            stop_token_ids=engine.stops, config=engine.reuse, verifier=fork)
    engine.synchronize()
    if fork.started is None:
        raise ValueError("Compute action was never reached")
    elapsed = time.perf_counter() - fork.started
    tokens = detail.sequences[0, inputs.shape[1]:].tolist()
    boundary, b = state["boundary"], state["block"]
    if tokens[:boundary] != source["token_ids"][:boundary] or len(detail.rounds) <= b:
        raise ValueError("Common prefix or source round lost")
    a = fork.action["accepted"]
    event = source["events"][b]
    if tokens[boundary:boundary + a] != event["proposal_ids"][:a]:
        raise ValueError("Fork failed to commit the selected cached prefix")
    if arm == "reject" and tokens[state["position"]] != state["recovery"]:
        raise ValueError("Reject action failed to recover")
    text = engine.tokenizer.decode(tokens, skip_special_tokens=True)
    return dict(request_id=record["id"], state=state, arm=arm, horizon=horizon, repeat=repeat,
                token_ids=tokens, text=text, quality=score(record, text), action=fork.action,
                rounds=detail.rounds, future_rounds=len(detail.rounds) - b - 1,
                strict_future_regret=fork.strict.spent, new_tokens=len(tokens) - boundary,
                post_decision_seconds=elapsed, max_new_tokens=limit,
                finish_reason="eos" if tokens[-1] in engine.stops else "length")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+", required=True)
    args = parser.parse_args()
    if not args.task_ids or len(set(args.task_ids)) != len(args.task_ids) or set(args.task_ids) - set(TASK_IDS):
        parser.error("Only unique frozen task IDs allowed")
    references = sources()
    selections = {q: selected_states(references[q]) for q in TASK_IDS if q in args.task_ids}
    records = {r["id"]: r for r in load_tasks(ROOT / "data/research/asd-budget/records.json")}
    original = protected(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    engine = load_engine(replace(RuntimeConfig.load(args.config), max_new_tokens=2048, db=None,
                                 retrieval="suffix", prefix_risk={}))
    info = {**provenance(engine), "status": "started", "selections": selections, "window": WINDOW,
            "original_sha256": original, "stop_token_ids": sorted(engine.stops),
            "runner_sha256": file_hash(Path(__file__)),
            "protocol_sha256": file_hash(ROOT / "docs/ASD_COMPUTE_PROTOCOL.md")}
    info["timing"] = "After common verification and action/source-check computation; includes KV commit and strict future; prefill/selector/predictor excluded, optimistic local diagnostic only"
    write(args.output / "provenance.json", info)
    try:
        count = 0
        for q, states in selections.items():
            baseline = execute_policy(engine, records[q], references[q])
            append(args.output / "baselines.jsonl", baseline)
            print("reproduced", q, flush=True)
            for state in states:
                jobs = [("natural", a, 0) for a in ("accept", "reject")]
                jobs += [("prefix", a, rep) for rep in range(2) for a in ("accept", "reject")]
                random.Random(f"asd-compute:{q}:{state['block']}").shuffle(jobs)
                seen = {}
                for horizon, arm, repeat in jobs:
                    row = execute(engine, records[q], references[q], state, arm, horizon, repeat)
                    key = horizon, arm
                    if key in seen and (row["token_ids"], row["action"], row["future_rounds"]) != seen[key]:
                        raise ValueError("Timed compute repeats changed tokens/action/rounds")
                    seen[key] = row["token_ids"], row["action"], row["future_rounds"]
                    append(args.output / "branches.jsonl", row)
                    count += 1
                print("completed", q, state["position"], flush=True)
        info.update(status="complete", baseline_rows=len(selections), branch_rows=count)
    except Exception as error:
        info.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        info["original_unchanged"] = original == protected(ROOT)
        write(args.output / "provenance.json", info)
        engine.close()
    if not info["original_unchanged"]:
        raise ValueError("Old published evidence changed")


if __name__ == "__main__":
    main()
