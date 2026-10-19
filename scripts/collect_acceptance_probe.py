"""Read-only Draft confidence probes on fixed, strictly checked decoding paths."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch

from memorydraft.evaluation.benefit_probe import protected as prior_protected
from memorydraft.evaluation.matrix_probe import state_digest, observed_rows
from memorydraft.evaluation.runtime_migration import check_result, file_hash, read_rows
from memorydraft.evaluation.splice_probe import inputs, input_digest, prefix_acceptance
from memorydraft.memory.store import MemoryStore
from memorydraft.retrieval.splice import SpliceReuseMatcher
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_runtime_migration import append
from scripts.run_splice_probe import Recorder


ROOT = Path(__file__).resolve().parents[1]
METHODS = {"Splice-Edit": "edit-splice", "Splice-OT": "ot-splice", "Splice-TailOT": "ot-tail-splice"}


def protected(root):
    hashes = prior_protected(root)
    hashes.update({p.relative_to(root).as_posix(): file_hash(p)
                   for p in sorted((root / "results/ot-benefit").rglob("*")) if p.is_file()})
    return hashes


class ConfidenceRecorder(Recorder):
    def __init__(self, generate, legacy):
        super().__init__(generate)
        self.legacy, self.states = legacy, []

    def __call__(self, *args, **kwargs):
        matcher = kwargs["matcher"]
        shadow = SpliceReuseMatcher(matcher.index, method="edit", config=matcher.config)

        def inspect(confirmed, draft, logits, budget):
            started = time.perf_counter()
            shadow.blocked = matcher.blocked
            pool = (shadow.splice_shortlist(confirmed, draft, budget)
                    if len(confirmed) >= 8 and len(draft) >= 4 and budget > len(draft) else [])
            legacy = [f for f in self.legacy if f["generated"] == len(confirmed)
                      and f["draft_tokens"] == draft]
            logp = logits.float().log_softmax(-1)
            probabilities = logp.exp()
            top = probabilities.topk(2, dim=-1).values
            confidence = torch.stack((top[:, 0], -(probabilities * logp).sum(-1) / math.log(logp.shape[-1]),
                                      top[:, 0] - top[:, 1]), dim=-1).cpu().tolist()

            def candidate(c):
                n = min(len(draft), len(c["tokens"]))
                ids = torch.tensor(c["tokens"][:n], device=logits.device, dtype=torch.long)
                token_probability = probabilities[torch.arange(n, device=logits.device), ids].cpu().tolist()
                return {**{k: c[k] for k in ("tokens", "prefix_length", "cheap", "source", "position")},
                        "token_probability": token_probability}

            state = dict(generated=len(confirmed), confirmed_sha256=state_digest(confirmed),
                         draft_tokens=draft, confidence=confidence, candidates=[candidate(c) for c in pool],
                         legacy=[dict(frame=f["state_sha256"], candidates=[candidate(c) for c in f["candidates"]]) for f in legacy],
                         shadow_lookups_exceed_live_budget=matcher.stats["guided_lookups"] >= matcher.config.max_fallback_calls,
                         diagnostic_seconds=time.perf_counter() - started)
            self.states.append(state)

        kwargs["on_draft"] = inspect
        return super().__call__(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks, events = inputs(ROOT)
    database = args.output.resolve() / "history.db"
    with MemoryStore(database) as store:
        store.append(events)
        if any(store.snapshot(t.request.user_id, t.request.order).snapshot_id != t.snapshot_id for t in tasks):
            raise ValueError("Changed legal memory snapshot")
    config = replace(RuntimeConfig.load(args.config), db=str(database), retrieval="suffix", benefit={})
    if (config.backend != "dflash" or config.threads != 4 or config.template_options != {"enable_thinking": False}
            or config.dtype != "float32" or config.attention != "eager"
            or config.reuse != {"max_tokens": 32, "min_match": 6, "cooldown_rounds": 1, "max_pending_tokens": 1024}):
        raise ValueError("Changed fixed inference profile")
    engine = load_engine(config)
    info = {**provenance(engine), "status": "started", "scope": "Diagnostic only: 16 known tasks, three existing paths, no throughput claim",
            "runner_sha256": file_hash(Path(__file__)), "protocol_sha256": file_hash(ROOT / "docs/ACCEPTANCE_PROTOCOL.md"),
            "inputs_sha256": input_digest(tasks, events), "original_sha256": protected(ROOT)}
    write(args.output / "provenance.json", info)
    legacy = read_rows(ROOT / "results/ot-matrix/offline/frames.jsonl")
    observed = observed_rows(ROOT)
    for frame in legacy:
        drafts = [c["draft_tokens"] for r in observed if r["request_id"] == frame["request_id"]
                  for c in r["calls"] if c["generated"] == frame["generated"]
                  and c["draft_oracle_accepted"] == frame["draft_accepted"]
                  and all(v["tokens"][:v["prefix_length"]] == c["draft_tokens"][:v["prefix_length"]]
                          for v in frame["candidates"])]
        if not drafts or any(d != drafts[0] for d in drafts):
            raise ValueError("Ambiguous legacy Draft reconstruction")
        frame["draft_tokens"] = drafts[0]
    previous = {(r["request_id"], r["method"]): r for r in read_rows(ROOT / "results/ot-matrix/mechanism/rows.jsonl")}
    rows, states, seen, paths_checked, empty_drafts = [], [], set(), 0, 0
    try:
        from memorydraft.runtime import engine as runtime_module
        for task in tasks:
            for method, retrieval in METHODS.items():
                engine.config = replace(config, retrieval=retrieval)
                recorder = ConfidenceRecorder(runtime_module.generate_reuse, [f for f in legacy if f["request_id"] == task.request.id])
                with patch.object(runtime_module, "generate_reuse", recorder):
                    result = engine.generate(task.request, "R+DFlash")
                check_result(result, task)
                old = previous.get((task.request.id, method))
                if old is not None:
                    if any(v != result.counters[k] for k, v in old["counters"].items()
                           if k != "alignment_seconds" and not k.startswith("matrix_")):
                        raise ValueError("Observer changed an existing decoding path")
                    paths_checked += 1
                row = {**asdict(result), "method": method, "project_id": task.project_id, "correct": True,
                       "rounds": recorder.rounds, "diagnostic_states": len(recorder.states)}
                row.pop("text")
                row["instrumented_latency_seconds"] = row.pop("latency_seconds")
                row.pop("ttft_seconds")
                rows.append(row)
                append(args.output / "rows.jsonl", row)
                draft_rounds = [r for r in recorder.rounds if r["draft_forward"]]
                if [r["generated_before"] for r in draft_rounds] != [s["generated"] for s in recorder.states]:
                    raise ValueError("Lost actual Draft observation")
                for s in recorder.states:
                    if not s["draft_tokens"]:
                        empty_drafts += 1
                        continue
                    generated = s["generated"]
                    if s["confirmed_sha256"] != state_digest(task.reference["token_ids"][:generated]):
                        raise ValueError("Diagnostic prefix differs from the verified Target prefix")
                    s.update(request_id=task.request.id, project_id=task.project_id, snapshot_id=task.snapshot_id)
                    remaining = task.reference["token_ids"][generated:]
                    s["draft_accepted"] = prefix_acceptance(s["draft_tokens"], remaining)
                    for c in [*s["candidates"], *(c for f in s["legacy"] for c in f["candidates"])]:
                        c["accepted"] = prefix_acceptance(c["tokens"], remaining)
                    # Probabilities are not part of the identity: repeated paths
                    # must not turn the same prefix into independent training data.
                    signature = state_digest([task.request.id, generated, s["draft_tokens"],
                                              [(c["tokens"], c["prefix_length"], c["position"]) for c in s["candidates"]]])
                    if signature not in seen:
                        seen.add(signature)
                        s["state_sha256"] = signature
                        states.append(s)
                        append(args.output / "states.jsonl", s)
            print("completed", task.request.id, flush=True)
        info.update(status="complete", rows=len(rows), states=len(states), projects=len({s["project_id"] for s in states}),
                    empty_drafts=empty_drafts,
                    candidates=sum(len(s["candidates"]) for s in states), previous_paths_checked=paths_checked,
                    original_unchanged=info["original_sha256"] == protected(ROOT))
        if len(rows) != 48 or paths_checked != 24 or not info["original_unchanged"]:
            raise ValueError("Changed diagnostic coverage or protected evidence")
    except Exception as error:
        info.update(status="failed", rows=len(rows), error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        write(args.output / "provenance.json", info)
        engine.close()


if __name__ == "__main__":
    main()
