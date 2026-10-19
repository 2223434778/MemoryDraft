"""Measure cost kernels separately from end-to-end timing; no Target answer fitting."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import statistics
import time

import torch
from transformers import DynamicCache

from memorydraft.evaluation.benefit_probe import calibration, inputs, protected
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.memory.store import MemoryStore
from memorydraft.retrieval.benefit import BenefitPolicy
from memorydraft.retrieval.splice import SpliceReuseMatcher
from memorydraft.retrieval.suffix import SuffixArrayIndex
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write


ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = replace(RuntimeConfig.load(args.config), retrieval="suffix", benefit={})
    engine = load_engine(config)
    info = {**provenance(engine), "status": "started", "original_sha256": protected(ROOT),
            "runner_sha256": file_hash(Path(__file__)), "scope": "Cost kernels, not end-to-end generation latency"}
    write(args.output / "provenance.json", info)
    measurements, profiles = [], []

    def measure(label, fn, **metadata):
        for _ in range(2):
            fn()
        values = []
        for _ in range(7):
            engine.synchronize()
            started = time.perf_counter()
            fn()
            engine.synchronize()
            values.append(time.perf_counter() - started)
        measurements.append(dict(label=label, **metadata, seconds=values))
        return statistics.median(values)

    try:
        target, draft = engine.target, engine.draft
        text = "CREATE TABLE inventory (id INTEGER, quantity INTEGER);\n" * 200
        tokens = engine.tokenizer.encode(text, add_special_tokens=False)
        for context in (256, 768, 1536):
            ids = torch.tensor([tokens[:context]], device=target.device)
            cache = DynamicCache()
            output = target(ids, past_key_values=cache, use_cache=True, output_hidden_states=True, logits_to_keep=1)
            features = torch.cat([output.hidden_states[i + 1] for i in draft.target_layer_ids], dim=-1)
            del output
            verify = {}
            for length in (15, 16, 24, 32):
                block = ids[:, :length + 1].clone()
                positions = torch.arange(context, context + length + 1, device=target.device).unsqueeze(0)

                def run_target():
                    cache.crop(context)
                    return target(block, position_ids=positions, past_key_values=cache, use_cache=True,
                                  output_hidden_states=True)

                verify[str(length)] = measure("target_verify", run_target, context=context, length=length)
            del cache
            block = ids.new_full((1, draft.block_size), draft.mask_token_id)
            dc = DynamicCache()
            # Populate the same frontier invariant as the live decoder: old Draft
            # KV plus 16 pending Target features, then the 16-token noise block.
            draft(target_hidden=features[:, :-16], noise_embedding=target.model.embed_tokens(block),
                  position_ids=torch.arange(context - 16 + draft.block_size, device=target.device).unsqueeze(0),
                  past_key_values=dc, use_cache=True, is_causal=False)
            dc.crop(context - 16)
            positions = torch.arange(context - 16, context + draft.block_size, device=target.device).unsqueeze(0)

            def run_draft():
                dc.crop(context - 16)
                hidden = draft(target_hidden=features[:, -16:], noise_embedding=target.model.embed_tokens(block),
                               position_ids=positions, past_key_values=dc, use_cache=True, is_causal=False)
                return target.lm_head(hidden[:, -draft.block_size + 1:]).argmax(-1)

            draft_seconds = measure("draft_forward", run_draft, context=context, pending=16)
            profiles.append(dict(context=context, draft=draft_seconds, verify=verify))
            del dc, features
            print("profiled", context, flush=True)
        tasks, events = inputs(ROOT)
        keyed = {t.request.id: t for t in tasks}
        frames = read_rows(ROOT / "results/ot-matrix/offline/frames.jsonl")
        samples = calibration(ROOT)
        lookups, solves = [], []
        with MemoryStore(args.output / "history.db") as store:
            store.append(events)
            for f in frames:
                task = keyed[f["request_id"]]
                source = SuffixArrayIndex(store.snapshot(task.request.user_id, task.request.order), engine.tokenizer,
                                          min_match=3, max_match=32, prompt_text="\n".join(m["content"] for m in task.request.messages))
                matcher = SpliceReuseMatcher(source, method="ot", tail_weighted=True)
                # The saved representation contains at least its retained Draft
                # prefix. Recover the actual 15-token Draft from prior traces.
                from memorydraft.evaluation.matrix_probe import observed_rows
                actual = next(c["draft_tokens"] for r in observed_rows(ROOT) if r["request_id"] == f["request_id"]
                              for c in r["calls"] if c["generated"] == f["generated"]
                              and all(item["tokens"][:item["prefix_length"]] == c["draft_tokens"][:item["prefix_length"]]
                                      for item in f["candidates"]))
                confirmed = task.reference["token_ids"][:f["generated"]]
                pool = matcher.splice_shortlist(confirmed, actual, 32)
                if not pool:
                    raise ValueError("Lost prior calibration candidate pool")
                placeholder = dict(profiles=profiles, lookup=.001, ot_candidate=.001)
                policy = BenefitPolicy(samples=[{k: s[k] for k in ("project", "features", "outcomes")} for s in samples], costs=placeholder)

                def lookup():
                    current = matcher.splice_shortlist(confirmed, actual, 32)
                    return policy.predict(current, actual, 32, 512)

                lookups.append(measure("shortlist_and_policy", lookup, request_id=task.request.id, generated=f["generated"]))
                solves.append(measure("tail_ot_pool", lambda: [matcher.candidate_score(c) for c in pool],
                                      candidates=len(pool), request_id=task.request.id) / len(pool))
        costs = dict(profiles=profiles, lookup=statistics.median(lookups), ot_candidate=statistics.median(solves))
        write(args.output / "costs.json", costs)
        write(args.output / "measurements.json", measurements)
        info.update(status="complete", original_unchanged=info["original_sha256"] == protected(ROOT),
                    costs_sha256=file_hash(args.output / "costs.json"))
        if not info["original_unchanged"]:
            raise ValueError("Protected evidence changed")
    except Exception as error:
        info.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        write(args.output / "provenance.json", info)
        engine.close()


if __name__ == "__main__":
    main()
