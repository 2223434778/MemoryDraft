"""Greedy proposal-quality diagnostic at identical Target-confirmed prefixes.

The Target trace is a scoring label only. Retrieval finishes before it is
generated, and only its already-confirmed prefix is passed to a drafter.
This is not an online verification or speed benchmark.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from transformers import GenerationConfig

from memorydraft.drafters.candidates import IndependentDraft
from memorydraft.memory.store import Snapshot
from memorydraft.retrieval.pool import HybridEncoder, retrieve
from memorydraft.retrieval.suffix import SuffixArrayIndex


class SelectedHistoryEncoder:
    name = "preselected-legal-history"

    def scores(self, query, texts):
        return [1.0] * len(texts)


def common_prefix(candidate, reference):
    return next((i for i, (x, y) in enumerate(zip(candidate, reference)) if x != y),
                min(len(candidate), len(reference)))


def compare_prefixes(backend, requests, store, encoder, output, *, max_new_tokens=256,
                     block_size=8, stride=16, history_budget=512):
    if min(block_size, stride, history_budget, max_new_tokens) <= 0:
        raise ValueError("Positive diagnostic budgets required")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with Path(output).open("w", encoding="utf-8") as stream:
        for request in requests:
            snapshot = store.snapshot(request["user_id"], request["order"])
            selected_ids = set(request.get("selected_history_ids", []))
            selected = Snapshot(snapshot.user_id, snapshot.before_order,
                                tuple(r for r in snapshot.records if r.record_id in selected_ids))
            texts, sources = {}, {}
            for condition, current, retriever in [
                ("H_dense", snapshot, encoder), ("H_hybrid", snapshot, HybridEncoder(encoder)),
                ("H_selected", selected, SelectedHistoryEncoder()),
            ]:
                bundle = retrieve(request["messages"][-1]["content"], current, backend.tokenizer,
                                  retriever, token_budget=history_budget, top_k=8)
                texts[condition] = "\n---\n".join(backend.tokenizer.decode(c.token_ids) for c in bundle.chunks)
                sources[condition] = [c.source_id for c in bundle.chunks]
            target_count = len(backend.tokenizer.encode(texts["H_dense"], add_special_tokens=False))
            filler = backend.tokenizer.encode(" The botanical garden contains ferns, orchids and moss beside a quiet pond.", add_special_tokens=False)
            texts["H_unrelated"] = backend.tokenizer.decode((filler * (target_count // len(filler) + 1))[:target_count])
            sources["H_unrelated"] = ["__synthetic_token_matched__"] if target_count else []
            payload_tokens = {name: len(backend.tokenizer.encode(text, add_special_tokens=False)) for name, text in texts.items()}
            if payload_tokens["H_unrelated"] != target_count:
                raise RuntimeError("Unrelated control did not preserve the payload token budget")
            # No answer labels have been computed while constructing these contexts.
            prefixes = {"D_native": backend.prompt(request["messages"]),
                        "D_format": backend.prompt(request["messages"], ""),
                        "H_empty": backend.prompt(request["messages"], "")}
            prefixes.update({name: backend.prompt(request["messages"], text) for name, text in texts.items()})
            reference = backend.generate(request["messages"], snapshot, method="AR", max_new_tokens=max_new_tokens)
            target = backend.prompt(request["messages"])
            config = GenerationConfig(do_sample=False, max_length=target.shape[1] + len(reference.token_ids) + 1,
                                      eos_token_id=backend.target.generation_config.eos_token_id,
                                      pad_token_id=backend.tokenizer.pad_token_id, use_cache=True)
            providers = {name: IndependentDraft(backend.draft, ids, target.shape[1], config, block_size)
                         for name, ids in prefixes.items()}
            index = SuffixArrayIndex(snapshot, backend.tokenizer)
            for position in range(0, len(reference.token_ids), stride):
                confirmed = torch.tensor([reference.token_ids[:position]], dtype=torch.long, device=backend.draft.device)
                expected = reference.token_ids[position:position + block_size]
                _, _, match_length = index.match(target[0].tolist() + reference.token_ids[:position], len(expected))
                lengths, proposals = {}, {}
                for name, provider in providers.items():
                    proposal = provider.propose(confirmed)[0, :len(expected)].tolist()
                    lengths[name] = common_prefix(proposal, expected)
                    proposals[name] = proposal
                    provider.reconcile(position)
                if proposals["H_empty"] != proposals["D_format"]:
                    raise RuntimeError("Empty H differs from identical D_format")
                row = dict(request_id=request["id"], scenario=request.get("scenario", "unspecified"),
                           position=position, budget=len(expected), lengths=lengths,
                           delta_vs_format={name: n - lengths["D_format"] for name, n in lengths.items()},
                           r_match_length=match_length, sources=sources, snapshot_id=snapshot.snapshot_id,
                           history_payload_tokens=payload_tokens, proposal_ids=proposals, reference_ids=expected,
                           prefix_tokens={name: ids.shape[1] for name, ids in prefixes.items()},
                           target_prompt_sha256=reference.target_prompt_sha256)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                rows.append(row)
    return summarize_prefixes(rows)


def summarize_prefixes(rows):
    result = {}
    for name in rows[0]["lengths"] if rows else []:
        per_request = {}
        for row in rows:
            per_request.setdefault(row["request_id"], []).append(row)
        means = [sum(r["lengths"][name] for r in group) / len(group) for group in per_request.values()]
        delta = [sum(r["delta_vs_format"][name] for r in group) / len(group) for group in per_request.values()]
        rng = random.Random(20261016)
        boot = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(1000))
        weak = [r for r in rows if r["r_match_length"] < 3]
        result[name] = dict(requests=len(per_request), positions=len(rows),
                            mean_accepted_length_per_request=sum(means) / len(means),
                            mean_delta_vs_format_per_request=sum(delta) / len(delta),
                            delta_bootstrap95_request=[boot[24], boot[974]],
                            worse_position_fraction=sum(r["delta_vs_format"][name] < 0 for r in rows) / len(rows),
                            weak_r_positions=len(weak),
                            weak_r_mean_delta=sum(r["delta_vs_format"][name] for r in weak) / len(weak) if weak else None,
                            survival_by_position={j: sum(r["lengths"][name] >= j for r in rows if r["budget"] >= j) /
                                                  sum(r["budget"] >= j for r in rows)
                                                  for j in range(1, max(r["budget"] for r in rows) + 1)})
    return result
