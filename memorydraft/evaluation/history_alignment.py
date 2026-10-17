"""Past-only history construction and paired proposal diagnostics; no answer-based retrieval."""
from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import GenerationConfig

from memorydraft.drafters.candidates import IndependentDraft
from memorydraft.evaluation.prefixes import SelectedHistoryEncoder, common_prefix, summarize_prefixes
from memorydraft.memory.store import MemoryEvent, Snapshot
from memorydraft.retrieval.pool import retrieve
from memorydraft.retrieval.suffix import SuffixArrayIndex


def past_requests(events: list[MemoryEvent]) -> list[dict]:
    """Construct old tasks exclusively from immutable old records, never current requests."""
    result = []
    for event in events:
        if not event.record_id.endswith("-old") or event.order != 1:
            continue
        if event.order < 1 or event.source_role != "assistant" or event.status != "active":
            raise ValueError("Expected an active past assistant record")
        if event.text.startswith("def "):
            tree = ast.parse(event.text)
            function = tree.body[0]
            if not isinstance(function, ast.FunctionDef):
                raise ValueError("Expected an old function")
            normalizers = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Attribute) and n.func.attr in {"lower", "upper"}}
            if normalizers != {"lower"}:
                raise ValueError("Historical normalization must come from the old lower-case task")
            memory = (f"Application memory: project {event.user_id}; function {function.name}(items, limit=10) returns a list. "
                      "It strips whitespace, skips empty labels, preserves input order, and stops at limit.")
            question = (f"Write the complete Python function {function.name}(items, limit=10). Normalize labels with lower(). "
                        "Use the variable names labels, item, label, and return labels. Return Python only, with no markdown fences.")
        else:
            duration = re.search(r"for (\w+) minutes", event.text)
            if not duration or duration[1] != "ten":
                raise ValueError("Expected the original ten-minute historical checklist")
            memory = (f"Application memory: {event.user_id} uses an eight-step release checklist covering preparation, backup, "
                      "tests, canary, monitoring, rollout, records and notification.")
            question = (f"Write the eight-step deployment checklist for {event.user_id}. The monitoring period is ten minutes. "
                        "Keep one sentence per numbered step, and begin with the project release checklist title.")
        result.append(dict(id=event.record_id + ":past-request", user_id=event.user_id, order=event.order - 1,
                           answer_order=event.order, record_id=event.record_id, source_content_hash=event.content_hash,
                           messages=[dict(role="system", content="You are a helpful assistant.\n" + memory),
                                     dict(role="user", content=question)]))
    return result


def freeze_teacher_history(backend, events, output: Path, *, max_new_tokens=256):
    """Generate each past answer once, before the caller loads any current test request."""
    output.mkdir(parents=True, exist_ok=False)
    tasks = past_requests(events)
    (output / "past-requests.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    replacements, evidence = {}, []
    for task in tasks:
        answer = backend.generate(task["messages"], Snapshot(task["user_id"], task["order"], ()),
                                  method="AR", max_new_tokens=max_new_tokens)
        replacements[(task["user_id"], task["record_id"], task["answer_order"])] = answer.text
        evidence.append(dict(past_request_id=task["id"], answer_order=task["answer_order"],
                             target_prompt_sha256=answer.target_prompt_sha256, token_ids=answer.token_ids,
                             text=answer.text, generation_seconds=answer.latency_seconds,
                             hit_output_budget=len(answer.token_ids) == max_new_tokens))
    derived = []
    for event in events:
        value = asdict(event)
        value["text"] = replacements.get((event.user_id, event.record_id, event.order), event.text)
        derived.append(value)
    (output / "history.jsonl").write_text("".join(json.dumps(t) + "\n" for t in derived), encoding="utf-8")
    (output / "generations.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def paired_delta(rows, left, right):
    groups = {}
    for row in rows:
        groups.setdefault(row["request_id"], []).append(row["lengths"][left] - row["lengths"][right])
    means = [sum(values) / len(values) for values in groups.values()]
    rng = random.Random(20261017)
    boot = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(2000))
    weak = [r for r in rows if r["r_match_length"] < 3]
    return dict(mean_delta=sum(means) / len(means), bootstrap95_request=[boot[49], boot[1949]],
                per_request={rid: sum(values) / len(values) for rid, values in groups.items()},
                weak_r_positions=len(weak), weak_r_mean_delta=sum(r["lengths"][left] - r["lengths"][right]
                                                               for r in weak) / len(weak) if weak else None)


def compare_history_sources(backend, requests, manual_store, teacher_store, output, *,
                            max_new_tokens=128, block_size=2, history_budget=512):
    rows = []
    with Path(output).open("w", encoding="utf-8") as stream:
        for request in requests:
            snapshots, texts, source_ids, payloads = {}, {}, {}, {}
            for name, store in (("H_manual", manual_store), ("H_teacher", teacher_store)):
                snapshot = store.snapshot(request["user_id"], request["order"])
                selected = set(request["selected_history_ids"])
                snapshot = Snapshot(snapshot.user_id, snapshot.before_order,
                                    tuple(r for r in snapshot.records if r.record_id in selected))
                if not snapshot.records or any(r.order >= request["order"] or r.user_id != request["user_id"] for r in snapshot.records):
                    raise ValueError("Missing or noncausal selected historical record")
                bundle = retrieve(request["messages"][-1]["content"], snapshot, backend.tokenizer,
                                  SelectedHistoryEncoder(), token_budget=history_budget)
                texts[name] = "\n---\n".join(backend.tokenizer.decode(c.token_ids) for c in bundle.chunks)
                payloads[name] = backend.tokenizer.encode(texts[name], add_special_tokens=False)
                snapshots[name], source_ids[name] = snapshot.snapshot_id, [c.source_id for c in bundle.chunks]
            matched = min(map(len, payloads.values()))
            for name in ("H_manual", "H_teacher"):
                texts[name + "_matched"] = backend.tokenizer.decode(payloads[name][:matched])
            if len(backend.tokenizer.encode(texts["H_manual_matched"], add_special_tokens=False)) != len(
                    backend.tokenizer.encode(texts["H_teacher_matched"], add_special_tokens=False)):
                raise RuntimeError("Matched history payloads have different token lengths")
            prefixes = {"D_native": backend.prompt(request["messages"]),
                        "D_format": backend.prompt(request["messages"], ""),
                        "H_empty": backend.prompt(request["messages"], "")}
            prefixes.update({name: backend.prompt(request["messages"], text) for name, text in texts.items()})
            manual_snapshot = manual_store.snapshot(request["user_id"], request["order"])
            reference = backend.generate(request["messages"], manual_snapshot, method="AR", max_new_tokens=max_new_tokens)
            target = backend.prompt(request["messages"])
            config = GenerationConfig(do_sample=False, max_length=target.shape[1] + len(reference.token_ids) + 1,
                                      eos_token_id=backend.target.generation_config.eos_token_id,
                                      pad_token_id=backend.tokenizer.pad_token_id, use_cache=True)
            providers = {name: IndependentDraft(backend.draft, ids, target.shape[1], config, block_size)
                         for name, ids in prefixes.items()}
            index = SuffixArrayIndex(manual_snapshot, backend.tokenizer)
            for position in range(len(reference.token_ids)):
                confirmed = torch.tensor([reference.token_ids[:position]], device=backend.draft.device, dtype=torch.long)
                expected = reference.token_ids[position:position + block_size]
                lengths, proposals = {}, {}
                for name, provider in providers.items():
                    proposal = provider.propose(confirmed)[0, :len(expected)].tolist()
                    proposals[name], lengths[name] = proposal, common_prefix(proposal, expected)
                    provider.reconcile(position)
                if proposals["H_empty"] != proposals["D_format"]:
                    raise RuntimeError("Empty H changed the matched D control")
                _, _, matched_r = index.match(target[0].tolist() + reference.token_ids[:position], len(expected))
                row = dict(request_id=request["id"], scenario=request["scenario"], position=position,
                           budget=len(expected), lengths=lengths, proposal_ids=proposals, reference_ids=expected,
                           delta_vs_format={name: n - lengths["D_format"] for name, n in lengths.items()},
                           r_match_length=matched_r, snapshot_id=manual_snapshot.snapshot_id,
                           history_snapshot_ids=snapshots, sources=source_ids,
                           history_payload_tokens={name: len(backend.tokenizer.encode(text, add_special_tokens=False))
                                                   for name, text in texts.items()},
                           prefix_tokens={name: ids.shape[1] for name, ids in prefixes.items()},
                           target_prompt_sha256=reference.target_prompt_sha256)
                stream.write(json.dumps(row) + "\n")
                rows.append(row)
            stream.flush()
    return dict(conditions=summarize_prefixes(rows),
                teacher_minus_manual=paired_delta(rows, "H_teacher", "H_manual"),
                teacher_minus_manual_matched=paired_delta(rows, "H_teacher_matched", "H_manual_matched"))
