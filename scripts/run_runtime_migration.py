"""Replay the original positive workload through legacy and production runtime paths."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

import torch

from memorydraft.evaluation.reuse_cases import confirmation_projects, messages
from memorydraft.evaluation.runtime_migration import (
    METHODS, ReplayTask, audit_rows, check_result, file_hash, frozen_inputs, input_hashes, read_rows,
)
from memorydraft.memory.store import MemoryStore
from memorydraft.retrieval.reuse import ReuseMatcher
from memorydraft.runtime.benchmark import provenance, write_json
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest, GenerationResult
import memorydraft.runtime.engine as engine_module
from scripts.run_reuse_pilot import CONFIRMATION_METHODS
from scripts.write_reuse_report import audit


ROOT = Path(__file__).resolve().parents[1]


def load_legacy(path, name, expected):
    if file_hash(path) != expected:
        raise ValueError("Legacy source differs from original runtime manifest: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def append(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def index_fingerprint(index):
    value = {key: getattr(index, key) for key in ("tokens", "starts", "ends", "sources")}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "results/reuse-confirmation")
    parser.add_argument("--legacy", type=Path, required=True, help="Hash-checked restored legacy source directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = input_hashes(args.source)
    old_info = json.loads((args.source / "provenance.json").read_text(encoding="utf-8"))
    old_projects = json.loads((args.source / "projects.json").read_text(encoding="utf-8"))
    old_initial = json.loads((args.source / "initial-history.json").read_text(encoding="utf-8"))
    old_refs = read_rows(args.source / "references.jsonl")
    old_rows = read_rows(args.source / "replay.jsonl")
    old_summary = audit(old_projects, old_initial, old_refs, old_rows, expected_projects=confirmation_projects(),
                        methods=CONFIRMATION_METHODS, repeats=3)
    if old_info["status"] != "complete" or old_summary != json.loads((args.source / "summary.json").read_text()):
        raise ValueError("Original confirmation failed its saved audit")
    old_hashes = old_info["source"]["files"]
    for path in ("memorydraft/retrieval/reuse.py", "memorydraft/evaluation/reuse_cases.py"):
        if file_hash(ROOT / path) != old_hashes[path]:
            raise ValueError("Shared retrieval or request definition changed: " + path)
    legacy_decode = load_legacy(args.legacy / "legacy_decode.py", "migration_legacy_decode",
                                old_hashes["memorydraft/backends/dflash_reuse.py"])
    legacy_suffix = load_legacy(args.legacy / "legacy_suffix.py", "migration_legacy_suffix",
                                old_hashes["memorydraft/retrieval/suffix.py"])
    tasks, events = frozen_inputs(args.source)
    database = args.output.resolve() / "history.db"
    with MemoryStore(database) as store:
        store.append(events)
        for task in tasks:
            if store.snapshot(task.request.user_id, task.request.order).snapshot_id != task.snapshot_id:
                raise ValueError("Prepared database differs from frozen snapshot")
    config = replace(RuntimeConfig.load(args.config), db=str(database))
    if config.backend != "dflash" or config.reuse != old_info["config"]:
        raise ValueError("Use the original DFlash/R settings")
    if config.target_revision != old_info["target"]["revision"]:
        raise ValueError("Target revision changed")
    engine = load_engine(config)
    info = {**provenance(engine), "status": "started", "methods": METHODS, "repeats": 3,
            "original_input_sha256": before, "original_runtime_source": old_info["source"]["source_sha256"],
            "legacy_sources": {path.name: file_hash(path) for path in sorted(args.legacy.glob("*.py"))},
            "runner_sha256": file_hash(Path(__file__)), "protocol_sha256": file_hash(ROOT / "docs/RUNTIME_MIGRATION_PROTOCOL.md"),
            "replay_module_sha256": file_hash(ROOT / "memorydraft/evaluation/runtime_migration.py"),
            "timing": "Synchronized engine call; all online snapshot/tokenize/index/decode work included. Load, warmup, resource audit and fresh AR checks separate. No HTTP/profiler."}
    write_json(args.output / "provenance.json", info)
    store = MemoryStore(database)

    @torch.inference_mode()
    def legacy(task, method):
        engine.synchronize()
        if engine.target.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(engine.target.device)
        started = time.perf_counter()
        snapshot = store.snapshot(task.request.user_id, task.request.order)
        snapshot_id = snapshot.snapshot_id
        ids = engine.tokenizer.apply_chat_template(list(task.request.messages), add_generation_prompt=True,
                                                   enable_thinking=False, return_tensors="pt").to(engine.target.device)
        prompt_sha = hashlib.sha256(json.dumps(ids[0].tolist(), sort_keys=True).encode()).hexdigest()
        index, matcher = None, None
        if method == "Legacy-R":
            index = legacy_suffix.SuffixArrayIndex(snapshot, engine.tokenizer, min_match=3, max_match=32,
                                                   prompt_text=task.request.messages[-1]["content"])
            matcher = ReuseMatcher(index, cursor=False)
        detail = legacy_decode.generate_reuse(engine.target, engine.draft, ids,
                                              max_new_tokens=task.request.max_new_tokens,
                                              stop_token_ids=engine.stops, matcher=matcher,
                                              config=legacy_decode.ReuseConfig(**config.reuse))
        engine.synchronize()
        tokens = detail.sequences[0, ids.shape[1]:].tolist()
        text = engine.tokenizer.decode(tokens, skip_special_tokens=True)
        latency = time.perf_counter() - started
        counters = {"rounds": len(detail.rounds), "draft_rounds": sum(r["route"] == "D" for r in detail.rounds),
                    "reuse_rounds": sum(r["route"] == "R" for r in detail.rounds),
                    "reuse_accepted_tokens": sum(r["accepted"] for r in detail.rounds if r["route"] == "R"),
                    "index_bytes": index.bytes_used if index else 0,
                    "feature_peak_tokens": detail.feature_peak_tokens, "buffer_flushes": detail.buffer_flushes}
        peak = torch.cuda.max_memory_allocated(engine.target.device) if engine.target.device.type == "cuda" else 0
        return GenerationResult(task.request.id, method, tokens, text, ids.shape[1], prompt_sha, snapshot_id,
                                latency, None, "eos" if tokens and tokens[-1] in engine.stops else "length", peak, counters)

    def execute(task, method):
        if method.startswith("Legacy"):
            return legacy(task, method)
        factory = engine_module.SuffixArrayIndex

        def aligned(snapshot, tokenizer, **kwargs):
            return factory(snapshot, tokenizer, **{**kwargs, "prompt_text": task.request.messages[-1]["content"]})

        control = patch.object(engine_module, "SuffixArrayIndex", aligned) if method.endswith("aligned") else nullcontext()
        with control:
            return engine.generate(task.request, "DFlash" if method == "Runtime-D" else "R+DFlash")

    rows = []
    try:
        resource_rows = []
        for task in tasks:
            snapshot = store.snapshot(task.request.user_id, task.request.order)
            old_index = legacy_suffix.SuffixArrayIndex(snapshot, engine.tokenizer, min_match=3, max_match=32,
                                                       prompt_text=task.request.messages[-1]["content"])
            aligned_index = engine_module.SuffixArrayIndex(snapshot, engine.tokenizer, min_match=3, max_match=32,
                                                            prompt_text=task.request.messages[-1]["content"])
            default_index = engine_module.SuffixArrayIndex(snapshot, engine.tokenizer, min_match=3, max_match=32,
                                                            prompt_text="\n".join(m["content"] for m in task.request.messages))
            if index_fingerprint(old_index) != index_fingerprint(aligned_index):
                raise ValueError("Aligned resource index differs")
            resource_rows.append({"request_id": task.request.id, "legacy": index_fingerprint(old_index),
                                  "aligned": index_fingerprint(aligned_index), "default": index_fingerprint(default_index),
                                  "legacy_tokens": len(old_index.tokens), "default_tokens": len(default_index.tokens)})
        write_json(args.output / "resources.json", resource_rows)
        warm = GenerationRequest("__warmup__", "__warmup__", 0, tuple(messages("Say hello.")), 16)
        warm_task = ReplayTask("__warmup__", 0, warm, {}, "")
        for method in METHODS:
            execute(warm_task, method)
        for task in tasks:
            result = engine.generate(task.request, "AR")
            try:
                check_result(result, task)
            except ValueError:
                write_json(args.output / "failure.json", {"stage": "fresh_AR", "actual": asdict(result), "frozen": task.reference})
                raise
            append(args.output / "fresh-ar.jsonl", asdict(result))
        info["fresh_ar_correct"] = len(tasks)
        write_json(args.output / "provenance.json", info)
        for task in tasks:
            for repeat in range(3):
                methods = list(METHODS)
                random.Random(f"runtime-migration-v1:{task.request.id}:{repeat}").shuffle(methods)
                for method in methods:
                    result = execute(task, method)
                    try:
                        check_result(result, task)
                    except ValueError:
                        write_json(args.output / "failure.json", {"stage": method, "actual": asdict(result), "frozen": task.reference})
                        raise
                    row = {**asdict(result), "method": method, "engine_method": result.method, "correct": True,
                           "project_id": task.project_id, "turn": task.turn, "repeat": repeat,
                           "budget": task.request.max_new_tokens}
                    row.pop("text")
                    append(args.output / "rows.jsonl", row)
                    rows.append(row)
            print("completed", task.request.id, flush=True)
        if input_hashes(args.source) != before:
            raise ValueError("Original evidence changed during replay")
        summary = audit_rows(rows, tasks)
        write_json(args.output / "summary.json", summary)
        info.update(status="complete", rows=len(rows), original_inputs_unchanged=True)
    except Exception as error:
        info.update(status="failed", error_type=type(error).__name__, error=str(error), rows=len(rows))
        raise
    finally:
        store.__exit__()
        engine.close()
        write_json(args.output / "provenance.json", info)


if __name__ == "__main__":
    main()
