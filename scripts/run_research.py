"""Run one bounded research stage; preserve provenance and first mismatch evidence."""
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from memorydraft.backends.transformers import TransformersBackend
from memorydraft.evaluation.prefixes import compare_prefixes
from memorydraft.evaluation.replay import read_jsonl, replay
from memorydraft.memory.store import MemoryStore
from memorydraft.retrieval.pool import HybridEncoder, LexicalEncoder, SemanticEncoder


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/a6000-qwen3.json")
parser.add_argument("--output", required=True)
parser.add_argument("--stage", choices=["smoke", "measure", "prefix"], default="smoke")
parser.add_argument("--split", choices=["dev", "test"], default="dev")
parser.add_argument("--suite", choices=["public", "memory", "combined"], default="combined")
parser.add_argument("--dtype", choices=["bfloat16", "float32"])
parser.add_argument("--attention", choices=["eager", "sdpa"])
parser.add_argument("--methods", nargs="+", default=["AR", "D_native", "D_adapter", "D", "PLD", "R"])
parser.add_argument("--blocks", nargs="+", type=int, default=[8])
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--hybrid", action="store_true")
parser.add_argument("--r-scope", choices=["snapshot", "shared_pool"], default="snapshot")
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
config = json.loads(Path(args.config).read_text())
if args.dtype:
    config["dtype"] = args.dtype
if args.attention:
    config["attention"] = args.attention
sources = []
if args.suite in {"public", "combined"}:
    sources.append(Path(f"data/research/specbench-{args.split}.jsonl"))
if args.suite in {"memory", "combined"}:
    sources.append(Path(f"data/research/memory-{args.split}.jsonl"))
requests = [r for p in sources for r in read_jsonl(p)]
if args.stage == "smoke":
    requests = requests[:2] + [r for r in requests if r["scenario"].startswith("synthetic-")][:6]
    requests = list({r["id"]: r for r in requests}.values())[:8]
provenance = dict(config=config, args=vars(args), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  request_ids=[r["id"] for r in requests],
                  input_sha256={p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  status="started", reference="Separate untimed AR trace; measured AR order randomized.")
if args.stage == "prefix":
    provenance["prefix_metric"] = "Greedy proposal quality; R stratification uses static historical suffix matching, excluding dynamic prompt lookup. Bootstrap resamples requests, not token positions."
manifest = Path("SOURCE_MANIFEST.json")
if manifest.exists():
    provenance["source_manifest"] = json.loads(manifest.read_text())
    provenance["source_manifest_mismatches"] = [
        name for name, expected in provenance["source_manifest"]["files"].items()
        if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected
    ]
verified_weights = Path("models/qwen3-weights-verified.json")
if "qwen3" in config["target"]["model"].lower() and verified_weights.exists():
    provenance["weight_verification"] = json.loads(verified_weights.read_text())


def save(name, value):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


save("provenance.json", provenance)
try:
    if provenance.get("source_manifest_mismatches"):
        raise RuntimeError("Deployed source does not match its manifest")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    started = time.perf_counter()
    backend = TransformersBackend.load(config)
    provenance["model_load_seconds"] = time.perf_counter() - started
    retrieval = config.get("retrieval", {})
    encoder = SemanticEncoder(retrieval["model"], retrieval["revision"]) if retrieval.get("kind") == "semantic" else LexicalEncoder()
    if args.hybrid:
        encoder = HybridEncoder(encoder)
    provenance["retriever"] = encoder.name
    provenance["tokenizer_fingerprint"] = backend.fingerprint
    with MemoryStore(out / "history.db") as store:
        store.import_jsonl("data/research/history.jsonl")
        for method in ("AR", "D_native", "PLD"):
            backend.generate([dict(role="user", content="Repeat exactly: The quick brown fox jumps over the lazy dog.")],
                             store.snapshot("__warmup__", 0), method=method, max_new_tokens=16)
        for block in args.blocks:
            print("running", args.stage, config["dtype"], config["attention"], "block", block, flush=True)
            if args.stage == "prefix":
                summary = compare_prefixes(backend, requests, store, encoder, out / f"prefix-k{block}.jsonl",
                                           max_new_tokens=args.max_new_tokens, block_size=block,
                                           history_budget=config.get("history_budget", 512))
            else:
                summary = replay(backend, store, requests, args.methods, out / f"replay-k{block}.jsonl",
                                 repeats=args.repeats, encoder=encoder, block_size=block,
                                 max_new_tokens=args.max_new_tokens, r_scope=args.r_scope,
                                 history_budget=config.get("history_budget", 512))
            save(f"summary-k{block}.json", summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    provenance["status"] = "complete"
except Exception as error:
    provenance.update(status="failed", exception_type=type(error).__name__, error=str(error))
    raise
finally:
    save("provenance.json", provenance)
