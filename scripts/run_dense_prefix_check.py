"""Exploratory all-prefix check of frozen test requests; never retunes a method."""
import hashlib
import json
from pathlib import Path

import torch

from memorydraft.backends.transformers import TransformersBackend
from memorydraft.evaluation.prefixes import compare_prefixes
from memorydraft.evaluation.replay import read_jsonl
from memorydraft.memory.store import MemoryStore
from memorydraft.retrieval.pool import SemanticEncoder


if __name__ == "__main__":
    base = Path("results/qwen3-pilot")
    original = json.loads((base / "prefix-test/provenance.json").read_text())
    if original["status"] != "complete":
        raise RuntimeError("Primary prefix experiment must finish first")
    output = base / "prefix-dense-check"
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(original)
    provenance.update(status="started", stride=1,
                      diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      purpose="Exploratory uniform all-prefix check after discrepancy between strided and online acceptance; no retuning or performance claim.")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    try:
        backend = TransformersBackend.load(original["config"])
        retrieval = original["config"]["retrieval"]
        encoder = SemanticEncoder(retrieval["model"], retrieval["revision"])
        requests = [r for path in original["input_sha256"] for r in read_jsonl(path)]
        block = original["args"]["blocks"][0]
        with MemoryStore(output / "history.db") as store:
            store.import_jsonl("data/research/history.jsonl")
            summary = compare_prefixes(backend, requests, store, encoder, output / f"prefix-k{block}.jsonl",
                                       max_new_tokens=original["args"]["max_new_tokens"], block_size=block,
                                       history_budget=original["config"].get("history_budget", 512), stride=1)
        actual = read_jsonl(output / f"prefix-k{block}.jsonl")
        measured = {r["request_id"]: r for r in read_jsonl(base / "memory-test" / f"replay-k{block}.jsonl") if r["method"] == "AR"}
        for request in requests:
            rows = sorted((r for r in actual if r["request_id"] == request["id"]), key=lambda r: r["position"])
            if [r["reference_ids"][0] for r in rows] != measured[request["id"]]["token_ids"]:
                raise RuntimeError("Diagnostic AR trace differs from measured AR")
        (output / f"summary-k{block}.json").write_text(json.dumps(summary, indent=2) + "\n")
        provenance["status"] = "complete"
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        provenance.update(status="failed", error=str(error))
        raise
    finally:
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
