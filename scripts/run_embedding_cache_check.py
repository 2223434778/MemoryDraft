"""Post-test paired cache ablation; cold first visits and warm repeat visits are explicit."""
import hashlib
import json
from pathlib import Path

import torch

from memorydraft.backends.transformers import TransformersBackend
from memorydraft.evaluation.replay import read_jsonl, replay
from memorydraft.memory.store import MemoryStore
from memorydraft.retrieval.pool import CachedSemanticEncoder, SemanticEncoder


root = Path(__file__).resolve().parents[1]
out = root / "results/qwen3-pilot/embedding-cache-check"


class CacheComparison:
    def __init__(self, backend, encoders):
        self.backend, self.encoders = backend, encoders
        self.tokenizer = backend.tokenizer
        self.visits = []

    def generate(self, messages, snapshot, *, method, **kwargs):
        actual = {"H_cached": "H", "RHD_cached": "RHD"}.get(method, method)
        encoder = self.encoders.get(method)
        before = encoder.stats() if encoder else None
        if encoder:
            kwargs["encoder"] = encoder
        result = self.backend.generate(messages, snapshot, method=actual, **kwargs)
        result.method = method
        if encoder:
            self.visits.append(dict(method=method, snapshot_id=snapshot.snapshot_id,
                                    before=before, after=encoder.stats()))
        return result


if __name__ == "__main__":
    if out.exists():
        raise RuntimeError("Preserve existing cache ablation; choose a new output for a justified rerun")
    out.mkdir(parents=True)
    provenance = dict(status="started", exploratory=True,
                      note="Post-test optimization on the same six synthetic projects, not an independent heldout validation. No offline embedding precomputation. Separate H/RHD caches start empty; repeat 0 is cold and repeats 1/2 are warm.")
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    mismatch = [p for p, digest in manifest["files"].items()
                if hashlib.sha256((root / p).read_bytes()).hexdigest() != digest]
    if mismatch:
        raise RuntimeError("Source manifest mismatch: " + repr(mismatch))
    provenance["source_manifest"] = manifest
    config = json.loads((root / "configs/a6000-qwen3.json").read_text())
    selection = json.loads((root / "results/qwen3-pilot/selection.json").read_text())
    config.update(dtype="float32", attention="eager")
    provenance.update(config=config, block=selection["block"], torch=torch.__version__,
                      gpu=torch.cuda.get_device_name(0), repeats=3, max_new_tokens=128,
                      input_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                    for p in ["data/research/history.jsonl", "data/research/memory-test.jsonl"]})
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    try:
        backend = TransformersBackend.load(config)
        retrieval = config["retrieval"]
        dense = SemanticEncoder(retrieval["model"], retrieval["revision"])
        caches = {}
        for method in ("H_cached", "RHD_cached"):
            # Reuse the immutable encoder weights; isolate cache warmth between methods.
            caches[method] = CachedSemanticEncoder(dense, capacity=4096)
        wrapped = CacheComparison(backend, caches)
        with MemoryStore(out / "history.db") as store:
            store.import_jsonl(root / "data/research/history.jsonl")
            for method in ("AR", "D_native", "PLD"):
                backend.generate([dict(role="user", content="Repeat exactly: The quick brown fox jumps over the lazy dog.")],
                                 store.snapshot("__warmup__", 0), method=method, max_new_tokens=16)
            dense.scores("warmup", ["Independent public encoder warmup, not a test history."])
            summary = replay(wrapped, store, read_jsonl(root / "data/research/memory-test.jsonl"),
                             ["AR", "D_native", "D", "H", "H_cached", "RD", "RHD", "RHD_cached"],
                             out / "replay-k2.jsonl", repeats=3, encoder=dense, block_size=selection["block"],
                             max_new_tokens=128, history_budget=config.get("history_budget", 512))
        (out / "summary-k2.json").write_text(json.dumps(summary, indent=2) + "\n")
        (out / "cache-visits.json").write_text(json.dumps(wrapped.visits, indent=2) + "\n")
        provenance.update(status="complete", tokenizer_fingerprint=backend.fingerprint,
                          cache_stats={name: encoder.stats() for name, encoder in caches.items()})
    except Exception as error:
        provenance.update(status="failed", error=str(error))
        raise
    finally:
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
