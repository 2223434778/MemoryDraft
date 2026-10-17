"""Run the two predeclared H diagnostics sequentially on one GPU."""
import hashlib
import json
import time
from pathlib import Path

import torch

from memorydraft.backends.transformers import TransformersBackend
from memorydraft.evaluation.history_alignment import compare_history_sources, freeze_teacher_history
from memorydraft.evaluation.replay import read_jsonl, replay
from memorydraft.memory.store import MemoryEvent, MemoryStore
from memorydraft.retrieval.pool import CachedSemanticEncoder, SemanticEncoder


root = Path(__file__).resolve().parents[1]
output = root / "results/h-causal"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


class MatchedRouterComparison:
    def __init__(self, backend, cached):
        self.backend, self.cached = backend, cached
        self.tokenizer = backend.tokenizer
        self.cache_visits = []

    def generate(self, messages, snapshot, *, method, **kwargs):
        actual = "RHD" if method.startswith("RHD_") else method
        if actual == "RHD":
            kwargs["enable_history"] = method != "RHD_off"
        before = None
        if method == "RHD_on_cached":
            kwargs["encoder"] = self.cached
            before = self.cached.stats()
        result = self.backend.generate(messages, snapshot, method=actual, **kwargs)
        result.method = method
        if before is not None:
            self.cache_visits.append(dict(snapshot_id=snapshot.snapshot_id, before=before, after=self.cached.stats()))
        return result


def verify_alignment_reference(path):
    baseline_path = root / "results/qwen3-pilot/memory-test/replay-k2.jsonl"
    baseline = {r["request_id"]: r for r in read_jsonl(baseline_path) if r["method"] == "AR"}
    groups = {}
    for row in read_jsonl(path):
        groups.setdefault(row["request_id"], []).append(row)
    for request_id, group in groups.items():
        ref = baseline[request_id]
        if [r["reference_ids"][0] for r in group] != ref["token_ids"]:
            raise RuntimeError("New scoring AR trace differs from the previous fixed Target")
        if any(r["target_prompt_sha256"] != ref["target_prompt_sha256"] for r in group):
            raise RuntimeError("Target prompt changed")


if __name__ == "__main__":
    if output.exists():
        raise RuntimeError("Preserve the existing causal experiment; do not overwrite evidence")
    output.mkdir(parents=True)
    provenance = dict(status="started", exploratory=True,
                      protocol_sha256=sha(root / "docs/H_CAUSAL_PROTOCOL.md"),
                      fixed_profile=dict(dtype="float32", attention="eager", block=2, max_new_tokens=128, repeats=3))
    try:
        manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
        mismatches = [p for p, digest in manifest["files"].items() if sha(root / p) != digest]
        if mismatches:
            raise RuntimeError("Source manifest mismatch: " + repr(mismatches))
        provenance["source_manifest"] = manifest
        provenance["weight_verification"] = json.loads((root / "models/qwen3-weights-verified.json").read_text())
        config = json.loads((root / "configs/a6000-qwen3.json").read_text())
        config.update(dtype="float32", attention="eager")
        provenance.update(config=config, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                          original_history_sha256=sha(root / "data/research/history.jsonl"))
        save(output / "provenance.json", provenance)
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        backend = TransformersBackend.load(config)
        provenance["tokenizer_fingerprint"] = backend.fingerprint
        events = [MemoryEvent(**row) for row in read_jsonl(root / "data/research/history.jsonl")
                  if row["user_id"].startswith("test-project-")]
        # Past-only asset generation must finish before current test requests are read.
        started = time.perf_counter()
        evidence = freeze_teacher_history(backend, events, output / "teacher-history")
        provenance.update(teacher_build_wall_seconds=time.perf_counter() - started,
                          teacher_generation_seconds=sum(e["generation_seconds"] for e in evidence),
                          frozen_teacher_history_sha256=sha(output / "teacher-history/history.jsonl"),
                          frozen_past_requests_sha256=sha(output / "teacher-history/past-requests.jsonl"),
                          current_requests_read_after_teacher_freeze=True)
        save(output / "provenance.json", provenance)
        print("Past history frozen; now loading current test requests", flush=True)
        requests = read_jsonl(root / "data/research/memory-test.jsonl")
        provenance["current_requests_sha256"] = sha(root / "data/research/memory-test.jsonl")
        with MemoryStore(output / "manual.db") as manual, MemoryStore(output / "teacher.db") as teacher:
            manual.append(events)
            teacher.import_jsonl(output / "teacher-history/history.jsonl")
            for request in requests:
                if any(r.order >= request["order"] for r in teacher.snapshot(request["user_id"], request["order"]).records):
                    raise RuntimeError("Teacher history is noncausal")
            prefix_summary = compare_history_sources(backend, requests, manual, teacher, output / "alignment-prefix.jsonl")
            verify_alignment_reference(output / "alignment-prefix.jsonl")
            save(output / "alignment-summary.json", prefix_summary)
            print("Alignment complete", json.dumps(prefix_summary["teacher_minus_manual"]), flush=True)
            retrieval = config["retrieval"]
            encoder = SemanticEncoder(retrieval["model"], retrieval["revision"])
            encoder.scores("warmup", ["Independent public encoder warmup, not a test history."])
            wrapped = MatchedRouterComparison(backend, CachedSemanticEncoder(encoder))
            for method in ("AR", "D_native", "PLD"):
                backend.generate([dict(role="user", content="Repeat exactly: The quick brown fox jumps over the lazy dog.")],
                                 manual.snapshot("__warmup__", 0), method=method, max_new_tokens=16)
            smoke = replay(wrapped, manual, requests[:2], ["AR", "RHD_off", "RHD_on"],
                           output / "router-smoke.jsonl", encoder=encoder, block_size=2, max_new_tokens=16)
            save(output / "router-smoke-summary.json", smoke)
            summary = replay(wrapped, manual, requests,
                             ["AR", "D_native", "RD", "RHD_off", "RHD_on", "RHD_on_cached"],
                             output / "router-replay.jsonl", repeats=3, encoder=encoder,
                             block_size=2, max_new_tokens=128, history_budget=config.get("history_budget", 512))
            save(output / "router-summary.json", summary)
            save(output / "cache-visits.json", wrapped.cache_visits)
            provenance["cache_stats"] = wrapped.cached.stats()
        provenance["status"] = "complete"
        print("Router comparison complete", json.dumps(summary), flush=True)
    except Exception as error:
        provenance.update(status="failed", exception_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(output / "provenance.json", provenance)
