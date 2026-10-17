"""Measure the matched H switch with the previously frozen Target history."""
import gzip
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from memorydraft.backends.transformers import TransformersBackend
from memorydraft.evaluation.history_alignment import past_requests
from memorydraft.evaluation.replay import read_jsonl, replay
from memorydraft.memory.store import MemoryEvent, MemoryStore
from memorydraft.retrieval.pool import CachedSemanticEncoder, SemanticEncoder
from scripts.run_h_causal import MatchedRouterComparison, save, sha


root = Path(__file__).resolve().parents[1]
METHODS = ("AR", "RHD_off", "RHD_on", "RHD_on_cached")
FROZEN_HISTORY_SHA256 = "55483ae5160588591aed932bca5fb62e188e301f39bd064c817e3abcab0c919a"
CURRENT_REQUESTS_SHA256 = "ae9dbf9ded81a429bd5b80f02bf425d937c84aa380d637f0a1f81f6f0aa19ce7"


def artifact_bytes(path):
    path = Path(path)
    return path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())


def artifact_rows(path):
    return [json.loads(line) for line in artifact_bytes(path).decode("utf-8").splitlines() if line.strip()]


def load_frozen_inputs(project):
    """Check the old causal asset before loading current requests; never regenerate it."""
    parent = project / "results/h-causal"
    provenance = json.loads((parent / "provenance.json").read_text(encoding="utf-8"))
    if provenance["status"] != "complete" or not provenance["current_requests_read_after_teacher_freeze"]:
        raise ValueError("Parent experiment incomplete or history not frozen first")
    history = artifact_bytes(parent / "teacher-history/history.jsonl")
    digest = hashlib.sha256(history).hexdigest()
    if digest != FROZEN_HISTORY_SHA256 or digest != provenance["frozen_teacher_history_sha256"]:
        raise ValueError("Frozen Target history changed")
    original = project / "data/research/history.jsonl"
    if sha(original) != provenance["original_history_sha256"]:
        raise ValueError("Original history changed")
    events = [MemoryEvent(**row) for row in read_jsonl(original) if row["user_id"].startswith("test-project-")]
    tasks = artifact_rows(parent / "teacher-history/past-requests.jsonl")
    past_digest = hashlib.sha256(artifact_bytes(parent / "teacher-history/past-requests.jsonl")).hexdigest()
    if past_digest != provenance["frozen_past_requests_sha256"] or tasks != past_requests(events):
        raise ValueError("Frozen past requests differ from past-only task construction")
    generations = json.loads((parent / "teacher-history/generations.json").read_text(encoding="utf-8"))
    generated = {row["past_request_id"]: row for row in generations}
    if len(generated) != len(tasks) or len(generations) != len(tasks):
        raise ValueError("Incomplete or duplicate past generations")
    replacements = {}
    for task in tasks:
        row = generated[task["id"]]
        if task["order"] != 0 or task["answer_order"] != row["answer_order"] or row["hit_output_budget"]:
            raise ValueError("Invalid past answer availability or truncation")
        replacements[(task["user_id"], task["record_id"], task["answer_order"])] = row["text"]
    expected = [replace(event, text=replacements.get((event.user_id, event.record_id, event.order), event.text))
                for event in events]
    teacher_events = [MemoryEvent(**json.loads(line)) for line in history.decode("utf-8").splitlines() if line.strip()]
    if teacher_events != expected:
        raise ValueError("Teacher history includes changes beyond the frozen old answers")
    requests_path = project / "data/research/memory-test.jsonl"
    if sha(requests_path) != CURRENT_REQUESTS_SHA256 or sha(requests_path) != provenance["current_requests_sha256"]:
        raise ValueError("Current requests changed")
    requests = read_jsonl(requests_path)
    reference_rows = artifact_rows(parent / "router-replay.jsonl")
    references = {}
    for request in requests:
        group = [row for row in reference_rows if row["method"] == "AR" and row["request_id"] == request["id"]]
        if len(group) != 3 or {row["repeat"] for row in group} != {0, 1, 2}:
            raise ValueError("Parent AR reference is incomplete")
        if any(not row["correct"] or (row["token_ids"], row["target_prompt_sha256"]) !=
               (group[0]["token_ids"], group[0]["target_prompt_sha256"]) for row in group):
            raise ValueError("Parent AR references disagree")
        references[request["id"]] = group[0]
    evidence = dict(parent_provenance_sha256=sha(parent / "provenance.json"),
                    frozen_teacher_history_sha256=digest, frozen_past_requests_sha256=past_digest,
                    generations_sha256=sha(parent / "teacher-history/generations.json"),
                    current_requests_sha256=sha(requests_path),
                    parent_router_replay_sha256=hashlib.sha256(artifact_bytes(parent / "router-replay.jsonl")).hexdigest(),
                    history_reused_without_regeneration=True)
    return teacher_events, requests, references, provenance, evidence


class FixedTargetComparison(MatchedRouterComparison):
    def __init__(self, backend, cached, requests, references, output):
        super().__init__(backend, cached)
        self.references = {json.dumps(request["messages"], sort_keys=True): references[request["id"]]
                           for request in requests}
        self.output = output

    def generate(self, messages, snapshot, *, method, **kwargs):
        result = super().generate(messages, snapshot, method=method, **kwargs)
        reference = self.references[json.dumps(messages, sort_keys=True)]
        limit = kwargs.get("max_new_tokens", 128)
        if (result.target_prompt_sha256 != reference["target_prompt_sha256"] or
                result.token_ids != reference["token_ids"][:limit]):
            save(self.output / "fixed-target-failure.json", dict(result=asdict(result), expected=reference))
            raise RuntimeError("Target prompt or greedy output changed from the previous experiment; evidence saved")
        if method == "RHD_off" and any(event["route"] == "H" for event in result.rounds):
            raise RuntimeError("H-off activated H")
        return result


def main():
    output = root / "results/h-teacher-router"
    if output.exists():
        raise RuntimeError("Preserve existing results; do not overwrite evidence")
    output.mkdir(parents=True)
    provenance = dict(status="started", exploratory=True, methods=METHODS,
                      protocol_sha256=sha(root / "docs/H_TEACHER_ROUTER_PROTOCOL.md"),
                      fixed_profile=dict(dtype="float32", attention="eager", block=2, max_new_tokens=128, repeats=3))
    try:
        manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
        mismatches = [path for path, digest in manifest["files"].items() if sha(root / path) != digest]
        if mismatches:
            raise RuntimeError("Source manifest mismatch: " + repr(mismatches))
        provenance["source_manifest"] = manifest
        events, requests, references, parent, evidence = load_frozen_inputs(root)
        provenance.update(evidence)
        config = json.loads((root / "configs/a6000-qwen3.json").read_text())
        config.update(dtype="float32", attention="eager")
        if config != parent["config"]:
            raise ValueError("Model, template, retrieval or precision profile changed")
        provenance.update(config=config, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                          weight_verification=json.loads((root / "models/qwen3-weights-verified.json").read_text()))
        save(output / "provenance.json", provenance)
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        backend = TransformersBackend.load(config)
        provenance["tokenizer_fingerprint"] = backend.fingerprint
        if backend.fingerprint != parent["tokenizer_fingerprint"]:
            raise ValueError("Tokenizer changed")
        retrieval = config["retrieval"]
        encoder = SemanticEncoder(retrieval["model"], retrieval["revision"])
        encoder.scores("warmup", ["Independent public encoder warmup, not a test history."])
        with MemoryStore(output / "teacher.db") as store:
            store.append(events)
            snapshots = {request["id"]: store.snapshot(request["user_id"], request["order"]) for request in requests}
            for request in requests:
                if any(event.order >= request["order"] or event.user_id != request["user_id"]
                       for event in snapshots[request["id"]].records):
                    raise ValueError("Noncausal or cross-user history")
            save(output / "snapshots.json", {key: asdict(value) | dict(snapshot_id=value.snapshot_id)
                                             for key, value in snapshots.items()})
            for method in ("AR", "D_native", "PLD"):
                backend.generate([dict(role="user", content="Repeat exactly: The quick brown fox jumps over the lazy dog.")],
                                 store.snapshot("__warmup__", 0), method=method, max_new_tokens=16)
            smoke_backend = FixedTargetComparison(backend, CachedSemanticEncoder(encoder), requests, references, output)
            smoke = replay(smoke_backend, store, requests[:2], METHODS, output / "smoke.jsonl",
                           encoder=encoder, block_size=2, max_new_tokens=16, history_budget=config["history_budget"])
            save(output / "smoke-summary.json", smoke)
            wrapped = FixedTargetComparison(backend, CachedSemanticEncoder(encoder), requests, references, output)
            provenance["timed_cache_initial_stats"] = wrapped.cached.stats()
            save(output / "provenance.json", provenance)
            print("Frozen Target history verified; smoke complete; starting 72 timed runs", flush=True)
            summary = replay(wrapped, store, requests, METHODS, output / "replay.jsonl", repeats=3,
                             encoder=encoder, block_size=2, max_new_tokens=128, history_budget=config["history_budget"])
            save(output / "summary.json", summary)
            save(output / "cache-visits.json", wrapped.cache_visits)
            provenance["cache_stats"] = wrapped.cached.stats()
            for request in requests:
                if store.snapshot(request["user_id"], request["order"]) != snapshots[request["id"]]:
                    raise ValueError("Frozen history mutated during replay")
        provenance["status"] = "complete"
        print("Target-history router comparison complete", json.dumps(summary), flush=True)
    except Exception as error:
        provenance.update(status="failed", exception_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(output / "provenance.json", provenance)


if __name__ == "__main__":
    main()
