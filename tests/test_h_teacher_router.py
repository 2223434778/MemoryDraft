import gzip
import json
import shutil
from pathlib import Path

import pytest

from memorydraft.memory.store import MemoryStore, Snapshot
from scripts.run_h_teacher_router import FixedTargetComparison, artifact_bytes, load_frozen_inputs
from tests.test_backend import backend


def test_frozen_teacher_assets_are_reused_and_protect_history_and_requests(tmp_path):
    project = Path(__file__).resolve().parents[1]
    for name in ("data/research/history.jsonl", "data/research/memory-test.jsonl",
                 "results/h-causal/provenance.json", "results/h-causal/teacher-history/history.jsonl",
                 "results/h-causal/teacher-history/past-requests.jsonl",
                 "results/h-causal/teacher-history/generations.json", "results/h-causal/router-replay.jsonl"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith("results/") and name.endswith(".jsonl"):
            target.with_suffix(".jsonl.gz").write_bytes(gzip.compress(artifact_bytes(project / name)))
        else:
            shutil.copyfile(project / name, target)
    events, requests, references, _, evidence = load_frozen_inputs(tmp_path)
    assert len(requests) == len(references) == 6 and evidence["history_reused_without_regeneration"]
    with MemoryStore(tmp_path / "test.db") as store:
        store.append(events)
        for request in requests:
            records = store.snapshot(request["user_id"], request["order"]).records
            assert len(records) == 4
            assert all(event.user_id == request["user_id"] and event.order < request["order"] for event in records)
    path = tmp_path / "results/h-causal/teacher-history/history.jsonl.gz"
    original = path.read_bytes()
    path.write_bytes(gzip.compress(gzip.decompress(original) + b"\n"))
    with pytest.raises(ValueError, match="history changed"):
        load_frozen_inputs(tmp_path)
    path.write_bytes(original)
    (tmp_path / "data/research/memory-test.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="requests changed"):
        load_frozen_inputs(tmp_path)


def test_fixed_target_wrapper_preserves_switch_and_saves_first_failure(backend, tmp_path):
    messages = [dict(role="user", content="w3 w4")]
    snapshot = Snapshot("u", 10, ())
    reference = backend.generate(messages, snapshot, method="AR", max_new_tokens=7)
    requests = [dict(id="r", messages=messages)]
    references = dict(r=dict(token_ids=reference.token_ids, target_prompt_sha256=reference.target_prompt_sha256))
    wrapped = FixedTargetComparison(backend, None, requests, references, tmp_path)
    for method in ("AR", "RHD_off", "RHD_on"):
        result = wrapped.generate(messages, snapshot, method=method, max_new_tokens=7, block_size=2)
        assert result.method == method and result.token_ids == reference.token_ids
        if method == "RHD_off":
            assert not any(event["route"] == "H" for event in result.rounds)
    references["r"]["target_prompt_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="Target prompt"):
        wrapped.generate(messages, snapshot, method="AR", max_new_tokens=7)
    failure = json.loads((tmp_path / "fixed-target-failure.json").read_text())
    assert failure["result"]["token_ids"] == reference.token_ids
