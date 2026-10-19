from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from memorydraft.memory.store import MemoryEvent, MemoryStore
from memorydraft.runtime.benchmark import benchmark, check_regression, profile_request
from memorydraft.runtime.config import RuntimeConfig
from memorydraft.runtime.contracts import GenerationRequest, RequestCancelled
from memorydraft.runtime.demo import DemoEngine
from memorydraft.runtime.engine import InferenceEngine


def request(**kwargs):
    return GenerationRequest.from_dict({"id": "test", "user_id": "u", "order": 10,
                                       "messages": [{"role": "user", "content": "w1 w2 w3"}],
                                       "max_new_tokens": 19, **kwargs})


@pytest.fixture
def engine():
    torch.set_num_threads(1)
    torch.manual_seed(119)
    raw = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[MASK]": 31,
                               **{f"w{i}": i + 2 for i in range(29)}}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] + ' ' }}{% endfor %}w0 "
    cfg = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=512, eos_token_id=None, pad_token_id=0)
    cfg._attn_implementation = "eager"
    target = Qwen3ForCausalLM(cfg).eval()

    class Draft(torch.nn.Module):
        block_size, mask_token_id, target_layer_ids = 4, 31, [0, 1]

        def forward(self, *, target_hidden, noise_embedding, past_key_values, **kwargs):
            length = target_hidden.shape[1] + self.block_size
            values = noise_embedding.new_zeros((1, 1, length, 8))
            past_key_values.update(values, values, 0)
            return torch.zeros_like(noise_embedding)

    return InferenceEngine(target, tokenizer, RuntimeConfig(target="unused-test-constructor", device="cpu"), Draft())


@pytest.mark.parametrize("method", ["AR", "DFlash", "R+DFlash"])
@pytest.mark.parametrize("budget", [1, 2, 19])
def test_real_model_stream_commits_exactly_reference(engine, method, budget):
    req = request(max_new_tokens=budget)
    reference = engine.generate(req, "AR")
    commits = []
    actual = engine.generate(req, method, on_tokens=lambda tokens: commits.extend(tokens))
    assert commits == actual.token_ids == reference.token_ids
    assert actual.prompt_sha256 == reference.prompt_sha256
    assert actual.snapshot_id == reference.snapshot_id
    assert 0 <= actual.ttft_seconds <= actual.latency_seconds
    engine.stops = [reference.token_ids[-1]]
    commits.clear()
    stopped = engine.generate(req, method, on_tokens=lambda tokens: commits.extend(tokens))
    assert commits == stopped.token_ids
    assert stopped.token_ids[-1] in engine.stops
    assert stopped.finish_reason == "eos"


@pytest.mark.parametrize("method", ["AR", "DFlash", "R+DFlash"])
def test_cancel_releases_model_and_does_not_pollute_next_request(engine, method):
    commits = []
    reference = engine.generate(request(), "AR")
    with pytest.raises(RequestCancelled):
        engine.generate(request(), method, on_tokens=lambda tokens: commits.extend(tokens),
                        should_stop=lambda: bool(commits))
    assert commits
    assert not engine.lock.locked()
    assert engine.generate(request(), method).token_ids == reference.token_ids


@pytest.mark.parametrize("retrieval", ["ot-budget-splice", "ot-gate-splice", "edit-budget-splice"])
def test_benefit_runtime_keeps_target_and_empty_history_fallback(engine, retrieval):
    from test_benefit import data

    reference = engine.generate(request(), "AR")
    engine.config = replace(engine.config, retrieval=retrieval, benefit=data())
    actual = engine.generate(request(), "R+DFlash")
    assert actual.token_ids == reference.token_ids
    assert actual.prompt_sha256 == reference.prompt_sha256
    assert actual.snapshot_id == reference.snapshot_id
    assert actual.counters["benefit_ot_calls"] == actual.counters["benefit_decisions"] == 0


def test_legal_snapshot_only_and_no_writeback(engine, tmp_path, monkeypatch):
    import memorydraft.runtime.engine as module

    db = tmp_path / "history.db"
    with MemoryStore(db) as store:
        store.append([MemoryEvent("u", "old", 1, "w1 w2 w3 w4"),
                      MemoryEvent("other", "private", 1, "w5 w6"),
                      MemoryEvent("u", "future", 20, "w7 w8")])
    engine.config = replace(engine.config, db=str(db))
    original = module.SuffixArrayIndex
    seen = []

    def checked(snapshot, *args, **kwargs):
        seen.append([r.record_id for r in snapshot.records])
        return original(snapshot, *args, **kwargs)

    monkeypatch.setattr(module, "SuffixArrayIndex", checked)
    ar = engine.generate(request(), "AR")
    result = engine.generate(request(), "R+DFlash")
    assert seen == [["old"]]
    assert result.snapshot_id == ar.snapshot_id
    with MemoryStore(db) as store:
        assert len(store.snapshot("u", 100).records) == 2


def test_context_and_output_limits_release_engine(engine):
    engine.config = replace(engine.config, max_new_tokens=8)
    with pytest.raises(ValueError, match="output token"):
        engine.generate(request(), "AR")
    engine.config = replace(engine.config, max_input_tokens=2)
    with pytest.raises(ValueError, match="context"):
        engine.generate(request(max_new_tokens=1), "AR")
    assert not engine.lock.locked()


@pytest.mark.parametrize("retrieval", ["edit", "ot"])
def test_runtime_selects_aligned_retrieval_and_reports_actual_cost(engine, retrieval):
    if retrieval == "ot":
        pytest.importorskip("ot")
    reference = engine.generate(request(), "AR")
    engine.config = replace(engine.config, retrieval=retrieval)
    result = engine.generate(request(), "R+DFlash")
    assert result.token_ids == reference.token_ids
    assert result.prompt_sha256 == reference.prompt_sha256
    assert result.snapshot_id == reference.snapshot_id
    assert 0 < result.counters["alignment_calls"] <= 4
    assert result.counters["alignment_seconds"] >= 0


@pytest.mark.parametrize("changes", [{"order": True}, {"max_new_tokens": 0}, {"temperature": 0.5},
                                     {"messages": [{"role": "assistant", "content": "x"}]}])
def test_invalid_requests_fail_explicitly(changes):
    with pytest.raises(ValueError):
        request(**changes)


def test_config_paths_and_packaged_audit_assets(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"target": "models/target", "db": "history.db"}))
    parsed = RuntimeConfig.load(config)
    assert parsed.target == str(tmp_path / "models/target")
    assert parsed.db == str(tmp_path / "history.db")
    root = Path(__file__).resolve().parents[1]
    assert json.loads((root / "memorydraft/assets/dflash-manifest.json").read_text()) == json.loads(
        (root / "configs/dflash-manifest.json").read_text())
    with pytest.raises(ValueError, match="float32"):
        RuntimeConfig(target="x", dtype="bfloat16")


@pytest.mark.parametrize("method", ["edit-guided", "ot-guided", "edit-splice", "ot-splice", "ot-tail-splice",
                                    "ot-order-splice", "ot-plan-splice", "ot-all-splice"])
def test_guided_runtime_counts_actual_draft_forwards(engine, method):
    if method.startswith("ot"):
        pytest.importorskip("ot")
    reference = engine.generate(request(), "AR")
    engine.config = replace(engine.config, retrieval=method)
    result = engine.generate(request(), "R+DFlash")
    assert result.token_ids == reference.token_ids
    assert result.counters["draft_forward_rounds"] >= result.counters["draft_rounds"]


def test_benchmark_pairs_outputs_and_preserves_existing_artifacts(tmp_path):
    output = tmp_path / "benchmark"
    report = benchmark(DemoEngine(), [request()], ["AR"], output, repeats=2)
    assert report["synthetic"] is True
    assert report["methods"]["AR"]["strict_correct"]
    assert report["methods"]["AR"]["p95_seconds"] is None
    assert report["methods"]["AR"]["independent_requests"] == 1
    assert json.loads((output / "provenance.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        benchmark(DemoEngine(), [request()], ["AR"], output)


@pytest.mark.parametrize("change", ["token_ids", "prompt_sha256", "snapshot_id"])
def test_benchmark_stops_on_first_token_prompt_or_snapshot_mismatch(tmp_path, change):
    class Broken(DemoEngine):
        methods = ("AR", "broken")

        def generate(self, req, method, **kwargs):
            result = super().generate(req, "AR", **kwargs)
            result.method = method
            if method == "broken":
                setattr(result, change, [999] if change == "token_ids" else "wrong")
            return result

    with pytest.raises(RuntimeError, match="mismatch"):
        benchmark(Broken(), [request()], ["broken"], tmp_path / "failed", repeats=1, warmup=False)
    assert json.loads((tmp_path / "failed/provenance.json").read_text())["status"] == "failed"
    assert (tmp_path / "failed/failure.json").exists()
    assert not (tmp_path / "failed/summary.json").exists()


def test_cpu_profiler_exports_real_operator_trace(engine, tmp_path):
    result = profile_request(engine, request(max_new_tokens=2), "AR", tmp_path / "profile")
    assert result["instrumented"]
    trace = json.loads(Path(result["trace"]).read_text())
    assert any(e.get("name") == "memorydraft::ar_generate" for e in trace["traceEvents"])


def test_regression_threshold_checks_compatibility_before_timing(tmp_path):
    class SimulatedEnvironment(DemoEngine):
        def describe(self):
            return {"synthetic": False, "backend": "unit-test-only", "torch": "test", "transformers": "test",
                    "gpu": "unit-test", "device": "test", "dtype": "test", "target_revision": "test-pinned"}

    # Simulated metadata exists only to exercise the checker, never as published performance.
    for name in ("current", "baseline"):
        benchmark(SimulatedEnvironment(), [request()], ["AR"], tmp_path / name, repeats=1, warmup=False)
        path = tmp_path / name / "rows.jsonl"
        row = json.loads(path.read_text())
        row["latency_seconds"] = 2.0 if name == "current" else 1.0
        path.write_text(json.dumps(row) + "\n")
    report = check_regression(tmp_path / "current", tmp_path / "baseline", max_slowdown=.1)
    assert report["passed"] is False
    assert report["methods"]["AR"]["current_over_baseline"] == 2.0
    path = tmp_path / "current/rows.jsonl"
    row = json.loads(path.read_text())
    row["snapshot_id"] = "changed"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="snapshots"):
        check_regression(tmp_path / "current", tmp_path / "baseline")
