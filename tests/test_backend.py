import copy

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from memorydraft.backends.transformers import TransformersBackend, check_compatibility
from memorydraft.drafters.candidates import IndependentDraft
from memorydraft.memory.store import MemoryEvent, Snapshot


@pytest.fixture(scope="module")
def backend():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, **{f"w{i}": i + 3 for i in range(29)}}
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tok.chat_template = "{% for m in messages %}{{ m['content'] + ' ' }}{% endfor %}w0 "
    cfg = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      eos_token_id=2, pad_token_id=0, bos_token_id=1)
    target = Qwen2ForCausalLM(cfg).eval()
    draft = Qwen2ForCausalLM(copy.deepcopy(cfg)).eval()
    return TransformersBackend(target, draft, tok)


def test_real_upstream_cache_and_hd_isolation(backend):
    snapshot = Snapshot("u", 10, (MemoryEvent("u", "r", 1, "w1 w2 w3 w4 w5 w6 w7 w8"),))
    messages = [{"role": "user", "content": "w1 w2 w3"}]
    ar = backend.generate(messages, snapshot, method="AR", max_new_tokens=19)
    for method in ("D", "H", "R", "RD", "RHD"):
        result = backend.generate(messages, snapshot, method=method, max_new_tokens=19, block_size=4)
        assert result.token_ids == ar.token_ids
        assert sum(e["committed"] for e in result.rounds) == len(ar.token_ids)
        for event in result.rounds:
            if event["after_state"]:
                assert event["after_state"]["kv_covered_output_len"] <= event["after_state"]["committed_output_len"]
        assert "_get_candidate_generator" not in backend.target.__dict__


@pytest.mark.parametrize("reject_at", [None, 0, 2])
@pytest.mark.parametrize("length", [1, 9])
def test_teacher_forcing_offsets_and_length(backend, monkeypatch, reject_at, length):
    messages = [{"role": "user", "content": "w3 w4"}]
    empty = Snapshot("u", 1, ())
    expected = backend.generate(messages, empty, method="AR", max_new_tokens=length).token_ids

    def propose(provider, confirmed):
        position = confirmed.shape[1]
        tokens = expected[position:position + 4]
        if reject_at is not None and len(tokens) > reject_at:
            tokens = list(tokens)
            tokens[reject_at] = (tokens[reject_at] + 1) % 32
        return torch.tensor([tokens], device=confirmed.device, dtype=torch.long)

    monkeypatch.setattr(IndependentDraft, "propose", propose)
    result = backend.generate(messages, empty, method="D", max_new_tokens=length, block_size=4)
    assert result.token_ids == expected
    assert len(result.token_ids) <= length
    if length > 1 and reject_at == 0:
        assert result.route_stats["D"]["accepted"] == 0


def test_route_switch_after_rejected_draft(backend, monkeypatch):
    from memorydraft.routing.policy import Router
    from memorydraft.retrieval.pool import ContinuationIndex
    sequence = iter(["H", "D", "R", "AR", "H", "R", "D", "H"] * 8)
    monkeypatch.setattr(Router, "choose", lambda *_: next(sequence))
    monkeypatch.setattr(ContinuationIndex, "match", lambda *args: ([7, 8], "test-source", 3))
    snapshot = Snapshot("u", 5, (MemoryEvent("u", "r", 1, "w3 w4 w5"),))
    messages = [{"role": "user", "content": "w3 w4"}]
    reference = backend.generate(messages, snapshot, method="AR", max_new_tokens=17)
    result = backend.generate(messages, snapshot, method="RHD", max_new_tokens=17, block_size=4)
    assert result.token_ids == reference.token_ids
    assert {e["route"] for e in result.rounds} >= {"H", "D", "R", "AR"}
    reused = [e for e in result.rounds if e["before_state"] and e["before_state"]["cached_prefix_tokens"] > 0]
    assert reused, "H/D must actually reuse KV"


def test_eos_and_empty_history(backend):
    messages = [{"role": "user", "content": "w3"}]
    empty = Snapshot("u", 1, ())
    reference = backend.generate(messages, empty, method="AR", max_new_tokens=10)
    previous = backend.target.generation_config.eos_token_id
    backend.target.generation_config.eos_token_id = reference.token_ids[0]
    try:
        for method in ("AR", "D", "H", "RHD"):
            result = backend.generate(messages, empty, method=method, max_new_tokens=10)
            assert result.token_ids == [reference.token_ids[0]]
    finally:
        backend.target.generation_config.eos_token_id = previous


def test_fail_closed_and_restore_on_error(backend, monkeypatch):
    empty = Snapshot("u", 1, ())
    with pytest.raises(ValueError, match="greedy"):
        backend.generate([{"role": "user", "content": "w3"}], empty, temperature=0.1)
    def fail(*_):
        raise RuntimeError("deliberate failure")
    monkeypatch.setattr(IndependentDraft, "propose", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        backend.generate([{"role": "user", "content": "w3"}], empty, method="H")
    assert "_get_candidate_generator" not in backend.target.__dict__
    assert not backend.lock.locked()


def test_incompatible_tokenizer_rejected(backend):
    other = copy.deepcopy(backend.tokenizer)
    other.add_tokens(["new-token"])
    with pytest.raises(ValueError, match="mapping"):
        check_compatibility(backend.tokenizer, other, backend.target, backend.draft)


def test_ar_and_d_do_not_pay_draft_retrieval(backend, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Draft-only retrieval must not inflate AR/D latency")
    monkeypatch.setattr("memorydraft.backends.transformers.retrieve", fail)
    empty = Snapshot("u", 1, ())
    messages = [{"role": "user", "content": "w3"}]
    ar = backend.generate(messages, empty, method="AR", max_new_tokens=3)
    d = backend.generate(messages, empty, method="D", max_new_tokens=3)
    assert ar.token_ids == d.token_ids
