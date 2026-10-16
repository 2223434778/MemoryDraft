import pytest

from memorydraft.memory.store import MemoryEvent, Snapshot
from memorydraft.retrieval.suffix import SuffixArrayIndex
from tests.test_backend import backend


class IntegerTokenizer:
    all_special_ids = [999999]

    def encode(self, text, **kwargs):
        return [int(x) for x in text.split()]


def test_suffix_preserves_large_ids_and_record_boundaries():
    snapshot = Snapshot("u", 10, tuple(MemoryEvent("u", str(i), 1, text) for i, text in enumerate([
        "100000 100001 100002 100003", "100000 100001 200000 200001",
        "100000 100001 200000 200001", "300000 300001 999999 300002 300003",
    ])))
    index = SuffixArrayIndex(snapshot, IntegerTokenizer(), min_match=2)
    assert index.match([100000, 100001], 8)[0] == [200000, 200001]
    assert index.match([100002, 100003], 8)[0] == []
    assert index.match([300000, 300001], 8)[0] == []
    assert index.match([100000, 100001, 100002], 8)[0] == [100003]
    assert index.match([34464, 34465], 8)[0] == []  # uint16 truncation must never alias.


@pytest.mark.parametrize("method", ["D_native", "D_adapter", "PLD"])
def test_upstream_controls_keep_target_and_output(backend, method):
    messages = [{"role": "user", "content": "w3 w4 w5 w3 w4 w5 w3 w4"}]
    snapshot = Snapshot("u", 5, ())
    ar = backend.generate(messages, snapshot, method="AR", max_new_tokens=17)
    result = backend.generate(messages, snapshot, method=method, max_new_tokens=17, block_size=4)
    assert result.token_ids == ar.token_ids
    assert result.target_prompt_sha256 == ar.target_prompt_sha256
    assert sum(e["committed"] for e in result.rounds) == len(ar.token_ids)
    assert "_get_candidate_generator" not in backend.target.__dict__


def test_r_snapshot_path_does_not_call_h_retrieval(backend, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("R must not depend on H semantic retrieval")
    monkeypatch.setattr("memorydraft.backends.transformers.retrieve", fail)
    messages = [{"role": "user", "content": "w3 w4"}]
    snapshot = Snapshot("u", 5, (MemoryEvent("u", "r", 1, "w3 w4 w5"),))
    ar = backend.generate(messages, snapshot, method="AR", max_new_tokens=9)
    assert backend.generate(messages, snapshot, method="R", max_new_tokens=9).token_ids == ar.token_ids


def test_matched_prefix_empty_history_control(backend, tmp_path):
    from memorydraft.evaluation.prefixes import compare_prefixes
    from memorydraft.memory.store import MemoryStore
    from memorydraft.retrieval.pool import LexicalEncoder
    import json
    request = dict(id="r", user_id="u", order=10, messages=[dict(role="user", content="w3 w4")])
    with MemoryStore(tmp_path / "history.db") as store:
        store.append([MemoryEvent("u", "old", 1, "w17 w18 w19")])
        request["selected_history_ids"] = ["old"]
        result = compare_prefixes(backend, [request], store, LexicalEncoder(), tmp_path / "prefix.jsonl",
                                  max_new_tokens=9, block_size=4, stride=3)
    rows = [json.loads(line) for line in (tmp_path / "prefix.jsonl").read_text().splitlines()]
    assert rows[0]["position"] == 0
    assert all(r["lengths"]["H_empty"] == r["lengths"]["D_format"] for r in rows)
    assert all(r["sources"]["H_selected"] == ["old"] for r in rows)
    assert all(r["history_payload_tokens"]["H_unrelated"] == r["history_payload_tokens"]["H_dense"] for r in rows)
    assert result["H_empty"]["mean_delta_vs_format_per_request"] == 0


def test_prompt_lookup_reserves_final_target_token():
    import torch
    from memorydraft.drafters.candidates import RecordedCandidates

    class LongLookup:
        def get_candidates(self, ids):
            return torch.cat((ids, torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])), 1), None

        def update_candidate_strategy(self, *args):
            pass

    wrapped = RecordedCandidates(LongLookup(), "R", max_length=5)
    ids, logits = wrapped.get_candidates(torch.tensor([[1, 2, 3]]))
    assert ids.tolist() == [[1, 2, 3, 4]]
    wrapped.update_candidate_strategy(torch.tensor([[1, 2, 3, 4, 12]]), None, 1)
    assert wrapped.events[0]["committed"] == 2
    assert wrapped.events[0]["proposed"] == 1


def test_checkpoint_sampling_defaults_cannot_override_greedy(backend, monkeypatch):
    import copy
    import torch
    configs = [copy.deepcopy(model.generation_config) for model in (backend.target, backend.draft)]
    for model in (backend.target, backend.draft):
        model.generation_config.transformers_version = "4.51.0"
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.6
        model.generation_config.top_p = 0.95
        model.generation_config.top_k = 20

    def fail(*args, **kwargs):
        raise AssertionError("Greedy experiment must never call multinomial sampling")

    monkeypatch.setattr(torch, "multinomial", fail)
    try:
        messages = [dict(role="user", content="w3 w4")]
        snapshot = Snapshot("u", 10, ())
        ar = backend.generate(messages, snapshot, method="AR", max_new_tokens=13)
        for method in ("D_native", "D_adapter", "D", "H", "PLD"):
            result = backend.generate(messages, snapshot, method=method, max_new_tokens=13, block_size=4)
            assert result.token_ids == ar.token_ids
    finally:
        for model, config in zip((backend.target, backend.draft), configs):
            model.generation_config = config
