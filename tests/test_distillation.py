import torch
import pytest

from memorydraft.evaluation.distillation import projects, response_sample, task_family
from tests.test_backend import backend


def test_project_split_and_past_tasks_are_independent_of_current_labels():
    groups = {split: projects(split) for split in ("train", "dev", "test")}
    ids = {split: {row["request"]["user_id"] for row in rows} for split, rows in groups.items()}
    assert len(ids["train"]) == 24 and len(ids["dev"]) == len(ids["test"]) == 8
    assert not ids["train"] & ids["dev"] and not ids["test"] & (ids["train"] | ids["dev"])
    for rows in groups.values():
        for row in rows:
            assert row["past"]["order"] == 0 < 1 < row["request"]["order"]
            assert row["past"]["messages"][-1] != row["request"]["messages"][-1]
            assert "token_ids" not in row and "answer" not in row
            assert task_family(row["request"]["messages"]) == row["request"]["scenario"]
    with pytest.raises(ValueError):
        projects("unknown")


def test_response_labels_mask_every_prompt_token_and_keep_answer_ids(backend):
    messages = [dict(role="user", content="w3 w4")]
    for history in ("", "w8 w9"):
        batch = response_sample(backend, messages, [7, 8, 2], history)
        n = backend.prompt(messages, history).shape[1]
        assert torch.all(batch["labels"][0, :n] == -100)
        assert batch["labels"][0, n:].tolist() == [7, 8, 2]
        assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    with pytest.raises(ValueError, match="Empty"):
        response_sample(backend, messages, [], "")


def test_prefix_scoring_compares_same_weights_and_keeps_reference_prefixes(backend, tmp_path):
    from memorydraft.memory.store import MemoryEvent, MemoryStore
    from scripts.run_h_training import score, teacher_labels, Variants
    request = dict(id="a", user_id="a", order=10, scenario="labels",
                   messages=[dict(role="user", content="w3 w4")])
    with MemoryStore(tmp_path / "memory.db") as store:
        store.append([MemoryEvent("a", "old", 1, "w3 w4 w5")])
        labels = teacher_labels(backend, [request], store)
        labels[0]["token_ids"] = labels[0]["token_ids"][:4]
        models = dict(base=backend.draft, trained=backend.draft)
        quality = score(backend, [request], labels, store, models, tmp_path / "prefix.jsonl")
        assert quality["base_H_minus_D"] == quality["trained_H_minus_D"]
        wrapped = Variants(backend, models, dict(D=2, H=2, off=2, on=2))
        a = wrapped.generate(request["messages"], store.snapshot("a", 10), method="TRAIN_D", max_new_tokens=4)
        b = wrapped.generate(request["messages"], store.snapshot("a", 10), method="BASE_D", max_new_tokens=4)
        assert a.token_ids == b.token_ids and backend.draft is models["trained"]


def test_lora_training_keeps_target_fixed_and_selects_saved_adapter(backend, tmp_path, monkeypatch):
    peft = pytest.importorskip("peft")
    if peft.__version__ != "0.11.1":
        pytest.skip("GPU training environment pins PEFT 0.11.1")
    import scripts.run_h_training as runner
    monkeypatch.setattr(runner, "out", tmp_path)
    monkeypatch.setattr(runner, "adapter", tmp_path / "adapter")
    target = {name: value.clone() for name, value in backend.target.state_dict().items()}
    item = response_sample(backend, [dict(role="user", content="w3 w4")], [7, 8, 2], "")
    samples = [("a", "empty", item)] * 4
    trained = runner.train(backend, samples, samples[:1])
    assert (tmp_path / "adapter/adapter_model.safetensors").is_file()
    assert trained.config.use_cache
    assert all(torch.equal(value, backend.target.state_dict()[name]) for name, value in target.items())
