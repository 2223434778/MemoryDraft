from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from memorydraft.evaluation.adaptation import (ALPHA, RANK, AdapterBank, attach_lora, block_logits,
    evaluate_blocks, parameter_guard, position_loss, sample_loss, teacher_example)
from tests.test_dflash_reuse import tiny


class ToyDraft(torch.nn.Module):
    block_size, mask_token_id, target_layer_ids = 4, 31, [0, 1]
    def __init__(self):
        super().__init__()
        self.config = Qwen3Config(hidden_size=32, num_hidden_layers=1)
        self.q_proj = torch.nn.Linear(32, 32, bias=False)
        self.v_proj = torch.nn.Linear(32, 32, bias=False)

    def forward(self, *, target_hidden, noise_embedding, position_ids, past_key_values, use_cache, **kwargs):
        assert past_key_values is None and not use_cache
        assert position_ids.shape[1] == target_hidden.shape[1]+4
        return self.q_proj(noise_embedding)+self.v_proj(target_hidden[..., :32].mean(1, keepdim=True))


def test_teacher_features_end_before_anchor_and_labels_follow_actual_prefix(tiny):
    sample = teacher_example(tiny.target, tiny.reference[:, :8], [0, 1], 4)
    assert sample["labels"].tolist() == tiny.reference[:, 8:11].tolist()
    assert sample["anchor"].tolist() == tiny.reference[:, 7:8].tolist()
    torch.testing.assert_close(sample["context"], tiny.features[:, :7], atol=2e-5, rtol=2e-5)
    assert not sample["context"].requires_grad
    stopped = teacher_example(tiny.target, tiny.reference[:, :8], [0, 1], 4, stops=sample["labels"][0, :1].tolist())
    assert stopped["labels"].shape[1] == 1


def test_corrected_teacher_labels_use_changed_prefix_not_old_draft_suffix(tiny):
    altered = tiny.reference[:, :8].clone()
    altered[:, 6] = (altered[:, 6]+1) % 32
    sample = teacher_example(tiny.target, altered, [0, 1], 4)
    with torch.no_grad():
        expected = tiny.target.generate(altered, attention_mask=torch.ones_like(altered), max_new_tokens=3,
            do_sample=False, use_model_defaults=False, eos_token_id=None)
    assert sample["labels"].tolist() == expected[:, -3:].tolist()
    assert sample["prefix_ids"].tolist() == altered.tolist()


def test_position_loss_masks_padding_and_prioritizes_first_tokens():
    logits = torch.randn(1, 3, 7, requires_grad=True)
    labels = torch.tensor([[1, 2, -100]])
    loss = position_loss(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits[0, :2], labels[0, :2], reduction="none")
    weight = torch.exp(-torch.arange(2)/7)
    torch.testing.assert_close(loss, (expected*weight).sum()/weight.sum())
    loss.backward()
    assert torch.equal(logits.grad[:, 2], torch.zeros_like(logits.grad[:, 2]))
    with pytest.raises(ValueError):
        position_loss(logits, torch.full_like(labels, -100))


def test_offline_dense_merge_restores_base_and_does_not_accumulate():
    draft = ToyDraft()
    bank = AdapterBank(draft)
    torch.manual_seed(43)
    state = {"base_model.model."+n+suffix: torch.randn(shape)*.01
             for n, w in bank.original.items() for suffix, shape in
             ((".lora_A.weight", (RANK, w.shape[1])), (".lora_B.weight", (w.shape[0], RANK)))}
    bank.select(state)
    first = {n:p.clone() for n, p in draft.state_dict().items()}
    bank.select(state)
    assert all(torch.equal(p, draft.state_dict()[n]) for n, p in first.items())
    for n, w in bank.original.items():
        prefix = "base_model.model."+n
        expected = w+state[prefix+".lora_B.weight"]@state[prefix+".lora_A.weight"]*(ALPHA/RANK)
        torch.testing.assert_close(draft.get_submodule(n).weight, expected)
    bank.select()
    assert all(torch.equal(draft.get_submodule(n).weight, w) for n, w in bank.original.items())
    bad = dict(state, unexpected=torch.zeros(1))
    with pytest.raises(ValueError):
        bank.select(bad)
    assert all(torch.equal(draft.get_submodule(n).weight, w) for n, w in bank.original.items())


def test_calibration_is_request_balanced_and_model_forward_is_differentiable(tiny):
    target = deepcopy(tiny.target).requires_grad_(False)
    draft = ToyDraft()
    sample = teacher_example(target, tiny.reference[:, :8], [0, 1], 4)
    sample_loss(target, draft, sample).backward()
    assert draft.q_proj.weight.grad.norm() > 0
    assert all(p.grad is None for p in target.parameters())
    a = evaluate_blocks(target, draft, [(dict(request_id="a"), sample)])
    b = evaluate_blocks(target, draft, [(dict(request_id="a"), sample)]*3+[(dict(request_id="b"), sample)])
    assert a["loss"] == pytest.approx(b["loss"])


def test_real_peft_updates_only_adapter_and_matches_dense_merge(tiny):
    peft = pytest.importorskip("peft")
    if peft.__version__ != "0.11.1":
        pytest.skip("Pinned PEFT stack only")
    target, draft = deepcopy(tiny.target), ToyDraft()
    original = deepcopy(draft)
    target_before = {n:p.clone() for n, p in target.state_dict().items()}
    from peft.tuners.lora import model as lora_module
    probes = lora_module.is_bnb_available, lora_module.is_bnb_4bit_available
    model = attach_lora(target, draft)
    assert (lora_module.is_bnb_available, lora_module.is_bnb_4bit_available) == probes
    guard = parameter_guard(target)
    sample = teacher_example(target, tiny.reference[:, :8], [0, 1], 4)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=.001)
    loss = sample_loss(target, model, sample)
    loss.backward()
    assert sum(float(p.grad.norm()) for p in params if p.grad is not None) > 0
    optimizer.step()
    assert parameter_guard(target) == guard
    assert all(torch.equal(p, target.state_dict()[n]) for n, p in target_before.items())
    state = {n:p.detach().clone() for n, p in peft.get_peft_model_state_dict(model).items()}
    bank = AdapterBank(original)
    bank.select(state)
    with torch.no_grad():
        torch.testing.assert_close(block_logits(target, model.eval(), sample), block_logits(target, original, sample), atol=1e-5, rtol=1e-4)


def test_audited_real_dflash_accepts_peft_gradient(tiny):
    import transformers
    peft = pytest.importorskip("peft")
    directory = Path(__file__).resolve().parents[1]/"models/dflash-qwen3-8b"
    if transformers.__version__ != "4.57.3" or not (directory/"config.json").exists():
        pytest.skip("Audited DFlash stack only on prepared server")
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.from_pretrained(directory, trust_remote_code=True)
    config.hidden_size, config.intermediate_size = 32, 64
    config.num_hidden_layers, config.num_target_layers = 1, 2
    config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 8
    config.layer_types, config.block_size, config.vocab_size = ["full_attention"], 4, 32
    config.dflash_config = dict(target_layer_ids=[0, 1], mask_token_id=31)
    config._attn_implementation = "eager"
    draft = AutoModel.from_config(config, trust_remote_code=True).float()
    target = deepcopy(tiny.target)
    model = attach_lora(target, draft)
    sample = teacher_example(target, tiny.reference[:, :8], [0, 1], 4)
    sample_loss(target, model, sample).backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.parameters() if p.requires_grad)


def test_frozen_question_subset_has_no_answers_or_evaluation_questions():
    from scripts.run_asd_adaptation_collect import questions
    rows = questions()
    assert sum(r["split"] == "train" for r in rows) == 32
    assert sum(r["split"] == "calibration" for r in rows) == 8
    assert all("answer" not in r and "token_ids" not in r for r in rows)


def test_fixed_evaluation_coverage_is_eight_domain_plus_two_each_transfer():
    from scripts.run_asd_adaptation_eval import records
    from collections import Counter
    assert dict(Counter(r["benchmark"] for r in records())) == dict(gsm8k=8, math500=2, mbpp=2, humaneval=2, squad=2, json=2)


def test_non_fp32_projection_rejected_before_optional_backend_dispatch(tiny):
    target = deepcopy(tiny.target)
    before = parameter_guard(target)
    with pytest.raises(ValueError, match="native FP32"):
        attach_lora(target, ToyDraft().double())
    assert parameter_guard(target) == before
