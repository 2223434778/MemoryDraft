from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.backends.dflash_reuse import generate_reuse
from scripts.run_asd_optimization_probe import LengthPolicy, records, TASK_IDS, TRANSFER_IDS


def test_frozen_records_and_length_decision_have_no_future_inputs():
    assert set(TASK_IDS + TRANSFER_IDS) <= records().keys()
    recent, budget = LengthPolicy("utility", "recent"), LengthPolicy("utility", "budget")
    assert recent.choose(15) == budget.choose(15) == 15
    recent.advances = budget.advances = [2, 3, 4, 5]
    assert recent.choose(15) == budget.choose(15) == 7
    recent.advances = budget.advances = [12] * 4
    budget.policy.spent = 6
    assert recent.choose(15) == 15 and budget.choose(15) == 7
    assert budget.choose(3) == 3


@pytest.fixture(scope="module")
def tiny():
    torch.set_num_threads(1)
    torch.manual_seed(119)
    config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=128, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "eager"
    target = Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[3, 4, 5]])
    with torch.inference_mode():
        reference = target.generate(ids, max_new_tokens=25, do_sample=False, use_model_defaults=False)

    class Draft(torch.nn.Module):
        block_size, mask_token_id, target_layer_ids = 8, 31, [0, 1]

        def forward(self, *, target_hidden, noise_embedding, past_key_values, position_ids, **kwargs):
            n = noise_embedding.shape[1]
            assert position_ids.shape[1] == target_hidden.shape[1] + n
            values = noise_embedding.new_zeros((1, 1, target_hidden.shape[1] + n, 8))
            past_key_values.update(values, values, 0)
            return torch.zeros_like(noise_embedding)

    return SimpleNamespace(target=target, draft=Draft(), ids=ids, reference=reference)


@pytest.mark.parametrize("draft_size", [None, 4, 8])
@pytest.mark.parametrize("verify_size", [1, 3, 7])
def test_real_kv_length_controls_obey_strict_ar_and_streaming(tiny, draft_size, verify_size):
    commits = []
    result = generate_reuse(tiny.target, tiny.draft, tiny.ids, max_new_tokens=25,
                            draft_block_size=draft_size, verify_length=lambda k: min(k, verify_size),
                            incremental_stop=True, on_tokens=commits.extend)
    assert result.sequences.tolist() == tiny.reference.tolist()
    assert commits == tiny.reference[0, 3:].tolist()
    assert all(r["proposed"] <= verify_size for r in result.rounds)


@pytest.mark.parametrize("approximate", [False, True])
def test_incremental_eos_and_multiple_stop_ids_preserve_actual_commit(tiny, approximate):
    config = ASDConfig(budget=8 if approximate else 0, selection="utility")
    plain = generate_reuse(tiny.target, tiny.draft, tiny.ids, max_new_tokens=25, verifier=ASDVerifier(config))
    chosen = plain.sequences[0, 8].item()
    expected = plain.sequences[0, 3:].tolist()
    expected = expected[:expected.index(chosen) + 1]
    answers = []
    for fast in (False, True):
        commits = []
        result = generate_reuse(tiny.target, tiny.draft, tiny.ids, max_new_tokens=25,
                                stop_token_ids=[chosen, 1000], verifier=ASDVerifier(config),
                                incremental_stop=fast, on_tokens=commits.extend)
        assert result.sequences[0, 3:].tolist() == commits
        assert commits[-1] == chosen and chosen not in commits[:-1]
        if not approximate:
            assert commits == expected
        answers.append(result.sequences.tolist())
    assert answers[0] == answers[1]


def test_transfer_gate_retains_quality_and_compares_matching_strict_reference():
    from scripts.run_asd_optimization_pipeline import select_transfer
    from scripts.run_asd_optimization_probe import ARMS
    rows = []
    for profile, arms in (("fp32-eager", list(ARMS)),
                          ("fp32-sdpa", ["D-fast", "ASD-fast", "U-fast"]),
                          ("bf16-sdpa", ["D-fast", "ASD-fast", "U-fast"])):
        for q in TASK_IDS:
            for arm in arms:
                seconds = .5 if profile == "bf16-sdpa" or arm.endswith("verify7") else 1.
                for limit, rep in ((2048, 0), (128, 0), (128, 1)):
                    rows.append(dict(experiment_profile=profile, request_id=q, arm=arm,
                                     max_new_tokens=limit, repeat=rep, token_ids=[1] * 128,
                                     latency_seconds=seconds, quality=dict(correct=arm != "U-verify7"), finish_reason="eos"))
    result = select_transfer(rows)
    assert result["selected_profile"] == "bf16-sdpa"
    assert "D-verify7" in result["fp32_arms"] and "U-verify7" not in result["fp32_arms"]


@pytest.mark.parametrize("value", [0, -1, 2.5, True, 9])
def test_invalid_draft_sizes_fail(tiny, value):
    with pytest.raises(ValueError):
        generate_reuse(tiny.target, tiny.draft, tiny.ids, max_new_tokens=5, draft_block_size=value)


@pytest.mark.parametrize("value", [0, -1, 2.5, True, 8])
def test_invalid_verification_lengths_fail(tiny, value):
    with pytest.raises(ValueError, match="Verification length"):
        generate_reuse(tiny.target, tiny.draft, tiny.ids, max_new_tokens=5, verify_length=lambda k: value)


def test_actual_dflash_short_blocks_and_short_verification(tiny):
    from pathlib import Path
    import transformers
    from transformers import AutoConfig, AutoModel

    directory = Path(__file__).resolve().parents[1] / "models/dflash-qwen3-8b"
    if transformers.__version__ != "4.57.3" or not directory.exists():
        pytest.skip("Audited server DFlash assets/environment only")
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    config.hidden_size, config.intermediate_size = 32, 64
    config.num_hidden_layers, config.num_target_layers = 1, 2
    config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 8
    config.layer_types, config.block_size, config.vocab_size = ["full_attention"], 8, 32
    config.dflash_config = dict(target_layer_ids=[0, 1], mask_token_id=31)
    for attention in ("eager", "sdpa"):
        config._attn_implementation = attention
        draft = AutoModel.from_config(config, trust_remote_code=True).float().eval()
        for size, verification in ((4, 3), (8, 1)):
            result = generate_reuse(tiny.target, draft, tiny.ids, max_new_tokens=25, draft_block_size=size,
                                    verify_length=lambda k: min(k, verification), incremental_stop=True)
            assert result.sequences.tolist() == tiny.reference.tolist()
