import pytest
import torch

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.backends.asd_cpu import CPUASDVerifier, exact_sums_safe


CONFIGS = [ASDConfig(budget=8), ASDConfig(budget=8, value="exact-run"),
           ASDConfig(budget=8, selection="utility"), ASDConfig(budget=0),
           ASDConfig(gate=0), ASDConfig(cap=0),
           ASDConfig(budget=.75, gate=1, cap=1, value="exact-run", selection="utility", regret_price=.5)]


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_persistent_ledger_and_events_match_tensor_reference(config, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(471)
    original = ASDVerifier(config, record=True)
    cpu = CPUASDVerifier(config, record=True)
    for index in range(48):
        k = index % 15 + 1
        logits = torch.randn(k, 19, generator=rng)
        proposal = logits.argmax(-1)
        # Include many viable low-regret errors, tied maxima and proposal EOS.
        for j in range(0, k, 3):
            proposal[j] = (proposal[j]+1) % 19
            logits[j, proposal[j]] = logits[j].max() - (index % 6)*.125
        stops = [int(proposal[k//2])] if index % 2 else []
        proposal, logits = proposal.to(device), logits.to(device)
        assert cpu.decide(proposal, logits, stop_token_ids=stops) == original.decide(proposal, logits, stop_token_ids=stops)
        assert cpu.summary() == original.summary()
        assert cpu.events == original.events
    assert cpu.execution_summary() == dict(fast_calls=48, fallback_calls=0)


@pytest.mark.parametrize("budget", [.5, .5-2**-25, .5+2**-25])
@pytest.mark.parametrize("selection", ["longest", "utility"])
def test_budget_gate_and_utility_boundaries_are_not_relaxed(budget, selection):
    logits = torch.tensor([[.5, 0.], [0., 1.], [.25, 0.], [0., 1.]])
    tokens = torch.ones(4, dtype=torch.long)
    config = ASDConfig(budget=budget, gate=.125, cap=2, selection=selection)
    ref, cpu = ASDVerifier(config, record=True), CPUASDVerifier(config, record=True)
    assert ref.decide(tokens, logits) == cpu.decide(tokens, logits)
    assert ref.events == cpu.events and ref.summary() == cpu.summary()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_selected_scores_reject_without_charging(bad):
    tokens = torch.tensor([1])
    logits = torch.tensor([[1., bad]])
    cpu = CPUASDVerifier()
    with pytest.raises(ValueError, match="Non-finite"):
        cpu.decide(tokens, logits)
    assert cpu.summary() == ASDVerifier().summary()


def test_conservative_lattice_guard_falls_back_for_extreme_dynamic_range():
    assert exact_sums_safe([.125, 32., -48.25])
    assert not exact_sums_safe([1., 2**-100])
    logits = torch.tensor([[1., 2**-100], [0., 1.]])
    tokens = torch.ones(2, dtype=torch.long)
    cpu, ref = CPUASDVerifier(), ASDVerifier()
    assert cpu.decide(tokens, logits) == ref.decide(tokens, logits)
    assert cpu.summary() == ref.summary()
    assert cpu.execution_summary() == dict(fast_calls=0, fallback_calls=1)


@pytest.mark.parametrize("dtype,k", [(torch.float64, 4), (torch.float32, 16)])
def test_unsupported_execution_profile_uses_original_verifier(dtype, k):
    logits = torch.tensor([[.1, 0.]]*k, dtype=dtype)
    tokens = torch.ones(k, dtype=torch.long)
    cpu, ref = CPUASDVerifier(), ASDVerifier()
    assert cpu.decide(tokens, logits) == ref.decide(tokens, logits)
    assert cpu.execution_summary() == dict(fast_calls=0, fallback_calls=1)


def test_custom_utility_extension_uses_original_hook():
    class Custom(CPUASDVerifier):
        def utility_costs(self, tokens, logits, costs):
            return costs*3
    config = ASDConfig(selection="utility")
    custom = Custom(config)
    custom.decide(torch.tensor([1]), torch.tensor([[.1, 0.]]))
    assert custom.fallback_calls == 1 and custom.fast_calls == 0


@pytest.mark.parametrize("tokens,logits", [(torch.tensor([], dtype=torch.long), torch.zeros(0, 2)),
                                         (torch.tensor([[0]]), torch.zeros(1, 2))])
def test_shape_errors_remain_explicit(tokens, logits):
    with pytest.raises(ValueError, match="aligned"):
        CPUASDVerifier().decide(tokens, logits)
