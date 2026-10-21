"""Experimental short-block ASD execution from one compact host transfer.

The original tensor verifier remains the reference and runtime default. Scores
stay FP32 until transfer, IDs are exact below 2**24, and budget arithmetic uses
Python FP64. A conservative lattice bound makes all regret prefix/charge sums
exact in FP64; unusual score ranges fall back to the tensor implementation.
"""
from __future__ import annotations

import math

import torch

from memorydraft.backends.asd import ASDConfig, ASDDecision, ASDVerifier


def exact_sums_safe(values):
    """Bound every partial sum/difference on a common binary integer lattice."""
    ratios = [v.as_integer_ratio() for v in values]
    denominator = max(d for _, d in ratios)
    return sum(abs(n) * (denominator // d) for n, d in ratios) <= 2**53


class CPUASDVerifier(ASDVerifier):
    def __init__(self, config=ASDConfig(), *, record=False):
        super().__init__(config, record=record)
        self.fast_calls = 0
        self.fallback_calls = 0

    def decide(self, tokens, logits, *, stop_token_ids=()):
        if tokens.ndim != 1 or logits.ndim != 2 or logits.shape[0] != tokens.numel() or not tokens.numel():
            raise ValueError("Expected nonempty [K] proposals and aligned [K,V] Target logits")
        if (logits.dtype != torch.float32 or tokens.numel() > 15 or logits.shape[1] >= 2**24
                or type(self).utility_costs is not ASDVerifier.utility_costs):
            self.fallback_calls += 1
            return super().decide(tokens, logits, stop_token_ids=stop_token_ids)
        top, top_ids = logits.max(-1)
        proposed = logits.gather(1, tokens[:, None]).squeeze(1)
        # Transfer 4*K FP32 values, never the full [K,V] vocabulary logits.
        maxima, candidates, ids, proposals = torch.stack(
            (top, proposed, top_ids.float(), tokens.float())).tolist()
        if not all(math.isfinite(v) for v in (*maxima, *candidates)):
            raise ValueError("Non-finite Target scores or exceeded ASD request budget")
        if not exact_sums_safe((*maxima, *candidates)):
            self.fallback_calls += 1
            return super().decide(tokens, logits, stop_token_ids=stop_token_ids)

        k = len(proposals)
        mismatch = [a != b for a, b in zip(proposals, ids)]
        regrets = [max(0., a-b) for a, b in zip(maxima, candidates)]
        costs = [r if bad else 0. for r, bad in zip(regrets, mismatch)]
        values = [float(k-i if self.config.value == "remaining" else
                        next((j-i for j in range(i+1, k) if mismatch[j]), k-i)) for i in range(k)]
        strict = next((i for i, bad in enumerate(mismatch) if bad), k)
        first_stop = next((i+1 for i, token in enumerate(proposals) if token in stop_token_ids), k+1)
        cumulative = [0.]
        count = 0
        feasible = k
        remaining = max(0., self.config.budget-self.spent)
        disabled = self.config.budget == 0 or self.config.gate == 0 or self.config.cap == 0
        for i in range(k):
            cumulative.append(cumulative[-1] + costs[i])
            count += mismatch[i]
            allowed = (not mismatch[i] if disabled else
                       (not mismatch[i] or regrets[i]/values[i] <= self.config.gate)
                       and count <= self.config.cap and cumulative[-1] <= remaining)
            if not allowed and feasible == k:
                feasible = i
        longest = min(feasible, first_stop)
        accepted = longest
        if self.config.selection == "utility":
            baseline = min(strict, first_stop)
            # Match tensor tie order: utility, least charge, longest prefix.
            def objective(length):
                gain = min(length+1, first_stop) - min(baseline+1, first_stop)
                return gain-self.config.regret_price*cumulative[length], -cumulative[length], length
            accepted = max(range(baseline, longest+1), key=objective)
        amount = cumulative[accepted]
        if not math.isfinite(amount) or self.spent + amount > self.config.budget + 1e-9:
            raise ValueError("Non-finite Target scores or exceeded ASD request budget")
        relaxed = sum(mismatch[:accepted])
        self.spent += amount
        self.relaxed += relaxed
        self.blocks += 1
        self.shortened += int(accepted < longest)
        self.foregone += longest-accepted
        self.fast_calls += 1
        decision = ASDDecision(accepted, strict, relaxed, amount)
        if self.record:
            self.events.append(dict(accepted=accepted, strict_accepted=strict, relaxed=relaxed, regret=amount,
                                    spent=self.spent, mismatch=mismatch, regrets=regrets, values=values,
                                    feasible_accepted=longest))
        return decision

    def execution_summary(self):
        return dict(fast_calls=self.fast_calls, fallback_calls=self.fallback_calls)
