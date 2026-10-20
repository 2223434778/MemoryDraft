"""Experimental approximate verification, separate from strict runtime defaults.

Independent tensor implementation of ASD's budgeted prefix equations:
https://arxiv.org/abs/2608.03447v3. Reference API inspected at
Kissmetothemoon/ASD 9426ce7b3b8cafb2d34f675a5bb9db2ebf0259f6 (Apache-2.0).
The exact-run value is an experimental alternative to remaining block length.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ASDConfig:
    budget: float = 2.0
    gate: float = .25
    cap: int = 2
    value: str = "remaining"
    selection: str = "longest"
    regret_price: float = 2.0

    def __post_init__(self):
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in (self.budget, self.gate, self.regret_price))
                or type(self.cap) is not int or self.cap < 0 or self.value not in {"remaining", "exact-run"}
                or self.selection not in {"longest", "utility"}):
            raise ValueError("Invalid ASD budget, gate, cap or suffix value")


@dataclass(frozen=True)
class ASDDecision:
    accepted: int
    strict_accepted: int
    relaxed: int
    regret: float


def suffix_values(mismatch, mode):
    k = mismatch.numel()
    positions = torch.arange(k, device=mismatch.device)
    if mode == "remaining":
        return (k - positions).double()
    # First mismatch strictly AFTER each position; the current token counts as 1.
    failures = torch.where(mismatch, positions, k)
    nearest = failures.flip(0).cummin(0).values.flip(0)
    following = torch.cat((nearest[1:], nearest.new_tensor([k])))
    return (following - positions).double()


class ASDVerifier:
    """One instance per request; a logit-regret ledger is not a quality guarantee."""

    def __init__(self, config=ASDConfig(), *, record=False):
        self.config, self.record = config, record
        self.spent, self.relaxed, self.blocks = 0., 0, 0
        self.shortened, self.foregone = 0, 0
        self.events = []

    def decide(self, tokens, logits, *, stop_token_ids=()):
        if tokens.ndim != 1 or logits.ndim != 2 or logits.shape[0] != tokens.numel() or not tokens.numel():
            raise ValueError("Expected nonempty [K] proposals and aligned [K,V] Target logits")
        top, top_ids = logits.max(-1)
        proposed = logits.gather(1, tokens[:, None]).squeeze(1)
        mismatch = tokens != top_ids
        regret = (top.double() - proposed.double()).clamp_min(0)
        values = suffix_values(mismatch, self.config.value)
        costs = torch.where(mismatch, regret, 0.)
        cumulative = costs.cumsum(0)
        counts = mismatch.long().cumsum(0)
        strict = (~mismatch).long().cumprod(0).sum()
        if self.config.budget == 0 or self.config.gate == 0 or self.config.cap == 0:
            feasible = ~mismatch  # Token identity, including distinct tied maxima.
        else:
            feasible = (~mismatch | (regret / values <= self.config.gate)) & (counts <= self.config.cap)
            feasible &= cumulative <= max(0., self.config.budget - self.spent)
        accepted = feasible.long().cumprod(0).sum()
        positions = torch.arange(tokens.numel(), device=tokens.device)
        first_stop = tokens.new_tensor(tokens.numel() + 1)
        if stop_token_ids:
            stop = torch.zeros_like(mismatch)
            for token in stop_token_ids:
                stop |= tokens == token
            first_stop = torch.where(stop, positions + 1, tokens.numel() + 1).min()
            accepted = torch.minimum(accepted, first_stop)
        longest = accepted
        if self.config.selection == "utility":
            lengths = torch.arange(tokens.numel() + 1, device=tokens.device)
            prefix_cost = torch.cat((costs.new_zeros(1), cumulative))
            baseline = torch.minimum(strict, first_stop)
            # Include recovery/bonus, but never count tokens beyond proposal EOS.
            gain = torch.minimum(lengths + 1, first_stop) - torch.minimum(baseline + 1, first_stop)
            utility = gain - self.config.regret_price * prefix_cost
            allowed = (lengths >= baseline) & (lengths <= longest)
            best = torch.where(allowed, utility, -torch.inf).max()
            tied = allowed & (utility == best)
            least_cost = torch.where(tied, prefix_cost, torch.inf).min()
            accepted = torch.where(tied & (prefix_cost == least_cost), lengths, 0).max()
        selected = positions < accepted
        charge = torch.where(selected, costs, 0.).sum()
        relaxed = (selected & mismatch).sum()
        finite = torch.isfinite(top).all() & torch.isfinite(proposed).all()
        a, s, n, amount, valid, ceiling = torch.stack((accepted.double(), strict.double(), relaxed.double(), charge,
                                                      finite.double(), longest.double())).tolist()
        if not valid or not math.isfinite(amount) or self.spent + amount > self.config.budget + 1e-9:
            raise ValueError("Non-finite Target scores or exceeded ASD request budget")
        self.spent += amount
        self.relaxed += int(n)
        self.blocks += 1
        self.shortened += int(a < ceiling)
        self.foregone += int(ceiling - a)
        decision = ASDDecision(int(a), int(s), int(n), amount)
        if self.record:
            self.events.append(dict(accepted=int(a), strict_accepted=int(s), relaxed=int(n), regret=amount,
                                    spent=self.spent, mismatch=mismatch.tolist(), regrets=regret.tolist(),
                                    values=values.tolist(), feasible_accepted=int(ceiling)))
        return decision

    def summary(self):
        return dict(budget=self.config.budget, spent=self.spent, relaxed_tokens=self.relaxed,
                    blocks=self.blocks, value=self.config.value, selection=self.config.selection,
                    regret_price=self.config.regret_price, shortened_blocks=self.shortened,
                    foregone_proposal_tokens=self.foregone, approximate=self.config.budget > 0)
