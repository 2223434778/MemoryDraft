"""Experimental concentration-priced utility; probabilities are not correctness."""
from __future__ import annotations

import math

import torch

from memorydraft.backends.asd import ASDConfig, ASDVerifier


def token_confidence(tokens, logits):
    if (tokens.ndim != 1 or logits.ndim != 2 or logits.shape[0] != tokens.numel()
            or not tokens.numel()):
        raise ValueError("Expected aligned nonempty proposal IDs and logits")
    top, top_ids = logits.max(-1)
    proposed = logits.gather(1, tokens[:, None]).squeeze(1)
    normalizer = logits.float().logsumexp(-1)
    top_probability = (top.float() - normalizer).exp()
    proposal_probability = (proposed.float() - normalizer).exp()
    # A matching token is counted once, rather than counted as a pair twice.
    mass = top_probability + torch.where(tokens != top_ids, proposal_probability, 0.)
    return dict(top_probability=top_probability, proposal_probability=proposal_probability,
                pair_mass=mass.clamp(0, 1))


class ConfidenceUtilityVerifier(ASDVerifier):
    def __init__(self, config=ASDConfig(budget=8, selection="utility", regret_price=2), *, alpha=.5, record=False):
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha < 1:
            raise ValueError("alpha must be finite and in [0, 1)")
        if config.selection != "utility":
            raise ValueError("Confidence pricing requires utility selection")
        super().__init__(config, record=record)
        self.alpha = alpha
        self.confidence = None

    def utility_costs(self, tokens, logits, costs):
        if self.alpha == 0:
            return costs
        self.confidence = token_confidence(tokens, logits)
        weights = 1 + self.alpha * (1 - 2 * self.confidence["pair_mass"].double())
        return costs * weights

    def summary(self):
        return {**super().summary(), "confidence_alpha": self.alpha,
                "confidence_pricing": "1+alpha*(1-2*(p_top+p_proposal)); mismatch only"}
