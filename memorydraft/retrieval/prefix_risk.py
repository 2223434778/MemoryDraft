"""Use calibrated Draft-prefix risk only to choose a historical handoff."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

from memorydraft.retrieval.acceptance import prefix_features
from memorydraft.retrieval.splice import SpliceProposal, SpliceReuseMatcher


def draft_confidence(logits):
    import torch

    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("Expected per-position Draft vocabulary logits")
    logp = logits.float().log_softmax(-1)
    probabilities = logp.exp()
    top = probabilities.topk(2, dim=-1).values
    values = torch.stack((top[:, 0], -(probabilities * logp).sum(-1) / math.log(logp.shape[-1]),
                          top[:, 0] - top[:, 1]), dim=-1).cpu().tolist()
    if any(not math.isfinite(x) or not 0 <= x <= 1 for row in values for x in row):
        raise ValueError("Non-finite or invalid Draft confidence")
    return values


@dataclass(frozen=True)
class PrefixRiskPolicy:
    weights: list[float]
    confidence: bool = True

    def __post_init__(self):
        if (len(self.weights) != 6 or type(self.confidence) is not bool
                or any(type(w) not in (int, float) or not math.isfinite(w) for w in self.weights)):
            raise ValueError("Invalid frozen prefix risk model")

    def survival(self, values):
        if (not 1 <= len(values) <= 15 or any(len(row) != 3 or any(
                not math.isfinite(x) or not 0 <= x <= 1 for x in row) for row in values)):
            raise ValueError("Prefix risk model expects 1–15 Draft positions")
        state = {"confidence": values}
        curve, previous = [], 1.
        for k in range(1, len(values) + 1):
            features = prefix_features(state, k, self.confidence)
            score = sum(w * x for w, x in zip(self.weights, features))
            probability = 1 / (1 + math.exp(-max(-40., min(40., score))))
            previous = min(previous, probability)
            curve.append(previous)
        return curve

    def select(self, pool, values):
        if not pool or any(not 1 <= c["prefix_length"] <= len(values) for c in pool):
            raise ValueError("Expected available Draft-prefix handoff candidates")
        baseline = max(range(len(pool)), key=lambda i: (pool[i]["cheap"], -pool[i]["prefix_length"], -pool[i]["position"]))
        curve = self.survival(values)
        safe = [i for i, c in enumerate(pool) if curve[c["prefix_length"] - 1] >= .5]
        selected = (max(safe, key=lambda i: (pool[i]["cheap"], -pool[i]["prefix_length"], -pool[i]["position"]))
                    if curve[pool[baseline]["prefix_length"] - 1] < .5 and safe else baseline)
        return baseline, selected, curve


class PrefixRiskReuseMatcher(SpliceReuseMatcher):
    def __init__(self, index, *, policy: PrefixRiskPolicy, **kwargs):
        super().__init__(index, method="edit", **kwargs)
        self.policy, self._logits = policy, None
        self.lookup = "edit-risk-splice"
        self.record_decisions, self.decisions = False, []
        self.stats.update(risk_evaluations=0, risk_switches=0, risk_unsafe_unchanged=0, risk_single_boundary=0)

    def set_draft_logits(self, logits):
        # The decoder owns this tensor; consume it once without modifying it.
        self._logits = logits

    def propose_with_draft(self, confirmed, draft_tokens, max_tokens):
        logits, self._logits = self._logits, None
        if (len(confirmed) < self.config.min_context or not self.hint_tokens <= len(draft_tokens) <= self.config.window
                or max_tokens <= len(draft_tokens) or not self.index.tokens
                or self.stats["guided_lookups"] >= self.config.max_fallback_calls):
            return None
        started = time.perf_counter()
        self.stats["guided_lookups"] += 1
        try:
            pool = self.splice_shortlist(confirmed, draft_tokens, max_tokens)
            if not pool:
                self.stats["guided_no_match"] += 1
                return None
            self.stats["alignment_calls"] += 1
            self.stats["alignment_candidates"] += len(pool)
            pool = [c for c in pool if c["cheap"] >= self.config.minimum_score]
            if not pool:
                return None
            baseline = selected = 0  # shortlist is already in the Edit tie order.
            values, curve = [], []
            if len({c["prefix_length"] for c in pool}) > 1:
                if self.policy.confidence:
                    if logits is None or logits.shape[0] != len(draft_tokens):
                        raise ValueError("Prefix risk requires logits from this Draft forward")
                    values = draft_confidence(logits)
                else:
                    values = [[0., 0., 0.] for _ in draft_tokens]
                baseline, selected, curve = self.policy.select(pool, values)
                self.stats["risk_evaluations"] += 1
                self.stats["risk_switches"] += int(selected != baseline)
                self.stats["risk_unsafe_unchanged"] += int(selected == baseline and curve[pool[baseline]["prefix_length"] - 1] < .5)
            else:
                self.stats["risk_single_boundary"] += 1
            if self.record_decisions:
                self.decisions.append(dict(generated=len(confirmed), draft_tokens=list(draft_tokens), confidence=values,
                    candidates=[{k: list(c[k]) if k == "tokens" else c[k]
                                 for k in ("tokens", "prefix_length", "cheap", "position", "source")} for c in pool],
                    baseline=baseline, selected=selected, survival=curve))
            item = pool[selected]
            self.stats["alignment_proposals"] += 1
            self.stats["splice_proposals"] += 1
            self.stats["splice_internal_proposals"] += int(item["prefix_length"] > self.hint_tokens)
            return SpliceProposal(item["tokens"], item["source"], 0, item["position"], item["end"],
                                  self.lookup, True, item["prefix_length"], len(draft_tokens))
        finally:
            self.stats["alignment_seconds"] += time.perf_counter() - started
