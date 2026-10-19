"""Empirical cost routing over raw Draft/history splices, before any OT solve.

The predictor uses previously verified prefix lengths, never OT plan features.
Its renewal-rate estimate is a development heuristic, not a latency guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

from memorydraft.retrieval.splice import SpliceProposal, SpliceReuseMatcher


def pool_features(pool, draft):
    """Only cheap, currently observable values; no IDs, answers or OT scores."""
    top = pool[0]
    agreement = next((i for i, (a, b) in enumerate(zip(top["tokens"], draft)) if a != b),
                     min(len(top["tokens"]), len(draft)))
    return [top["cheap"], top["cheap"] - pool[1]["cheap"] if len(pool) > 1 else 1.,
            top["prefix_length"] / 32, agreement / 32, len(pool) / 8,
            min(c["prefix_length"] for c in pool) / 32]


def interpolate(curve, length):
    points = sorted((int(k), v) for k, v in curve.items())
    if length <= points[0][0]:
        return points[0][1]
    for (a, x), (b, y) in zip(points, points[1:]):
        if length <= b:
            return x + (y - x) * (length - a) / (b - a)
    return points[-1][1]


@dataclass(frozen=True)
class BenefitPolicy:
    samples: list
    costs: dict
    neighbours: int = 3
    lengths: tuple = (16, 24, 32)
    minimum_gain: float = .02

    def __post_init__(self):
        if (type(self.neighbours) is not int or self.neighbours < 1 or not self.samples
                or tuple(self.lengths) != (16, 24, 32)
                or not math.isfinite(self.minimum_gain) or not 0 <= self.minimum_gain <= 1):
            raise ValueError("Invalid benefit policy")
        for sample in self.samples:
            if (not isinstance(sample.get("project"), str) or len(sample.get("features", [])) != 6
                    or not all(math.isfinite(x) and 0 <= x <= 1 for x in sample["features"])
                    or set(sample.get("outcomes", {})) != {"draft", "edit", "ot"}):
                raise ValueError("Invalid calibration sample")
            for accepted, proposed in sample["outcomes"].values():
                if not all(type(x) is int for x in (accepted, proposed)) or not 0 <= accepted <= proposed <= 32:
                    raise ValueError("Invalid verified calibration prefix")
        if not self.costs.get("profiles") or set(self.costs) != {"profiles", "lookup", "ot_candidate"}:
            raise ValueError("Missing measured costs")
        values = [self.costs["lookup"], self.costs["ot_candidate"]]
        for p in self.costs["profiles"]:
            if type(p["context"]) is not int or p["context"] < 1 or set(p["verify"]) != {"15", "16", "24", "32"}:
                raise ValueError("Invalid cost profile")
            values += [p["draft"], *p["verify"].values()]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in values):
            raise ValueError("Costs must be finite and positive")

    def predict(self, pool, draft, budget, context, *, allow_ot=True, adaptive=True):
        features = pool_features(pool, draft)
        return self.predict_features(features, len(draft), budget, context, allow_ot=allow_ot, adaptive=adaptive)

    def predict_features(self, features, draft_length, budget, context, *, allow_ot=True, adaptive=True):
        # A project with many nearby states must not dominate the small pilot.
        nearest = {}
        for s in self.samples:
            distance = sum((a - b) ** 2 for a, b in zip(features, s["features"]))
            if s["project"] not in nearest or distance < nearest[s["project"]][0]:
                nearest[s["project"]] = (distance, s)
        selected = sorted(nearest.values(), key=lambda v: (v[0], v[1]["project"]))[:self.neighbours]
        weights = [1 / (.25 + distance) for distance, _ in selected]
        profile = min(self.costs["profiles"], key=lambda p: abs(p["context"] - context))
        options = []
        for route in ("draft", "edit", "ot"):
            if route == "ot" and (not allow_ot or features[4] == 1 / 8 or len(selected) < self.neighbours):
                continue
            lengths = [draft_length] if route == "draft" else sorted({min(n, budget) for n in (self.lengths if adaptive else (32,))})
            for length in lengths:
                if route != "draft" and length <= draft_length:
                    continue
                advance = sum(w * (1 + min(s["outcomes"][route][0], length))
                              for w, (_, s) in zip(weights, selected)) / sum(weights)
                seconds = (profile["draft"] + interpolate(profile["verify"], length) + self.costs["lookup"]
                           + (self.costs["ot_candidate"] * features[4] * 8 if route == "ot" else 0))
                options.append(dict(route=route, length=length, advance=advance, seconds=seconds, rate=advance / seconds))
        cheap = max((o for o in options if o["route"] != "ot"), key=lambda o: (o["rate"], -o["length"], o["route"] == "draft"))
        expensive = max((o for o in options if o["route"] == "ot"), key=lambda o: (o["rate"], -o["length"]), default=None)
        choice = expensive if expensive and expensive["rate"] > cheap["rate"] * (1 + self.minimum_gain) else cheap
        return dict(choice=choice, features=features, projects=[s["project"] for _, s in selected],
                    options=options, profile_context=profile["context"])


class BenefitReuseMatcher(SpliceReuseMatcher):
    def __init__(self, index, *, policy: BenefitPolicy, prompt_tokens=0, allow_ot=True, adaptive=True, **kwargs):
        super().__init__(index, method="ot" if allow_ot else "edit", tail_weighted=allow_ot, **kwargs)
        self.policy, self.prompt_tokens = policy, prompt_tokens
        self.allow_ot, self.adaptive = allow_ot, adaptive
        self.decisions = []
        self.stats.update(benefit_decisions=0, benefit_ot_calls=0, benefit_ot_candidates=0,
                          benefit_draft=0, benefit_edit=0, benefit_short=0)

    def propose_with_draft(self, confirmed, draft_tokens, max_tokens):
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
            decision = self.policy.predict(pool, draft_tokens, max_tokens, self.prompt_tokens + len(confirmed) - 1,
                                           allow_ot=self.allow_ot, adaptive=self.adaptive)
            choice = decision["choice"]
            self.decisions.append(dict(generated=len(confirmed), **decision))
            self.stats["benefit_decisions"] += 1
            route = choice["route"]
            if route == "draft":
                self.stats["benefit_draft"] += 1
                return None
            self.stats["alignment_calls"] += 1
            ranked = []
            if route == "ot":
                self.stats["benefit_ot_calls"] += 1
                self.stats["benefit_ot_candidates"] += len(pool)
            else:
                self.stats["benefit_edit"] += 1
            for item in pool:
                self.stats["alignment_candidates"] += 1
                score = self.candidate_score(item) if route == "ot" else item["cheap"]
                ranked.append((score, -item["prefix_length"], -item["position"], item))
            score, _, _, item = max(ranked, key=lambda x: x[:3])
            if score < self.config.minimum_score:
                return None
            tokens = item["tokens"][:choice["length"]]
            self.stats["benefit_short"] += int(len(tokens) < min(max_tokens, len(item["tokens"])))
            self.stats["alignment_proposals"] += 1
            self.stats["splice_proposals"] += 1
            self.stats["splice_internal_proposals"] += int(item["prefix_length"] > self.hint_tokens)
            return SpliceProposal(tokens, item["source"], 0, item["position"], item["end"],
                                  "benefit-" + route + "-splice", True, item["prefix_length"], len(draft_tokens))
        finally:
            self.stats["alignment_seconds"] += time.perf_counter() - started
