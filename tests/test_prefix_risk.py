from dataclasses import replace

import numpy as np
import pytest
import torch

from memorydraft.retrieval.prefix_risk import PrefixRiskPolicy, PrefixRiskReuseMatcher, draft_confidence
from memorydraft.retrieval.splice import SpliceReuseMatcher
from memorydraft.runtime.config import RuntimeConfig
from test_splice_reuse import index


def pool():
    return [dict(tokens=tuple(range(32)), prefix_length=14, cheap=.95, position=20, end=80, source="old"),
            dict(tokens=tuple(range(32, 64)), prefix_length=7, cheap=.8, position=30, end=80, source="old")]


def test_only_switches_unsafe_baseline_to_safe_existing_handoff():
    policy = PrefixRiskPolicy([4., -8., 0., 0., 0., 0.], confidence=False)
    baseline, selected, curve = policy.select(pool(), [[0., 0., 0.]] * 15)
    assert (baseline, selected) == (0, 1) and curve[6] >= .5 > curve[13]
    assert policy.select(pool(), [[0., 0., 0.]] * 15)[2] == sorted(curve, reverse=True)
    assert PrefixRiskPolicy([10., 0., 0., 0., 0., 0.]).select(pool(), [[.9, .1, .8]] * 15)[1] == 0
    assert PrefixRiskPolicy([-10., 0., 0., 0., 0., 0.]).select(pool(), [[.9, .1, .8]] * 15)[1] == 0


def test_confidence_matches_saved_feature_definition_without_mutating_logits():
    logits = torch.randn(15, 97)
    original = logits.clone()
    values = draft_confidence(logits)
    probabilities = torch.softmax(logits, -1)
    expected_entropy = -(probabilities * probabilities.log()).sum(-1) / np.log(97)
    np.testing.assert_allclose(np.asarray(values)[:, 1], expected_entropy.numpy(), rtol=1e-6)
    assert torch.equal(original, logits)
    with pytest.raises(ValueError, match="invalid"):
        draft_confidence(torch.full((15, 10), float("nan")))


def test_matcher_single_boundary_skips_confidence_and_preserves_raw_splice():
    policy = PrefixRiskPolicy([0.] * 6)
    matcher = PrefixRiskReuseMatcher(index(), policy=policy)
    confirmed, draft = list(range(1, 21)), [200, 201, *range(21, 34)]
    original = SpliceReuseMatcher(index(), method="edit").propose_with_draft(confirmed, draft, 32)
    actual = matcher.propose_with_draft(confirmed, draft, 32)
    assert actual.tokens == original.tokens and actual.draft_prefix == original.draft_prefix
    assert matcher.stats["risk_evaluations"] == 0 and matcher.stats["risk_single_boundary"] == 1
    matcher.observe(actual, 1, confirmed + list(actual.tokens[:1]))
    assert matcher.blocked is None


def test_matcher_consumes_current_logits_and_filters_weak_history(monkeypatch):
    matcher = PrefixRiskReuseMatcher(index(), policy=PrefixRiskPolicy([4., -8., 0., 0., 0., 0.]))
    monkeypatch.setattr(matcher, "splice_shortlist", lambda *a: pool())
    with pytest.raises(ValueError, match="this Draft"):
        matcher.propose_with_draft(list(range(20)), list(range(15)), 32)
    matcher.set_draft_logits(torch.randn(15, 97))
    p = matcher.propose_with_draft(list(range(20)), list(range(15)), 32)
    assert p.tokens == pool()[1]["tokens"] and p.draft_prefix == 7 and len(p.tokens) == 32
    assert matcher._logits is None and matcher.stats["risk_switches"] == 1
    weak = pool()
    weak[1]["cheap"] = .4
    monkeypatch.setattr(matcher, "splice_shortlist", lambda *a: weak)
    assert matcher.propose_with_draft(list(range(20)), list(range(15)), 32).draft_prefix == 14


def test_runtime_requires_explicit_valid_frozen_policy():
    cfg = RuntimeConfig(backend="demo")
    with pytest.raises(ValueError, match="frozen"):
        replace(cfg, retrieval="edit-risk-splice")
    with pytest.raises(ValueError, match="requires edit-risk"):
        replace(cfg, prefix_risk=dict(weights=[0.] * 6))
    assert replace(cfg, retrieval="edit-risk-splice", prefix_risk=dict(weights=[0.] * 6)).retrieval == "edit-risk-splice"
    with pytest.raises(ValueError, match="Invalid"):
        PrefixRiskPolicy([float("inf")] * 6)
