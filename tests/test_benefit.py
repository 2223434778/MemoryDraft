from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest

from memorydraft.evaluation.benefit_probe import calibration, policy_data
from memorydraft.retrieval.benefit import BenefitPolicy, BenefitReuseMatcher, interpolate, pool_features
from memorydraft.runtime.config import RuntimeConfig
from test_splice_reuse import index


def data(ot=32, edit=6, draft=6, expensive=0.001):
    return dict(samples=[dict(project=str(i), features=[.9, .1, .25, .25, .25, .125],
                              outcomes=dict(draft=[draft, 15], edit=[edit, 32], ot=[ot, 32])) for i in range(4)],
                costs=dict(profiles=[dict(context=512, draft=.02, verify={"15": .03, "16": .031, "24": .04, "32": .05})],
                           lookup=.001, ot_candidate=expensive))


def pool():
    return [dict(cheap=.9, prefix_length=8, tokens=tuple(range(32)), position=20, end=80, source="old"),
            dict(cheap=.8, prefix_length=4, tokens=tuple(range(1, 33)), position=30, end=80, source="old")]


def test_gate_accounts_for_ot_cost_and_empirical_gain():
    draft = list(range(15))
    cheap = BenefitPolicy(**data()).predict(pool(), draft, 32, 512)
    assert cheap["choice"]["route"] == "ot"
    expensive = BenefitPolicy(**data(expensive=.5)).predict(pool(), draft, 32, 512)
    assert expensive["choice"]["route"] == "draft"
    one = BenefitPolicy(**data()).predict(pool()[:1], draft, 32, 512)
    assert all(o["route"] != "ot" for o in one["options"])
    assert all(0 <= f <= 1 for f in pool_features(pool(), draft))


def test_length_uses_survival_and_target_cost_not_longest_available_span():
    policy = BenefitPolicy(**data(ot=17, edit=17, draft=4))
    decision = policy.predict(pool(), list(range(15)), 32, 512)
    assert decision["choice"]["route"] == "edit" and decision["choice"]["length"] == 16
    assert policy.predict(pool(), list(range(15)), 20, 512)["choice"]["length"] <= 20
    assert policy.predict(pool(), list(range(15)), 32, 512, adaptive=False)["choice"]["length"] == 32
    assert interpolate({"15": .03, "32": .05}, 15) == .03
    assert interpolate({"15": .03, "32": .05}, 40) == .05


def test_matcher_skips_solver_before_gate_and_preserves_raw_history():
    matcher = BenefitReuseMatcher(index(), policy=BenefitPolicy(**data(ot=30, edit=30, draft=0)), allow_ot=False)
    matcher.solver = Mock(side_effect=AssertionError("OT was not authorized by the policy"))
    confirmed, draft = list(range(1, 21)), [200, 201, *range(21, 34)]
    proposal = matcher.propose_with_draft(confirmed, draft, 32)
    assert proposal and proposal.lookup == "benefit-edit-splice"
    assert proposal.tokens[:proposal.draft_prefix] == tuple(draft[:proposal.draft_prefix])
    assert proposal.tokens[proposal.draft_prefix:] == tuple(index().tokens[proposal.position:proposal.position + len(proposal.tokens) - proposal.draft_prefix])
    assert matcher.stats["benefit_ot_calls"] == 0
    assert matcher.propose(confirmed, 32).lookup == "global"
    matcher.observe(proposal, 0, confirmed)
    assert matcher.blocked is None


def test_ot_runs_only_after_gate_and_has_its_own_cost_counter():
    matcher = BenefitReuseMatcher(index(), policy=BenefitPolicy(**data()))
    matcher.splice_shortlist = lambda *a: pool()
    matcher.candidate_score = Mock(side_effect=[.8, .9])
    p = matcher.propose_with_draft(list(range(20)), list(range(15)), 32)
    assert p and p.position == 30 and len(p.tokens) == 32
    assert matcher.stats["benefit_ot_calls"] == 1 and matcher.stats["benefit_ot_candidates"] == 2
    assert matcher.candidate_score.call_count == 2


def test_calibration_excludes_whole_project_and_runtime_requires_measured_policy():
    from pathlib import Path

    samples = calibration(Path(__file__).resolve().parents[1])
    assert len(samples) == 14
    project = samples[0]["project"]
    policy = policy_data(samples, data()["costs"], project)
    assert all(s["project"] != project for s in policy["samples"])
    config = RuntimeConfig(backend="demo", retrieval="ot-budget-splice", benefit=policy)
    assert config.benefit == policy
    with pytest.raises(ValueError):
        replace(config, benefit={})
    with pytest.raises(ValueError):
        replace(config, retrieval="suffix")


@pytest.mark.parametrize("change", [lambda d: d.update(samples=[]),
                                   lambda d: d["costs"].update(ot_candidate=float("nan")),
                                   lambda d: d["samples"][0]["outcomes"].update(ot=[33, 32]),
                                   lambda d: d["samples"][0]["features"].append(0)])
def test_rejects_invalid_or_missing_calibration(change):
    config = deepcopy(data())
    change(config)
    with pytest.raises(ValueError):
        BenefitPolicy(**config)
