from copy import deepcopy

import numpy as np
import pytest

from memorydraft.retrieval.acceptance import AcceptancePredictor, prefix_features, tail_features


def state(project="a", accepted=15, p=.9):
    return dict(project_id=project, draft_tokens=list(range(15)), draft_accepted=accepted,
                confidence=[[p, 1-p, p/2] for _ in range(15)],
                candidates=[dict(prefix_length=6, cheap=.8, source="history", tokens=list(range(32)),
                                 token_probability=[p] * 15, accepted=32 if accepted==15 else accepted)])


def test_shared_prefix_conditional_history_and_monotone_survival():
    predictor = AcceptancePredictor([0.] * 6, [0.] * 9, True, ("x",))
    s = state()
    draft = predictor.draft_survival(s)
    candidate = predictor.candidate_survival(s, s["candidates"][0])
    assert candidate[:6] == draft[:6] == [.5] * 6
    assert candidate[6:] == [.25] * 26
    assert all(a >= b for a, b in zip(candidate, candidate[1:]))


def test_features_cannot_read_answers_or_project_identity():
    a, b = state(), state("other", accepted=0)
    b["candidates"][0]["accepted"] = 0
    assert prefix_features(a, 8, True) == prefix_features(b, 8, True)
    assert tail_features(a, a["candidates"][0], 20, True) == tail_features(b, b["candidates"][0], 20, True)


def test_project_exclusion_and_confidence_ablation():
    states = [state(p, accepted=15, p=.99) for p in "abcd"] + [state(p, accepted=1, p=.2) for p in "abcd"]
    fitted = AcceptancePredictor.fit(states, excluded_project="a")
    changed = deepcopy(states)
    for s in changed:
        if s["project_id"] == "a":
            s["draft_accepted"] = 9
    other = AcceptancePredictor.fit(changed, excluded_project="a")
    assert fitted == other and "a" not in fitted.projects
    assert sum(fitted.draft_survival(state(p=.99))) > sum(fitted.draft_survival(state(p=.2)))
    ablated = AcceptancePredictor.fit(states, excluded_project="a", confidence=False)
    np.testing.assert_allclose(ablated.draft_survival(state(p=.99)), ablated.draft_survival(state(p=.2)))


def test_censored_full_acceptance_does_not_invent_a_tail_rejection():
    fitted = AcceptancePredictor.fit([state(p) for p in "abcd"], excluded_project="a")
    assert fitted.candidate_survival(state(), state()["candidates"][0])[-1] > .95
    with pytest.raises(ValueError):
        AcceptancePredictor.fit([state("a")], excluded_project="a")
