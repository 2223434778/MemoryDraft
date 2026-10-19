import numpy as np
import pytest

from memorydraft.memory.store import MemoryEvent, Snapshot
from memorydraft.retrieval.matrix import MatrixReuseMatcher, assess_alignment, local_order_penalty, plan_features
from memorydraft.retrieval.splice import tail_transport_score
from memorydraft.retrieval.suffix import SuffixArrayIndex
from memorydraft.retrieval.transport import AlignmentConfig, transport_score


def test_raw_and_tail_ablation_reproduce_existing_scores():
    ot = pytest.importorskip("ot")
    query = [*range(10), 201, 202, *range(10, 22)]
    history = list(range(24))
    cfg = AlignmentConfig()
    raw = assess_alignment(query, history, cfg, ot.sinkhorn, draft_prefix=7)
    tail = assess_alignment(query, history, cfg, ot.sinkhorn, draft_prefix=7, tail=True)
    assert raw.score == pytest.approx(transport_score(query, history, cfg, ot.sinkhorn), abs=1e-12)
    assert tail.score == pytest.approx(tail_transport_score(query, history, cfg, ot.sinkhorn), abs=1e-12)


def test_plan_features_distinguish_same_matched_mass_with_reversed_order():
    n = 12
    a = np.r_[np.full(n, .5 / n), .5]
    diagonal = np.diag(a)
    reversed_prefix = diagonal.copy()
    reversed_prefix[:8, :8] = np.fliplr(reversed_prefix[:8, :8])
    equal = np.ones((n, n), dtype=bool)
    good = plan_features(diagonal, equal, a, draft_prefix=4)
    bad = plan_features(reversed_prefix, equal, a, draft_prefix=4)
    assert good["support"] == bad["support"] == 1
    assert good["concentration"] == bad["concentration"] == 1
    assert good["continuity"] == 1 and bad["continuity"] == 0
    assert local_order_penalty(diagonal[:n, :n]).diagonal()[1:-1].max() == 0
    assert local_order_penalty(reversed_prefix[:n, :n]).min() >= 0


def test_forced_anchor_is_not_counted_as_pre_anchor_reliability():
    n = 12
    a = np.r_[np.full(n, .5 / n), .5]
    plan = np.zeros((n + 1, n + 1))
    plan[np.arange(8, n), np.arange(8, n)] = a[8:n]
    plan[:8, n], plan[n, :8], plan[n, n] = a[:8], a[:8], .5 - a[:8].sum()
    features = plan_features(plan, np.eye(n, dtype=bool), a, draft_prefix=4)
    assert features["support"] == features["concentration"] == features["continuity"] == 0
    assert features["dropped"] == 1
    result = assess_alignment(list(range(n)), list(range(n)), AlignmentConfig(),
                              lambda *a, **k: plan, draft_prefix=4, structure=True)
    assert not result.admissible and result.score == float("-inf")


def test_order_pass_changes_only_real_cost_and_validates_each_plan():
    ot = pytest.importorskip("ot")
    costs = []

    def solver(a, b, cost, *args, **kwargs):
        costs.append(cost.copy())
        return ot.sinkhorn(a, b, cost, *args, **kwargs)

    result = assess_alignment(list(range(16)), list(range(16)), AlignmentConfig(), solver,
                              draft_prefix=4, order=True)
    assert result.admissible and len(costs) == 2
    np.testing.assert_array_equal(costs[0][-1], costs[1][-1])
    np.testing.assert_array_equal(costs[0][:, -1], costs[1][:, -1])
    assert np.any(costs[1][:-1, :-1] > costs[0][:-1, :-1])
    invalid = assess_alignment(list(range(8)), list(range(8)), AlignmentConfig(),
                               lambda *a, **k: np.zeros((9, 9)), draft_prefix=4, order=True)
    assert not invalid.admissible


@pytest.mark.parametrize("variant", ["order", "plan", "all"])
def test_matrix_candidates_keep_raw_history_bounds_and_existing_budget(variant):
    pytest.importorskip("ot")

    class Tokens:
        all_special_ids = [999]

        def encode(self, text, add_special_tokens=False):
            return list(map(int, text.split()))

    index = SuffixArrayIndex(Snapshot("u", 10, (MemoryEvent("u", "old", 1,
                            " ".join(map(str, range(1, 81)))),)), Tokens())
    matcher = MatrixReuseMatcher(index, variant=variant)
    prefix = [*range(1, 20), 200]
    p = matcher.propose_with_draft(prefix, list(range(21, 36)), 32)
    assert p is not None and p.tokens == tuple(range(21, 53))
    matcher.observe(p, 0, prefix)
    assert matcher.stats["splice_zero_history"] == 1
    assert matcher.stats["matrix_solves"] <= 2 * matcher.stats["alignment_candidates"]
    assert matcher.stats["guided_lookups"] == 1
    with pytest.raises(ValueError):
        MatrixReuseMatcher(index, variant="unknown")
