from dataclasses import replace

import numpy as np
import pytest

from memorydraft.memory.store import MemoryEvent, MemoryStore, Snapshot
from memorydraft.retrieval.reuse import ReuseMatcher
from memorydraft.retrieval.suffix import SuffixArrayIndex
from memorydraft.retrieval.transport import AlignmentConfig, AlignedReuseMatcher, edit_score, transport_score


class Tokens:
    all_special_ids = [999]

    def encode(self, text, add_special_tokens=False):
        return [int(value) for value in text.split()]


def index(text=None):
    text = text or " ".join(map(str, range(1, 45)))
    return SuffixArrayIndex(Snapshot("u", 10, (MemoryEvent("u", "old", 1, text),)), Tokens())


@pytest.mark.parametrize("method", ["edit", "ot"])
def test_realign_changed_context_copies_original_span_and_keeps_real_suffix_length(method):
    if method == "ot":
        pytest.importorskip("ot")
    source = index()
    prefix = list(range(1, 25))
    prefix[-2] = 199
    matcher = AlignedReuseMatcher(source, method=method)
    assert ReuseMatcher(source).propose(prefix, 8) is None
    proposal = matcher.propose(prefix, 8)
    assert proposal.tokens == tuple(range(25, 33))
    assert proposal.context_qualified and proposal.matched == 1 and proposal.lookup == method
    matcher.observe(proposal, 2, [*prefix, *proposal.tokens[:2]])
    assert matcher.stats["alignment_accepted_tokens"] == 2


@pytest.mark.parametrize("method", ["edit", "ot"])
def test_exact_fast_path_and_budget_are_distinct(method):
    if method == "ot":
        pytest.importorskip("ot")
    matcher = AlignedReuseMatcher(index(), method=method, config=AlignmentConfig(max_fallback_calls=2))
    proposal = matcher.propose(list(range(1, 20)), 7)
    assert not proposal.context_qualified and proposal.lookup == "global"
    assert matcher.stats["alignment_calls"] == 0
    assert matcher.propose([300] * 7, 8) is None
    for _ in range(4):
        assert matcher.propose([300] * 12, 8) is None
    assert matcher.stats["alignment_calls"] == 2


def test_alignment_never_crosses_special_record_user_or_time_boundary(tmp_path):
    with MemoryStore(tmp_path / "events.db") as store:
        store.append([MemoryEvent("u", "old", 1, " ".join(map(str, range(1, 23))) + " 999 40 41"),
                      MemoryEvent("u", "future", 20, "1 2 3 4 5 6 7 8 9 10 11 12 888"),
                      MemoryEvent("other", "private", 1, "1 2 3 4 5 6 7 8 9 10 11 12 777")])
        source = SuffixArrayIndex(store.snapshot("u", 10), Tokens())
    matcher = AlignedReuseMatcher(source, method="edit")
    prefix = list(range(1, 17))
    prefix[-2] = 199
    proposal = matcher.propose(prefix, 32)
    assert proposal.tokens == tuple(range(17, 23)) and proposal.source == "old"
    assert not {999, 888, 777, 40, 41} & set(proposal.tokens)
    assert all(position < source.ends[0] for position in matcher.shortlist(prefix))


def test_ot_identity_edits_and_invalid_solver_fail_closed():
    pytest.importorskip("ot")
    query = tuple(range(24))
    config = AlignmentConfig()
    identity = transport_score(query, query, config)
    changed = (*query[:10], 199, *query[10:])
    assert identity > transport_score(query, changed, config) > config.minimum_score
    assert transport_score(query, tuple(range(100, 124)), config) < config.minimum_score
    assert transport_score(query, query, config,
                           lambda a, b, cost, *args, **kwargs: np.full_like(cost, np.nan)) == -float("inf")
    assert transport_score(query, query, config,
                           lambda a, b, cost, *args, **kwargs: np.zeros_like(cost)) == -float("inf")
    assert edit_score(query, changed) == 1 - 1 / 25


@pytest.mark.parametrize("setting", [{"window": 129}, {"shortlist": 0}, {"iterations": 501},
                                     {"entropy": float("nan")}, {"minimum_score": -1}, {"max_shift": -1}])
def test_invalid_budget_rejected(setting):
    with pytest.raises(ValueError):
        AlignmentConfig(**setting)
