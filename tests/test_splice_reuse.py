import numpy as np
import pytest

from memorydraft.memory.store import MemoryEvent, MemoryStore, Snapshot
from memorydraft.retrieval.guided import GuidedReuseMatcher
from memorydraft.retrieval.splice import SpliceReuseMatcher, tail_transport_score
from memorydraft.retrieval.suffix import SuffixArrayIndex
from memorydraft.retrieval.transport import AlignmentConfig


class Tokens:
    all_special_ids = [999]

    def encode(self, text, add_special_tokens=False):
        return list(map(int, text.split()))


def index():
    return SuffixArrayIndex(Snapshot("u", 10, (MemoryEvent("u", "old", 1,
                            " ".join(map(str, range(1, 81)))),)), Tokens())


@pytest.mark.parametrize("method,tail", [("edit", False), ("ot", False), ("ot", True)])
def test_internal_anchor_retains_new_tokens_and_deduplicates_identical_splices(method, tail):
    if method == "ot":
        pytest.importorskip("ot")
    confirmed, draft = list(range(1, 21)), [200, 201, *range(21, 34)]
    assert GuidedReuseMatcher(index(), method="edit").propose_with_draft(confirmed, draft, 32) is None
    matcher = SpliceReuseMatcher(index(), method=method, tail_weighted=tail)
    pool = matcher.splice_shortlist(confirmed, draft, 32)
    assert len(pool) == 1
    p = matcher.propose_with_draft(confirmed, draft, 32)
    assert p.tokens == (200, 201, *range(21, 51))
    assert 4 < p.draft_prefix <= len(draft) and p.original_draft == len(draft)
    assert p.tokens[:p.draft_prefix] == tuple(draft[:p.draft_prefix])
    assert p.tokens[p.draft_prefix:] == tuple(index().tokens[p.position:p.position + 32 - p.draft_prefix])
    matcher.observe(p, p.draft_prefix + 3, confirmed + list(p.tokens[:p.draft_prefix + 3]))
    assert matcher.stats["splice_history_accepted_tokens"] == 3
    assert matcher.stats["splice_prefix_accepted_tokens"] == p.draft_prefix
    assert matcher.stats["splice_internal_proposals"] == 1
    assert matcher.blocked[1] == p.position + 3


def test_rejection_in_retained_draft_does_not_invent_historical_feedback():
    matcher = SpliceReuseMatcher(index(), method="edit")
    prefix, draft = list(range(1, 21)), [200, 201, *range(21, 34)]
    p = matcher.propose_with_draft(prefix, draft, 32)
    matcher.observe(p, 1, prefix + [200])
    assert matcher.blocked is None
    assert matcher.stats["splice_history_accepted_tokens"] == 0
    assert matcher.stats["splice_extension_accepted_tokens"] == 0
    assert matcher.stats["splice_zero_history"] == 1
    with pytest.raises(ValueError):
        matcher.observe(p, len(p.tokens) + 1, prefix)


def test_bounds_budget_and_exact_suffix_are_preserved(tmp_path):
    matcher = SpliceReuseMatcher(index(), method="edit", config=AlignmentConfig(max_fallback_calls=1))
    assert matcher.propose(list(range(1, 21)), 32).lookup == "global"
    assert matcher.stats["guided_lookups"] == 0
    assert matcher.propose_with_draft(list(range(1, 21)), [300] * 15, 32) is None
    assert matcher.propose_with_draft(list(range(1, 21)), list(range(21, 36)), 32) is None
    assert matcher.stats["guided_lookups"] == 1
    matcher = SpliceReuseMatcher(index(), method="edit")
    assert matcher.propose_with_draft(list(range(1, 21)), list(range(21, 36)), 15) is None
    assert matcher.propose_with_draft(list(range(1, 21)), [1] * 33, 40) is None
    with MemoryStore(tmp_path / "history.db") as store:
        store.append([MemoryEvent("u", "old", 1, " ".join(map(str, range(1, 45))) + " 999 55 56"),
                      MemoryEvent("u", "future", 20, " ".join(map(str, range(1, 90)))),
                      MemoryEvent("other", "private", 1, " ".join(map(str, range(1, 100))))])
        source = SuffixArrayIndex(store.snapshot("u", 10), Tokens())
    matcher = SpliceReuseMatcher(source, method="edit")
    p = matcher.propose_with_draft(list(range(1, 21)), [200, 201, *range(21, 34)], 32)
    assert p.source == "old" and p.tokens[-1] == 44
    assert not {999, 55, 56} & set(p.tokens)


def test_tail_emphasis_and_invalid_solver_plan():
    ot = pytest.importorskip("ot")
    q = list(range(32))
    early, late = q.copy(), q.copy()
    early[3], late[25] = 200, 200
    cfg = AlignmentConfig()
    assert tail_transport_score(q, early, cfg, ot.sinkhorn) > tail_transport_score(q, late, cfg, ot.sinkhorn)
    assert tail_transport_score([], q, cfg, ot.sinkhorn) == 0
    assert tail_transport_score(q, q, cfg, lambda *a, **k: np.zeros((33, 33))) == float("-inf")
    with pytest.raises(ValueError):
        SpliceReuseMatcher(index(), method="edit", tail_weighted=True)
