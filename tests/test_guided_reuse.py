from dataclasses import replace

import pytest

from memorydraft.memory.store import MemoryEvent, MemoryStore, Snapshot
from memorydraft.retrieval.guided import GuidedReuseMatcher
from memorydraft.retrieval.suffix import SuffixArrayIndex
from memorydraft.retrieval.transport import AlignmentConfig


class Tokens:
    all_special_ids = [999]

    def encode(self, text, add_special_tokens=False):
        return list(map(int, text.split()))


def index():
    return SuffixArrayIndex(Snapshot("u", 10, (MemoryEvent("u", "old", 1,
                            " ".join(map(str, range(1, 70)))),)), Tokens())


@pytest.mark.parametrize("method", ["edit", "ot"])
def test_hint_disambiguates_endpoint_after_insertion_and_preserves_raw_span(method):
    if method == "ot":
        pytest.importorskip("ot")
    matcher = GuidedReuseMatcher(index(), method=method)
    prefix = list(range(1, 21)) + [201, 202, 203, 204, 205]
    assert matcher.propose(prefix, 32) is None
    proposal = matcher.propose_with_draft(prefix, list(range(21, 36)), 32)
    assert proposal.tokens == tuple(range(21, 53))
    assert proposal.matched == 0 and proposal.context_qualified
    assert proposal.lookup == method + "-guided"
    matcher.observe(proposal, 3, prefix + list(proposal.tokens[:3]))
    assert matcher.stats["alignment_accepted_tokens"] == 3


def test_absent_hint_and_short_continuation_cannot_replace_draft():
    matcher = GuidedReuseMatcher(index(), method="edit")
    assert matcher.propose_with_draft(list(range(1, 20)), [300] * 15, 32) is None
    assert matcher.stats["alignment_calls"] == 0
    # The hint exists near the end, but cannot yield a longer verification block.
    assert matcher.propose_with_draft(list(range(40, 60)), list(range(60, 75)), 32) is None
    assert matcher.propose_with_draft(list(range(1, 20)), list(range(20, 35)), 15) is None
    assert matcher.stats["alignment_proposals"] == 0


def test_lookup_budget_bounds_misses_and_exact_path_does_not_spend_it():
    matcher = GuidedReuseMatcher(index(), method="edit", config=AlignmentConfig(max_fallback_calls=2))
    assert matcher.propose(list(range(1, 21)), 32).lookup == "global"
    assert matcher.stats["guided_lookups"] == 0
    for _ in range(5):
        assert matcher.propose_with_draft(list(range(1, 20)), [300] * 15, 32) is None
    assert matcher.stats["guided_lookups"] == 2


def test_guided_candidates_respect_user_time_record_and_special_boundaries(tmp_path):
    with MemoryStore(tmp_path / "history.db") as store:
        store.append([MemoryEvent("u", "old", 1, " ".join(map(str, range(1, 45))) + " 999 55 56"),
                      MemoryEvent("u", "future", 20, " ".join(map(str, range(1, 90)))),
                      MemoryEvent("private", "other", 1, " ".join(map(str, range(1, 100))))])
        source = SuffixArrayIndex(store.snapshot("u", 10), Tokens())
    matcher = GuidedReuseMatcher(source, method="edit")
    prefix = list(range(1, 21))
    prefix[-1] = 200
    proposal = matcher.propose_with_draft(prefix, list(range(21, 36)), 32)
    assert proposal.source == "old" and proposal.tokens == tuple(range(21, 45))
    assert not {999, 55, 56} & set(proposal.tokens)


@pytest.mark.parametrize("hint_tokens", [0, True, 33])
def test_invalid_hint_length_rejected(hint_tokens):
    with pytest.raises(ValueError):
        GuidedReuseMatcher(index(), method="edit", hint_tokens=hint_tokens)
