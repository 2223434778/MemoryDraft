import pytest

from memorydraft.backends.candidate_repair import SuffixRepair
from memorydraft.backends.dflash_reuse import ReuseConfig, generate_reuse
from tests.test_dflash_reuse import CheckedDraft, tiny


def test_only_suffix_after_rejected_token_is_retried_once():
    repair = SuffixRepair(4)
    fresh = repair.propose_with_draft([], [10, 11, 12, 13, 14, 15], 6)
    repair.observe(fresh, 1, [10])
    retry = repair.propose([10, 99], 10)
    assert retry.tokens == (12, 13, 14, 15)
    assert retry.lookup == "suffix-repair"
    repair.observe(retry, 0, [10, 99])
    assert repair.propose([], 10) is None


def test_too_short_suffix_or_all_accepted_does_not_retry():
    repair = SuffixRepair(8)
    fresh = repair.propose_with_draft([], [1, 2, 3], 3)
    for accepted in (1, 2, 3):
        repair.observe(fresh, accepted, [])
        assert repair.propose([], 10) is None
    repair.observe(fresh, 0, [])
    assert repair.propose([], 1) is None


@pytest.mark.parametrize("budget", [0, 1, 2, 9, 29])
def test_live_target_repair_keeps_correct_features_catchup_and_stream(tiny, budget):
    published = []
    draft = CheckedDraft(tiny.features)
    result = generate_reuse(tiny.target, draft, tiny.ids, max_new_tokens=budget,
                            matcher=SuffixRepair(4), config=ReuseConfig(cooldown_rounds=0),
                            on_tokens=published.extend)
    assert result.sequences.tolist() == tiny.reference[:, :4+budget].tolist()
    assert published == tiny.reference[0, 4:4+budget].tolist()
    if budget == 29:
        assert any(r["lookup"] == "suffix-repair" for r in result.rounds)
        assert any(end-start > 1 for start, end in draft.calls)
        assert all(not r["draft_forward"] for r in result.rounds if r["lookup"] == "suffix-repair")


@pytest.mark.parametrize("value", [0, 1, True, 2.5])
def test_invalid_repair_budget(value):
    with pytest.raises(ValueError):
        SuffixRepair(value)
