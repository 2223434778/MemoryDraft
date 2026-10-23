from copy import deepcopy
from pathlib import Path

import pytest

from memorydraft.evaluation.asd_comparison import ARMS, BASELINES, BENCHMARKS, REPEATS, aggregate, arms_for, records, select_candidates

ROOT = Path(__file__).resolve().parents[1]


def fixture_rows():
    return [dict(phase="screen",request_id=b+":fixture",benchmark=b,stage=s,arm=a,repeat=i,
        token_ids=list(range(128)),rounds=[{}]*8,asd=None,finish_reason="eos",prompt_sha256=b,snapshot_id="empty",
        same_input=True,strict_token_match=True,latency_seconds=.9 if a == "base/ASD-Utility" else 1.,
        quality=dict(correct=True)) for b in BENCHMARKS for s,n in REPEATS["screen"].items() for a in ARMS for i in range(n)]


def test_frozen_reserve_split_and_balanced_shards():
    a,b = records(ROOT,"screen"), records(ROOT,"confirm")
    assert len(a) == 24 and len(b) == 72
    assert not {r["id"] for r in a} & {r["id"] for r in b}
    assert len(ARMS) == len(set(ARMS)) == 14
    for phase,n in (("screen",4),("confirm",12)):
        assert all(sum(r["benchmark"] == b for r in records(ROOT,phase)) == n for b in BENCHMARKS)
    with pytest.raises(ValueError):
        records(ROOT,"development")


def test_selection_uses_screen_and_keeps_policy_and_adapter_axes():
    result = aggregate(fixture_rows(),"screen")
    selected = select_candidates(result)
    assert all(a[0] == "base/ASD-Utility" and not a[1].startswith("base/") for a in selected.values())
    assert all(arms_for(b,"confirm",selected)[:2] == BASELINES for b in BENCHMARKS)
    assert all(result["families"][b]["base/ASD-Utility"]["natural"]["base/ASD"]["ratio"] == pytest.approx(.9) for b in BENCHMARKS)


@pytest.mark.parametrize("bad",["drop","duplicate","input","strict","repeat","nan","budget"])
def test_incomplete_or_unsafe_matrix_cannot_select(bad):
    rows = fixture_rows()
    if bad == "drop":rows.pop()
    elif bad == "duplicate":rows.append(deepcopy(rows[0]))
    elif bad == "input":rows[0]["prompt_sha256"] = "different"
    elif bad == "strict":rows[0]["strict_token_match"] = False
    elif bad == "repeat":rows[0]["token_ids"] = [9]
    elif bad == "nan":rows[0]["latency_seconds"] = float("nan")
    else:rows[0]["asd"] = dict(spent=8.1)
    with pytest.raises(ValueError):
        aggregate(rows,"screen")


def test_quality_losses_are_preserved_and_precede_speed_for_selection():
    rows = fixture_rows()
    for row in rows:
        if row["arm"] == "base/ASD-Utility":row["quality"]["correct"] = False
    result = aggregate(rows,"screen")
    assert result["families"]["gsm8k"]["base/ASD-Utility"]["natural"]["base/ASD"]["lost"] == 1
    assert all(a[0] != "base/ASD-Utility" for a in select_candidates(result).values())


def test_short_eos_is_retained_but_excluded_from_equal128_contrast():
    rows = fixture_rows()
    for row in rows:
        if row["stage"] == "prefix" and row["arm"] == "base/ASD-Utility":row["token_ids"] = [1,2]
    result = aggregate(rows,"screen")
    item = result["families"]["gsm8k"]["base/ASD-Utility"]
    assert item["natural"]["requests"] == 1
    assert item["prefix"]["base/ASD"]["n"] == 0
    assert item["prefix"]["base/ASD"]["ratio"] is None


def test_confirmation_rejects_missing_or_changed_selection():
    with pytest.raises(ValueError):arms_for("gsm8k","confirm")
    bad = {b:["base/ASD","base/tree32"] for b in BENCHMARKS}
    with pytest.raises(ValueError):arms_for("gsm8k","confirm",bad)


def test_first_token_eos_does_not_invent_a_verification_round():
    rows = fixture_rows()
    for row in rows:
        row["token_ids"],row["rounds"] = [1],[]
    result = aggregate(rows,"screen")
    assert result["families"]["squad"]["base/ASD"]["natural"]["base/ASD"]["call_density_ratio"] is None


def test_execute_routes_frozen_policies_through_real_tiny_backend(engine):
    # This fixture is imported below from the existing runtime tests.
    from scripts.run_asd_comparison import execute
    record = dict(benchmark="gsm8k",id="fixture",question="3+4?",answer="#### 7")
    for arm in ("base/DFlash","base/ASD","base/ASD-Run","base/ASD-Utility","base/Confidence-Utility","base/tree32"):
        row = execute(engine,record,arm,8)
        assert row["token_ids"] and row["arm"] == arm
        if arm not in ("base/DFlash","base/tree32"):assert row["asd"]["budget"] == 8


from test_runtime import engine  # noqa: E402,F401
