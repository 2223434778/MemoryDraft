from copy import deepcopy

import pytest

from memorydraft.evaluation.asd_tree import ARMS, REPEATS, aggregate


def fixture():
    ref = dict(request_id="a",benchmark="gsm8k",token_ids=[1,2],text="ok",finish_reason="stop",
               prompt_sha256="fixed",snapshot_id="fixed",quality={"correct":True})
    rows = []
    for stage,repeats in REPEATS.items():
        for arm in ARMS:
            for repeat in range(repeats):
                rows.append(dict(ref,stage=stage,arm=arm,repeat=repeat,rounds=[],asd=None,latency_seconds=1.))
    return rows,[ref]


def test_complete_paired_statistics_and_short_eos_not_forced_to_128():
    rows,refs = fixture()
    result = aggregate(rows,refs)
    assert result["timed_generations"] == 20
    value = result["families"]["gsm8k"]["BudgetTree-ASD"]
    assert value["natural"]["ASD"]["ratio"] == 1
    assert value["natural"]["correct"] == 1
    assert value["prefix"]["ASD"]["n"] == 0


@pytest.mark.parametrize("change", ["coverage","repeat","strict","input","budget","trajectory","time"])
def test_incomplete_or_invalid_evidence_is_rejected(change):
    rows,refs = fixture()
    if change == "coverage":rows.pop()
    if change == "repeat":rows[0]["repeat"] = 1
    if change == "strict":
        for row in rows:
            if row["arm"] == "BudgetTree":row["token_ids"]=[9]
    if change == "input":rows[0]["snapshot_id"]="changed"
    if change == "budget":rows[-1]["asd"]={"spent":9,"relaxed_tokens":1}
    if change == "trajectory":rows[0]["rounds"]=[{"accepted":1}]
    if change == "time":rows[0]["latency_seconds"]=float("nan")
    with pytest.raises(ValueError):aggregate(rows,refs)


def test_quality_and_runtime_benefit_reported_separately():
    rows,refs = fixture()
    for row in rows:
        row["quality"]=deepcopy(row["quality"])
        if row["arm"] == "BudgetTree-ASD":
            row["quality"]["correct"]=False
            row["latency_seconds"] = .9
    result = aggregate(rows,refs)["families"]["gsm8k"]["BudgetTree-ASD"]["natural"]
    assert result["ASD"]["ratio"] == pytest.approx(.9)
    assert result["ASD"]["lost"] == ["a"] and result["correct"] == 0
