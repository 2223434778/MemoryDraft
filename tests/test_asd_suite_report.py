from copy import deepcopy

import pytest

from scripts.run_asd_suite import prefix_agreement
from scripts.write_asd_suite_report import aggregate, assessed, audit_events, paired, summarize


def test_bootstrap_uses_question_ratios_and_is_reproducible():
    one=paired([.8,1.2,.9,1.1],"fixed")
    assert one==paired([.8,1.2,.9,1.1],"fixed")
    assert one["n"]==4 and one["slower"]==2
    assert one["ci95"][0]<1<one["ci95"][1]
    assert paired([],"fixed") is None
    with pytest.raises(ValueError):
        paired([0],"fixed")


def test_repeats_are_merged_before_uncertainty_and_ar_calls_are_not_zero():
    rows=[dict(request_id="q",benchmark="gsm8k",method="AR",stage="natural",latency_seconds=s,
               token_ids=[1]*10,finish_reason="eos",quality=dict(correct=False),asd=None,rounds=[])
          for s in (1.,2.,3.)]
    item=aggregate(rows)["q","AR","natural"]
    assert item["seconds"]==2 and item["target_forwards"]==10 and not item["quality"]["correct"]


def test_length_cropped_last_block_is_not_used_to_hide_earlier_divergence():
    full=dict(token_ids=list(range(30)))
    prefix=dict(token_ids=list(range(10)),rounds=[dict(generated_before=1,proposed=15),dict(generated_before=8,proposed=2)])
    assert prefix_agreement(full,prefix)["agrees"]
    prefix["token_ids"][9]=99
    assert prefix_agreement(full,prefix)["agrees"]
    prefix["token_ids"][3]=99
    assert not prefix_agreement(full,prefix)["agrees"]


def test_restricted_code_is_unassessed_without_hiding_test_failures():
    assert not assessed(dict(benchmark="mbpp",quality=dict(correct=False,error="ValueError: unsupported import")))
    assert assessed(dict(benchmark="mbpp",quality=dict(correct=False,error="AssertionError")))
    assert assessed(dict(benchmark="math500",quality=dict(correct=False,error="missing boxed final answer")))


def test_cost_decomposition_retains_censored_primary_and_only_labels_common_eos_as_diagnostic():
    from memorydraft.evaluation.asd_suite import BENCHMARKS
    from scripts.run_asd_suite import METHODS
    entries={}
    for b in BENCHMARKS:
        for i in range(2):
            for m in METHODS:
                for stage in ("natural","prefix"):
                    q=f"{b}:{i}"
                    entries[q,m,stage]=dict(request_id=q,benchmark=b,method=m,seconds=4. if m=="AR" else 1.,
                        tokens=128,finish_reason="length" if i else "eos",quality=dict(correct=not i,f1=float(not i)),
                        target_forwards=128 if m=="AR" else 16,spread=0.,complete128=True,regret=0.,relaxed=0.)
    result=summarize(entries)
    c=result["comparisons"]["gsm8k"]["DFlash/AR"]
    assert c["natural"]["n"]==2 and c["common_eos"]["n"]==1
    assert c["natural"]["ratio"]==pytest.approx(c["length_ratio_geomean"]*c["target_density_ratio_geomean"]*c["cost_per_target_forward_ratio_geomean"])
    assert result["methods"]["gsm8k"]["DFlash"]["truncated"]==1


def test_scalar_mechanism_audit_does_not_invent_uncommitted_eos_and_rejects_reset_budget():
    row=dict(method="DFlash",token_ids=[10,11,99],asd=dict(spent=0.,blocks=1,relaxed_tokens=0),
             rounds=[dict(generated_before=1,proposed=2,accepted=2,strict_accepted=2,relaxed_tokens=0,regret=0.)],
             events=[dict(mismatch=[False,False],regrets=[0.,0.],
             values=[2.,1.],accepted=2,strict_accepted=2,relaxed=0,feasible_accepted=2,regret=0.,spent=0.)])
    result=audit_events(row,{99})
    assert result==dict(compatible_blocks=1,uncommitted_eos_not_recorded=0)
    row["token_ids"][-1]=12
    assert audit_events(row,{99})["uncommitted_eos_not_recorded"]==1
    row["events"][0]["spent"]=1.
    with pytest.raises(ValueError,match="ledger"):
        audit_events(row,{99})


def suite_matrix():
    from memorydraft.evaluation.asd_suite import BENCHMARKS
    from scripts.run_asd_suite import METHODS
    records={f"{b}:{i}":dict(id=f"{b}:{i}",benchmark=b) for b in BENCHMARKS for i in range(8)}
    rounds=[dict(generated_before=1,proposed=1,accepted=1,strict_accepted=1,relaxed_tokens=0,regret=0.)]
    events=[dict(mismatch=[False],regrets=[0.],values=[1.],accepted=1,strict_accepted=1,relaxed=0,
                 feasible_accepted=1,regret=0.,spent=0.)]
    rows=[]; mechanism=[]
    for q,r in records.items():
        for m in METHODS:
            item=dict(request_id=q,benchmark=r["benchmark"],method=m,token_ids=[10,99],text="answer",
                      prompt_sha256=q,snapshot_id=q,latency_seconds=1.,quality=dict(correct=True),finish_reason="eos",
                      asd=None if m=="AR" else dict(spent=0.,budget=8.,blocks=1,relaxed_tokens=0),
                      rounds=[] if m=="AR" else deepcopy(rounds))
            for stage,repeats,limit in (("natural",3,4096),("prefix",2,128)):
                for rep in range(repeats):
                    row=deepcopy(item)
                    row.update(stage=stage,repeat=rep,max_new_tokens=limit)
                    if stage=="prefix":row["prefix_agreement"]=prefix_agreement(item,row)
                    rows.append(row)
            if int(q.rsplit(":",1)[1])<2 and m!="AR":
                mechanism.append(dict(deepcopy(item),events=deepcopy(events)))
    return rows,mechanism,records


def test_full_audit_rejects_missing_rows_changed_grading_and_wrong_mechanism_coverage(monkeypatch):
    from scripts import write_asd_suite_report as report
    monkeypatch.setattr(report,"score",lambda r,text:dict(correct=True))
    rows,mechanism,records=suite_matrix()
    numeric,detail=report.audit(rows,mechanism,records,{99})
    assert len(numeric)==96 and all(r["exact"] for r in numeric)
    assert detail["compatible_blocks"]==48
    with pytest.raises(ValueError,match="matrix"):
        report.audit(rows[:-1],mechanism,records,{99})
    changed=deepcopy(rows); changed[0]["quality"]["correct"]=False
    with pytest.raises(ValueError,match="scoring"):
        report.audit(changed,mechanism,records,{99})
    changed=deepcopy(mechanism); changed[0]["request_id"]="gsm8k:2"
    with pytest.raises(ValueError,match="coverage"):
        report.audit(rows,changed,records,{99})
    changed=deepcopy(mechanism); changed[0]["rounds"][0]["accepted"]=0
    with pytest.raises(ValueError,match="dimensions"):
        report.audit(rows,changed,records,{99})
