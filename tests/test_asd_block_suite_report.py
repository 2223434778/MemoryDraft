from copy import deepcopy

import pytest
import torch

from scripts import write_asd_block_suite_report as report
from scripts.run_asd_block_suite import ALL_ARMS, REPEATS, LIMITS, RecordedVerifier, records_and_positions
from scripts.run_asd_block_probe import prefix_check
from scripts.run_asd_budget_probe import POLICIES


def matrix():
    all_records,positions=records_and_positions();records=all_records[:2]
    rows=[];events=[];previous={}
    for record in records:
        for m,size in ALL_ARMS:
            k=size-1;tokens=torch.tensor([99]+[1]*(k-1));logits=torch.zeros(k,100)
            logits.scatter_(1,tokens[:,None],1.)
            verifier=None if m=="AR" else RecordedVerifier(POLICIES[m],record=True)
            if verifier:verifier.decide(tokens,logits,stop_token_ids=[99])
            event=None if verifier is None else verifier.events[0]
            item=dict(request_id=record["id"],benchmark=record["benchmark"],method=m,experimental_block_size=size,training_block_size=16,
                      token_ids=[10,99],text="answer",prompt_sha256=record["id"],snapshot_id=record["id"],finish_reason="eos",
                      asd=None if verifier is None else verifier.summary(),latency_seconds=1.,
                      rounds=[] if verifier is None else [dict(generated_before=1,proposed=k,accepted=1,strict_accepted=k,
                                                             relaxed_tokens=0,regret=0.,draft_forward=True)])
            for stage,reps in REPEATS.items():
                for rep in range(reps):
                    row=deepcopy(item);row.update(stage=stage,repeat=rep,max_new_tokens=LIMITS[stage])
                    if stage=="natural":row["quality"]=dict(correct=True)
                    else:row["prefix_agreement"]=prefix_check(item,row,size)
                    if size==16 and rep==0:previous[record["id"],m,stage]=deepcopy(row)
                    rows.append(row)
            if verifier is not None:
                row=deepcopy(item);row.pop("latency_seconds")
                row.update(events=[deepcopy(event)],quality=dict(correct=True),instrumented_latency_seconds=9.)
                events.append(row)
    return rows,events,records,positions,previous


def test_full_repeats_and_mechanisms_reject_missing_quality_and_identity_tampering(monkeypatch):
    monkeypatch.setattr(report.preflight_report,"score",lambda r,text:dict(correct=True))
    rows,events,records,positions,previous=matrix()
    result=report.audit(rows,events,records,positions,previous,{99})
    assert result["rows"]==130 and result["native_b16_cases"]==50
    assert result["mechanism_rows"]==24 and result["independent_mechanism_blocks"]==24
    assert len(result["numeric"])==12
    with pytest.raises(ValueError,match="matrix"):report.audit(rows[:-1],events,records,positions,previous,{99})
    changed=deepcopy(rows);next(r for r in changed if r["experimental_block_size"]==24 and r["stage"]=="natural" and r["repeat"]==2)["quality"]["correct"]=False
    with pytest.raises(ValueError,match="repeat or quality"):report.audit(changed,events,records,positions,previous,{99})
    changed=deepcopy(events);changed[0]["events"][0]["proposal_token_ids"][0]=98
    with pytest.raises(ValueError,match="identities"):report.audit(rows,changed,records,positions,previous,{99})
    with pytest.raises(ValueError,match="mechanism coverage"):report.audit(rows,events[:-1],records,positions,previous,{99})


def test_exact_candidate_eos_and_persistent_summary_are_audited():
    _,events,_,_,_=matrix()
    assert report.audit_events(events[0],{99})==1
    changed=deepcopy(events[0]);changed["events"][0]["spent"]=1.
    with pytest.raises(ValueError,match="ledger"):report.audit_events(changed,{99})
    changed=deepcopy(events[0]);changed["asd"]["shortened_blocks"]=1
    with pytest.raises(ValueError,match="summary"):report.audit_events(changed,{99})


def test_short_eos_is_not_fixed128_and_repeat_median_preserves_cost_identity():
    rows,_,records,_,_=matrix()
    for r in rows:
        if r["stage"]=="natural" and r["repeat"]==2:r["latency_seconds"]=5.
    entries=report.aggregate(rows)
    assert len(entries)==52 and entries[records[0]["id"],"DFlash",16,"natural"]["seconds"]==1.
    result=report.summarize(entries,records)
    c=result["comparisons"]["gsm8k"]["ASD-Utility24/ASD16"]
    assert c["natural"]["n"]==2 and c["natural"]["ci95"] is None and c["prefix_equal128"] is None
    assert c["natural"]["ratio"]==pytest.approx(c["length_ratio"]*c["target_density_ratio"]*c["cost_per_forward_ratio"])
    assert c["preflight_overlap"]["natural"]["n"]==1 and c["other_development"]["natural"]["n"]==1
