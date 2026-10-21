from copy import deepcopy
from pathlib import Path

import pytest
import torch

from memorydraft.backends.asd import ASDVerifier
from memorydraft.evaluation.runtime_migration import read_rows
from scripts import run_asd_block_suite as runner
from scripts.run_asd_budget_probe import POLICIES


def test_development_shards_keep_every_family_and_first_stage_gpu_assignment():
    records,positions=runner.records_and_positions()
    assert len(records)==48 and {r["split"] for r in records}=={"development"}
    for shard in (0,1):
        selected=[r for r in records if positions[r["id"]]%2==shard]
        assert len(selected)==24
        assert all(sum(r["benchmark"]==b for r in selected)==4 for b in runner.BENCHMARKS)
        assert sum(positions[r["id"]]<2 for r in selected)==6


@pytest.mark.parametrize("method",list(POLICIES))
@pytest.mark.parametrize("k",[15,23,31])
def test_complete_proposals_are_only_recorded_in_instrumented_pass_without_changing_decision(method,k):
    generator=torch.Generator().manual_seed(120+k)
    logits=torch.randn(k,7,generator=generator)
    tokens=torch.randint(0,7,(k,),generator=generator)
    plain=ASDVerifier(POLICIES[method],record=True)
    observed=runner.RecordedVerifier(POLICIES[method],record=True)
    assert observed.decide(tokens,logits,stop_token_ids=[6])==plain.decide(tokens,logits,stop_token_ids=[6])
    assert observed.summary()==plain.summary()
    event=deepcopy(observed.events[0])
    assert event.pop("proposal_token_ids")==tokens.tolist()
    assert event.pop("target_top_token_ids")==logits.argmax(-1).tolist()
    assert event==plain.events[0]
    silent=runner.RecordedVerifier(POLICIES[method],record=False)
    silent.decide(tokens,logits)
    assert silent.events==[]


def fake_execute(record,method,limit):
    row=dict(request_id=record["id"],benchmark=record["benchmark"],method=method,token_ids=[10,99],text="7",
             prompt_sha256="p",snapshot_id="s",finish_reason="eos",max_new_tokens=limit,
             latency_seconds=1.,rounds=[] if method=="AR" else [dict(generated_before=1,proposed=1)],
             asd=None if method=="AR" else dict(spent=0,budget=POLICIES[method].budget))
    return row


def test_question_runs_full_65_row_matrix_with_quality_and_separate_mechanism(monkeypatch,tmp_path):
    record=dict(id="q",benchmark="gsm8k")
    monkeypatch.setattr(runner,"execute",lambda engine,r,m,limit:fake_execute(r,m,limit))
    monkeypatch.setattr(runner,"run",lambda engine,r,m,size,limit:fake_execute(r,m,limit))
    monkeypatch.setattr(runner,"score",lambda r,text:dict(correct=True))
    previous={}
    for m in ("AR",*POLICIES):
        for stage,limit in runner.LIMITS.items():
            row=fake_execute(record,m,limit)
            if stage=="natural":row["quality"]=dict(correct=True)
            previous["q",m,stage]=row

    def mechanism(engine,r,m,size):
        row=fake_execute(r,m,4096)
        row["instrumented_latency_seconds"]=row.pop("latency_seconds")
        row["events"]=[]
        return row

    monkeypatch.setattr(runner,"record_mechanism",mechanism)
    result=runner.run_question(None,record,0,previous,tmp_path)
    assert result==dict(main=65,mechanism=12)
    rows=read_rows(tmp_path/"rows.jsonl")
    assert len({(r["method"],r["experimental_block_size"],r["stage"],r["repeat"]) for r in rows})==65
    assert all(r["quality"]["correct"] for r in rows if r["stage"]=="natural")
    assert len(read_rows(tmp_path/"mechanism.jsonl"))==12
    other=tmp_path/"other";other.mkdir()
    assert runner.run_question(None,record,2,previous,other)==dict(main=65,mechanism=0)
    assert not (other/"mechanism.jsonl").exists()


def test_repeat_or_native_change_stops_instead_of_mixing_trajectories(monkeypatch,tmp_path):
    record=dict(id="q",benchmark="gsm8k")
    previous={("q",m,s):fake_execute(record,m,limit) for m in ("AR",*POLICIES) for s,limit in runner.LIMITS.items()}
    for (q,m,s),r in previous.items():
        if s=="natural":r["quality"]=dict(correct=True)
    monkeypatch.setattr(runner,"score",lambda r,text:dict(correct=True))
    monkeypatch.setattr(runner,"execute",lambda engine,r,m,limit:fake_execute(r,m,limit))
    calls={}

    def changed(engine,r,m,size,limit):
        row=fake_execute(r,m,limit);key=m,size,limit;calls[key]=calls.get(key,0)+1
        if size==24 and calls[key]>1:row["token_ids"]=[11,99]
        return row

    monkeypatch.setattr(runner,"run",changed)
    with pytest.raises(ValueError,match="repeat changed"):
        runner.run_question(None,record,2,previous,tmp_path)
