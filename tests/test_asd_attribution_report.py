from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
import shutil

import pytest

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from scripts.write_asd_attribution_report import ROOT, analyze, audit_decisions, audit_frozen, decompose, quality_review, render, same_states, totals
from test_asd import scores


def test_factorization_separates_shorter_output_from_round_density():
    rows=[dict(method="ASD",latency_seconds=10,token_ids=[1]*100,counters=dict(rounds=10)),
          dict(method="ASD-Run",latency_seconds=4,token_ids=[1]*80,counters=dict(rounds=5))]
    t=totals(rows,["ASD","ASD-Run"])
    d=decompose(t["ASD"],t["ASD-Run"])
    assert d==dict(length_ratio=.8,round_density_ratio=.625,round_cost_ratio=.8,latency_ratio=.4)


def test_common_state_check_does_not_roll_out_alternate_prefixes():
    config=ASDConfig(budget=8)
    policy=ASDVerifier(config,record=True)
    policy.decide(*scores([.6,-1,10,-1,-1,-1,-1]))
    policy.decide(*scores([-1,-1]))
    for e in policy.events:e["generated_before"]=1
    row=dict(method="ASD",events=policy.events)
    before=deepcopy(row)
    values=same_states([row],{"ASD":config,"ASD-Run":ASDConfig(budget=8,value="exact-run"),
                              "ASD-Conservative":ASDConfig(budget=8,gate=.0625)})
    assert values["ASD"]["states"]==1
    assert values["ASD"]["methods"]["ASD"]["local_extra_proposals"]==2
    assert values["ASD"]["methods"]["ASD-Run"]["regret"]==0
    assert row==before


def test_stricter_gate_audit_rejects_event_that_only_passes_original_asd():
    policy=ASDVerifier(ASDConfig(budget=8),record=True)
    policy.decide(*scores([.5,-1,-1,-1]))
    policy.events[0]["generated_before"]=1
    row=dict(events=policy.events,asd=policy.summary(),token_ids=[1]*6,
             counters=dict(rounds=1),finish_reason="length")
    audit_decisions(row,policy.config)
    with pytest.raises(ValueError,match="Scalar"):
        audit_decisions(row,ASDConfig(budget=8,gate=.0625))


def test_unaccepted_proposal_eos_clips_utility_before_generation_finishes():
    import torch.nn.functional as functional
    policy=ASDVerifier(ASDConfig(budget=8,selection='utility'),record=True)
    tokens,logits=scores([-1,-1,.8,-1,-1,10])
    logits=functional.pad(logits,(0,1),value=-100.)
    logits[2,2]=logits[2,1]
    logits[2,1]=-100.
    tokens[2]=2
    policy.decide(tokens,logits,stop_token_ids=[2])
    event=policy.events[0]
    event['generated_before']=1
    assert event['accepted']==2 and event['feasible_accepted']==3
    row=dict(events=policy.events,asd=policy.summary(),token_ids=[1]*16,
             counters=dict(rounds=1),finish_reason='length')
    assert audit_decisions(row,policy.config)==[dict(event=0,unaccepted_proposal_eos=3)]


def test_frozen_evidence_coverage_and_two_legacy_checkout_exceptions(tmp_path,monkeypatch):
    from scripts import write_asd_attribution_report as report
    monkeypatch.setattr(report,"ROOT",tmp_path)
    name="results/research-audit/model_preflight.json"
    p=tmp_path/name
    p.parent.mkdir(parents=True)
    p.write_bytes(b"{}\r\n")
    sha=hashlib.sha256(b"{}\n").hexdigest()
    files={"x.py":"x"}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    manifest=dict(files={name:sha,"memorydraft/x.py":"x","scripts/run_asd_budget_probe.py":"runner"})
    info=dict(original_sha256={name:sha},source=dict(files=files,sha256=digest),runner_sha256="runner")
    audit_frozen(info,manifest)
    (p.parent/"future.json").write_bytes(b"new publication")
    audit_frozen(info,manifest)
    info["original_sha256"]={}
    with pytest.raises(ValueError,match="coverage"):audit_frozen(info,manifest)
    info["original_sha256"]={name:sha}
    p.write_bytes(b"changed\r\n")
    with pytest.raises(ValueError,match="evidence changed"):audit_frozen(info,manifest)


def test_separate_quality_audit_preserves_original_and_binds_raw_output(tmp_path):
    original=dict(correct=False,parsed_answer="5",expected_answer="x=5")
    requests={"math500:23":{"ASD":dict(quality=original)}}
    methods={"math500":{"ASD":dict(correct=0)}}
    review=dict(mechanism_sha256={},equivalent_answers=[dict(request_id="math500:23",method="ASD",
                parsed_answer="5",expected_answer="x=5",rationale="only valid root")])
    p=tmp_path/"quality-audit.json"
    p.write_text(json.dumps(review),encoding="utf-8")
    assert quality_review(tmp_path,requests,methods)["correct"]["math500"]["ASD"]==1
    assert original["correct"] is False
    review["equivalent_answers"][0]["parsed_answer"]="6"
    p.write_text(json.dumps(review),encoding="utf-8")
    with pytest.raises(ValueError,match="equivalence"):quality_review(tmp_path,requests,methods)


@pytest.fixture
def evidence(tmp_path):
    source=ROOT/"results/asd-attribution"
    if not source.exists():pytest.skip("GPU attribution evidence is not yet published")
    target=tmp_path/"evidence"
    shutil.copytree(source,target)
    return target


def test_real_report_rebuilds_without_rewriting_evidence(evidence):
    path=evidence/"math-a/mechanism/rows.jsonl.gz"
    sha=file_hash(path)
    a,infos=analyze(evidence)
    assert len(a["requests"])==24
    assert render(a,infos)==(ROOT/"docs/ASD_ATTRIBUTION_RESULTS.md").read_text(encoding="utf-8")
    assert file_hash(path)==sha


@pytest.mark.parametrize("kind",["duplicate","budget","trajectory","prefix","decision","policy"])
def test_corrupt_published_evidence_rejected(evidence,kind):
    if kind=="policy":
        path=evidence/"math-a/provenance.json"
        data=json.loads(path.read_text(encoding="utf-8"))
        data["policies"]["ASD-Conservative"]["gate"]=.25
        path.write_text(json.dumps(data),encoding="utf-8")
    else:
        stage="prefix" if kind=="prefix" else "natural" if kind=="trajectory" else "mechanism"
        path=evidence/f"math-a/{stage}/rows.jsonl"
        rows=read_rows(path)
        if kind=="duplicate":rows.append(deepcopy(rows[0]))
        elif kind=="budget":rows[0]["asd"]["spent"]=9
        elif kind=="trajectory":rows[0]["token_ids"][40]=999999
        elif kind=="prefix":rows[0]["full_prefix"]["common_prefix_tokens"]=0
        else:rows[0]["events"][0]["values"][0]=999
        path.with_suffix(".jsonl.gz").write_bytes(gzip.compress("".join(json.dumps(r)+"\n" for r in rows).encode(),mtime=0))
    with pytest.raises(ValueError):analyze(evidence)
