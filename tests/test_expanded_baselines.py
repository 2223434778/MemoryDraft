from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from memorydraft.evaluation.expanded_suite import request_for
from memorydraft.memory.store import Snapshot
from scripts import run_expanded_baselines as runner
from scripts.report_expanded_baselines import analyze


def test_published_costs_and_canonical_mask_frozen(tmp_path):
    root=Path(__file__).resolve().parents[1]
    for folder in ('expanded-suite','external-trees'):
        source=root/'results'/folder;dest=tmp_path/'results'/folder;dest.mkdir(parents=True)
        for p in source.iterdir():
            if p.name in ('publication.json','rows.jsonl.gz','calibration.json','quality-v3-canonical-preflight.json','bastion-calibration.json','bastion-profile.json'):
                shutil.copyfile(p,dest/p.name)
    prior,tree,mask,hashes=runner.frozen_inputs(tmp_path)
    assert len(prior)==546 and tree.max_nodes==32 and len(hashes)==5
    assert mask=={'canitedit:78_llm_inference'}
    (tmp_path/'results/external-trees/bastion-profile.json').write_text('{}')
    with pytest.raises(ValueError,match='frozen reference'):runner.frozen_inputs(tmp_path)


def test_new_request_factory_reaches_official_adapter(monkeypatch):
    calls=[]
    def external(*args,**kwargs):
        calls.append(kwargs['request_factory'](args[1],args[3]));return 'official'
    monkeypatch.setattr(runner,'external_execute',external)
    r=dict(id='x',benchmark='ifeval',prompt='Follow the actual instruction',limit=16)
    assert runner.execute(None,r,'BASTION',None,None,None,{})=='official'
    assert calls==[request_for(r,16)]


def test_prepared_inputs_reject_prompt_or_limit_changes(monkeypatch):
    record=dict(id='x',benchmark='ifeval',prompt='frozen',limit=16)
    monkeypatch.setattr(runner,'load_records',lambda _: [record])
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return [1,2]
    engine=SimpleNamespace(tokenizer=Tokenizer(),config=SimpleNamespace(template_options={}))
    req=request_for(record,16)
    prior=[dict(request_id='x',arm='DFlash',prompt_sha256=hashlib.sha256(json.dumps([1,2]).encode()).hexdigest(),
        snapshot_id=Snapshot(req.user_id,req.order,()).snapshot_id,max_new_tokens=16)]
    assert runner.prepared_records(engine,prior)==[record]
    record['limit']=17
    with pytest.raises(ValueError,match='frozen input'):runner.prepared_records(engine,prior)


def test_five_arm_analysis_retains_strict_and_prior_differences():
    rows=[]
    for a in runner.ARMS:
        rows.append(dict(request_id='x',arm=a,benchmark='ifeval',prompt_sha256='p',snapshot_id='s',latency_seconds=2 if a=='AR' else 1,
            token_ids=[2] if a=='BASTION' else [1],text=a if a=='BASTION' else 'same',max_new_tokens=4,finish_reason='eos',
            quality=dict(correct=a!='BASTION'),rounds=[],ttft_seconds=None,peak_gpu_bytes=1))
    prior=[dict(rows[1],token_ids=[3])]
    a=analyze(rows,prior)
    assert a['strict_identity_differences']==[dict(id='x',arm='BASTION')]
    assert a['previous_output_differences']==[dict(id='x',arm='DFlash')]
    assert a['families']['ifeval']['BufferedKV-ASD']['AR']['paired_ratio']==.5
    assert a['families']['ifeval']['BASTION']['same_ar']==0
