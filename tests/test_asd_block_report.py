from copy import deepcopy

import pytest

from scripts import write_asd_block_report as report
from scripts.run_asd_block_probe import ARMS, prefix_check, selected_records


def matrix():
    rows=[];previous={};records=selected_records()
    for record in records:
        q=record["id"]
        for m,size in (*ARMS,("AR",16)):
            item=dict(request_id=q,benchmark=record["benchmark"],method=m,experimental_block_size=size,training_block_size=16,
                      token_ids=[10,99],text="answer",quality=dict(correct=True),prompt_sha256=q,snapshot_id=q,
                      finish_reason="eos",asd=None if m=="AR" else dict(spent=0.,budget=0. if m=="DFlash" else 8.),latency_seconds=1.,
                      rounds=[] if m=="AR" else [dict(generated_before=1,proposed=size-1,accepted=1)])
            for stage,repeats,limit in (("natural",1,4096),("prefix",1 if m=="AR" else 2,128)):
                for rep in range(repeats):
                    row=deepcopy(item);row.update(stage=stage,repeat=rep,max_new_tokens=limit)
                    if stage=="prefix":row["prefix_agreement"]=prefix_check(item,row,size)
                    if size==16 and rep==0:previous[q,m,stage]=deepcopy(row)
                    rows.append(row)
    return rows,records,previous


def test_block_matrix_rejects_missing_row_quality_tamper_and_changed_training_length(monkeypatch):
    monkeypatch.setattr(report,"score",lambda r,text:dict(correct=True))
    rows,records,previous=matrix()
    result=report.audit(rows,records,previous,{99})
    assert result["rows"]==228 and len(result["numeric"])==36 and result["native_b16_cases"]==84
    with pytest.raises(ValueError,match="matrix"):report.audit(rows[:-1],records,previous,{99})
    changed=deepcopy(rows);next(r for r in changed if r["experimental_block_size"]==24)["quality"]["correct"]=False
    with pytest.raises(ValueError,match="scoring"):report.audit(changed,records,previous,{99})
    changed=deepcopy(rows);changed[0]["training_block_size"]=32
    with pytest.raises(ValueError,match="Invalid"):report.audit(changed,records,previous,{99})
    changed=deepcopy(rows);next(r for r in changed if r["experimental_block_size"]==24)["asd"]["budget"]=16.
    with pytest.raises(ValueError,match="policy budget"):report.audit(changed,records,previous,{99})


def test_prefix_timing_repeats_merge_without_becoming_more_questions():
    rows,_,_=matrix()
    rows[1]["latency_seconds"]=2.
    entries=report.merge(rows)
    assert len(entries)==156
    assert entries["gsm8k:885","DFlash",16,"prefix"]["seconds"]==1.5
