from copy import deepcopy
import hashlib
import json

import pytest

from memorydraft.evaluation.asd_extension_resume import completed_payloads,identity_difference
from memorydraft.evaluation.asd_paper_extension import aggregate
from test_asd_paper_extension import fixture


def saved_prefix(tmp_path):
    rows,refs,records,frontier=fixture()
    records=[r for r in records if r['benchmark']=='gsm8k']
    payloads=dict(rows=[r for r in rows if r['benchmark']=='gsm8k'],
        references=[r for r in refs if r['benchmark']=='gsm8k'],
        frontier=[r for r in frontier if r['benchmark']=='gsm8k'],rollout=[])
    for name,values in payloads.items():
        (tmp_path/(name+'.jsonl')).write_bytes(b''.join((json.dumps(r)+'\n').encode() for r in values))
    return records,payloads


def test_resume_keeps_exact_complete_group_bytes_and_excludes_partial_request(tmp_path):
    records,payloads=saved_prefix(tmp_path)
    path=tmp_path/'rows.jsonl';original=path.read_bytes()
    path.write_bytes(original+json.dumps(dict(payloads['rows'][0],request_id='not-complete')).encode()+b'\n')
    kept,digests=completed_payloads(tmp_path,records,1)
    assert kept['rows'][0]==original and len(kept['rows'][1])==18
    assert digests['rows']==hashlib.sha256(path.read_bytes()).hexdigest()
    empty,_=completed_payloads(tmp_path,records,0)
    assert all(not data and not rows for data,rows in empty.values())


@pytest.mark.parametrize('change',['missing-row','duplicate-row','missing-ref','missing-control','duplicate-control'])
def test_resume_rejects_incomplete_or_duplicated_completed_prefix(tmp_path,change):
    records,payloads=saved_prefix(tmp_path)
    name='rows' if change.endswith('row') else 'references' if change.endswith('ref') else 'frontier'
    values=payloads[name]
    if change.startswith('missing'):values.pop()
    else:values.append(deepcopy(values[0]))
    (tmp_path/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in values))
    with pytest.raises(ValueError):completed_payloads(tmp_path,records,1)


@pytest.mark.parametrize('completed',[-1,2,True])
def test_resume_rejects_invalid_completed_prefix(tmp_path,completed):
    records,_=saved_prefix(tmp_path)
    with pytest.raises(ValueError):completed_payloads(tmp_path,records,completed)


def test_only_global_frontier_ids_are_required_for_a_shard(tmp_path):
    records,_=saved_prefix(tmp_path)
    (tmp_path/'frontier.jsonl').write_bytes(b'')
    kept,_=completed_payloads(tmp_path,records,1,frontier=set())
    assert kept['frontier']==(b'',[])


def test_identity_difference_includes_first_divergence_lengths_and_binding():
    left=dict(request_id='case',arm='BudgetTree',stage='natural',token_ids=[1,2,3])
    right=dict(token_ids=[1,9])
    value=identity_difference(left,right,'strict-vs-AR')
    assert (value['index'],value['left_id'],value['right_id'])==(1,2,9)
    assert (value['left_tokens'],value['right_tokens'])==(3,2)
    assert value['left_sha256']!=value['right_sha256']
    assert identity_difference(left,left,'same') is None
    assert identity_difference(dict(left,token_ids=[1]),right,'short')['left_id'] is None


def test_default_still_rejects_identity_failure_and_observer_preserves_it():
    rows,refs,records,frontier=fixture()
    for row in rows:
        if row['benchmark']=='gsm8k' and row['arm']=='BudgetTree' and row['stage']=='natural':row['token_ids']=[1,3]
    with pytest.raises(ValueError,match='identity'):aggregate(rows,refs,records,frontier)
    result=aggregate(rows,refs,records,frontier,strict_token_identity=False)
    assert not result['strict_AR_bit_identity_claim']
    assert len(result['strict_identity_differences'])==1
    assert result['strict_identity_differences'][0]['index']==1
    assert next(r for r in rows if r['benchmark']=='gsm8k' and r['arm']=='BudgetTree' and r['stage']=='natural')['token_ids']==[1,3]


@pytest.mark.parametrize('change',['repeat','input','budget','time','fixed-repeat'])
def test_geometry_observation_cannot_bypass_other_invariants(change):
    rows,refs,records,frontier=fixture()
    target=next(r for r in rows if r['stage']=='natural' and r['arm']=='ASD')
    if change=='repeat':target['token_ids']=[999]
    if change=='input':target['prompt_sha256']='changed'
    if change=='budget':target['asd']['spent']=9
    if change=='time':target['latency_seconds']=0
    if change=='fixed-repeat':next(r for r in rows if r['stage']=='fixed' and r.get('role')=='after')['token_ids'][0]=999
    with pytest.raises(ValueError):aggregate(rows,refs,records,frontier,strict_token_identity=False)
