from copy import deepcopy
from pathlib import Path
import hashlib
import json

import pytest

from memorydraft.evaluation.asd_paper_metrics import ROLES,acceptance,inputs,message_sha,open_units,validate_triplet
from scripts.download_asd_reward import verify
from scripts.run_asd_paper_metrics import fixed_ar
from scripts.run_asd_reward import CONTROLS,control_messages,pair_score

ROOT=Path(__file__).resolve().parents[1]


def test_fixed_ar_disables_inherited_eos_and_restores_stops_even_on_error(engine):
    from memorydraft.evaluation.asd_paper import request_for
    task=dict(id='gsm8k:fixture',benchmark='gsm8k',question='w1',answer='#### 1')
    expected=engine.generate(request_for(task,8),'AR').token_ids
    engine.stops=[expected[0]]
    assert len(engine.generate(request_for(task,8),'AR').token_ids)==1
    assert fixed_ar(engine,task,8)['token_ids']==expected and engine.stops==[expected[0]]
    engine._closed=True
    with pytest.raises(RuntimeError):fixed_ar(engine,task,8)
    assert engine.stops==[expected[0]]


def test_same_requests_and_mt_histories_reuse_original_references():
    left,refs=inputs(ROOT,0);right,_=inputs(ROOT,1)
    assert len(left)==125 and len(right)==124
    assert not {r['id'] for r in left}&{r['id'] for r in right}
    for record in left+right:
        if record['benchmark']=='mtbench' and record['turn']==2:
            assert record['messages'][1]['content']==refs[record['conversation_id']+':turn1']['text']


def triplet():
    old=dict(request_id='x',benchmark='gsm8k',prompt_sha256='prompt',snapshot_id='snapshot',
        token_ids=list(range(256)),latency_seconds=2.)
    rows=[dict(deepcopy(old),role=role,stop_policy='ignore-eos') for role in ROLES]
    rows[1]['latency_seconds']=8.
    return rows,old


@pytest.mark.parametrize('change',['input','tokens','time','missing','order'])
def test_invalid_ar_bridge_cannot_enter_speed_table(change):
    rows,old=triplet()
    if change=='input':rows[1]['prompt_sha256']='different'
    elif change=='tokens':rows[0]['token_ids'][-1]=99
    elif change=='time':rows[1]['latency_seconds']=float('nan')
    elif change=='missing':rows.pop()
    else:rows.reverse()
    with pytest.raises(ValueError):validate_triplet(rows,old)


def test_bridge_and_final_block_metrics_do_not_count_unemitted_tokens():
    rows,old=triplet();result=validate_triplet(rows,old)
    assert result['ar_seconds']/result['strict_seconds']==4
    row=dict(token_ids=list(range(5)),rounds=[dict(proposed=7,accepted=6,generated_before=0)])
    value=acceptance(row)
    assert value['selected_draft']==6 and value['emitted_draft']==5 and value['verification_nodes']==8


def test_open_reward_scores_realized_dialogues_without_method_name_prompts():
    units=open_units(ROOT)
    assert len(units)==24 and sum(u['benchmark']=='mtbench' for u in units)==8
    for unit in units:
        for messages in unit['answers'].values():
            assert messages[0]['role']=='user' and messages[-1]['role']=='assistant'
            assert all(m['role']!='system' for m in messages)
            assert len(messages)==(4 if unit['benchmark']=='mtbench' else 2)
            assert message_sha(messages)==message_sha(deepcopy(messages))


def test_reward_controls_and_equal_scores_have_no_ab_position_semantics():
    assert len(CONTROLS)==8 and all(a!=b for _,_,a,b in CONTROLS)
    assert pair_score(3,2)==1 and pair_score(2,3)==0
    assert pair_score(3,3)==.5 and pair_score(3,2,identical=True)==.5
    assert pair_score(None,3) is None
    assert [m['role'] for m in control_messages('question','answer')]==['user','assistant']


def test_checkpoint_verification_accepts_git_blobs_and_lfs_hashes(tmp_path):
    path=tmp_path/'model';path.write_bytes(b'checkpoint')
    info=dict(bytes=10,sha256=hashlib.sha256(b'checkpoint').hexdigest())
    assert verify(path,info)
    assert verify(path,dict(bytes=10,git_blob=hashlib.sha1(b'blob 10\0checkpoint').hexdigest()))
    path.write_bytes(b'Checkpoint');assert not verify(path,info)


def test_complete_ar_table_preserves_direct_measurements_and_labels_estimates(monkeypatch):
    from memorydraft.evaluation import asd_paper_metrics as module
    from memorydraft.evaluation.asd_paper import ARMS,BENCHMARKS
    old=[];new=[]
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            row=dict(request_id=benchmark,benchmark=benchmark,prompt_sha256='prompt',snapshot_id='snapshot',
                token_ids=list(range(256)),latency_seconds=1. if arm=='BudgetTree-ASD' else 2.,
                arm=arm,stage='fixed',repeat=0,
                rounds=[dict(proposed=3,accepted=3,generated_before=i*4) for i in range(64)])
            old.append(row)
            if arm=='DFlash':
                new.extend(dict(deepcopy(row),role=role,stop_policy='ignore-eos',latency_seconds=8. if role=='AR' else 2.) for role in ROLES)
    monkeypatch.setattr(module,'read_rows',lambda root,name:iter(old))
    result=module.aggregate(None,new)
    assert result['requests']==7 and result['timed_generations']==21
    for family in result['families'].values():
        assert family['ar_tps']==32 and family['measured_current_dflash_speedup']==4
        assert family['arms']['BudgetTree-ASD']['split_session_speedup_vs_ar']==8
        assert family['arms']['BudgetTree-ASD']['anchor_adjusted_estimated_speedup_vs_ar']==8
    assert all(v['tau_selected']==v['tau_emitted']==3 for v in result['acceptance'])
    with pytest.raises(ValueError):module.aggregate(None,new[:-1])


from test_runtime import engine  # noqa: E402,F401
