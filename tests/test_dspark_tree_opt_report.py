from copy import deepcopy

import pytest

from scripts.run_dspark_tree_opt import ARMS, REPEATS
from scripts.write_dspark_tree_opt_report import summarize


def matrix():
    ref=dict(request_id='alpaca:0',benchmark='alpaca',prompt_sha256='p',snapshot_id='s')
    rows=[]
    for stage,repeats in REPEATS.items():
        for repeat in range(repeats):
            for arm in ARMS:
                rows.append(dict(ref,arm=arm,stage=stage,repeat=repeat,token_ids=[1]*(256 if stage=='fixed' else 128),
                    text='same',finish_reason='length',asd=dict(spent=0),
                    rounds=[dict(regret=0,relaxed_tokens=0,tree_budget=dict(nodes=8,built_nodes=8 if arm.endswith('-Fast') else 32))],
                    selected_tree_sha256='same',quality=None,latency_seconds=1 if arm.endswith('-Fast') else 2,
                    peak_gpu_bytes=1,counters=dict(compile_graphs_before=4,compile_graphs_after=4)))
    return rows,[ref],['alpaca:0']


def test_report_measures_paired_time_and_retains_compute_only_differences():
    result=summarize(*matrix())
    assert result['timed_rows']==16 and result['exact_fast_reference_generations']==4
    assert all(r['mean_time_ratio']==.5 for r in result['comparisons'])


def test_frozen_reference_coverage_and_shared_mt_history():
    from scripts.run_dspark_tree_opt import ROOT, conversation_record, read_reference_files, request_for, selected_records
    refs=read_reference_files(ROOT);count=0
    for original in selected_records(ROOT):
        first=None
        for turn in ((1,2) if original['benchmark']=='mtbench' else (0,)):
            record=conversation_record(original,turn,first) if turn else original
            ref=refs[record['id']]
            assert ref['benchmark']==record['benchmark'] and ref['reference_reused']
            if 'messages' in ref:assert list(request_for(record,4096).messages)==ref['messages']
            if turn==1:first=ref['text']
            count+=1
    assert count==28


@pytest.mark.parametrize('failure',['missing','input','token','tree','budget','fixed','compile'])
def test_report_rejects_invalid_evidence(failure):
    rows,refs,ids=deepcopy(matrix());row=next(r for r in rows if r['arm'].endswith('-Fast') and r['stage']=='fixed')
    if failure=='missing':rows.pop()
    elif failure=='input':row['prompt_sha256']='other'
    elif failure=='token':row['token_ids'][0]=2
    elif failure=='tree':row['selected_tree_sha256']='different'
    elif failure=='budget':row['asd']['spent']=9
    elif failure=='fixed':row['token_ids'].pop()
    elif failure=='compile':row['counters']['compile_graphs_after']=5
    with pytest.raises(ValueError):summarize(rows,refs,ids)
