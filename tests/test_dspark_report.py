import copy

import pytest

from scripts.run_dspark_tree import ARMS,REPEATS
from scripts.write_dspark_tree_report import summarize
from memorydraft.evaluation.asd_paper import BENCHMARKS


def fixture():
    refs=[];rows=[]
    for benchmark in BENCHMARKS:
        identity=benchmark+':test'
        refs.append(dict(request_id=identity,benchmark=benchmark,prompt_sha256='same',snapshot_id='same'))
        for stage,repeats in REPEATS.items():
            n=128 if stage=='prefix' else 256 if stage=='fixed' else 10
            for repeat in range(repeats):
                for arm in ARMS:
                    rows.append(dict(request_id=identity,benchmark=benchmark,prompt_sha256='same',snapshot_id='same',
                        arm=arm,stage=stage,repeat=repeat,token_ids=[1]*n,rounds=[dict(proposed=7,accepted=5,generated_before=1)],
                        asd=None,finish_reason='eos' if stage=='natural' else 'length',latency_seconds=.8 if arm=='DSpark-Tree-ASD' else 1.,
                        quality={'correct':True} if stage=='natural' else None,peak_gpu_bytes=100))
    return rows,refs,[r['request_id'] for r in refs]


def test_dspark_report_full_matrix_and_direct_comparison():
    rows,refs,ids=fixture();result=summarize(rows,refs,ids)
    comparison=next(r for r in result['comparisons'] if (r['benchmark'],r['stage'],r['candidate'],r['baseline'])==
                    ('gsm8k','natural','DSpark-Tree-ASD','DFlash-Tree-ASD'))
    assert result['timed_rows']==len(BENCHMARKS)*len(ARMS)*sum(REPEATS.values())
    assert comparison['mean_time_ratio']==pytest.approx(.8)
    assert comparison['candidate_correct']==comparison['baseline_correct']==1


@pytest.mark.parametrize('change',['missing','input','repeat','work'])
def test_dspark_report_rejects_incomplete_or_mismatched_runs(change):
    rows,refs,ids=fixture()
    if change=='missing':rows.pop()
    elif change=='input':rows[0]['prompt_sha256']='changed'
    elif change=='repeat':rows[0]['token_ids']=[2]*10
    else:next(r for r in rows if r['stage']=='fixed')['token_ids']=[1]
    with pytest.raises(ValueError):summarize(rows,refs,ids)
