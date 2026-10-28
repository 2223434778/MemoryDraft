from copy import deepcopy

import pytest

from memorydraft.evaluation.asd_paper import BENCHMARKS
from memorydraft.evaluation.branch_drift import ARMS, PREFIX_ARMS, summarize


def matrix():
    refs, rows = [], []
    for benchmark in BENCHMARKS:
        ref = dict(request_id=benchmark+':example',benchmark=benchmark,token_ids=list(range(128)),
                   prompt_sha256='prompt',snapshot_id='snapshot')
        refs.append(ref)
        for stage, arms in (('natural',ARMS),('prefix',PREFIX_ARMS)):
            for arm in arms:
                rows.append(dict(ref,stage=stage,arm=arm,max_new_tokens=4096 if stage=='natural' else 128,
                    latency_seconds=.8 if arm=='DriftTree-ASD' else 1.,finish_reason='stop',asd=None,rounds=[],
                    quality=dict(correct=True) if stage=='natural' else None))
    return rows, refs


def test_summary_uses_direct_paired_latency_and_valid_common_prefix():
    rows, refs = matrix()
    actual = summarize(rows,refs)['families']['gsm8k']['DriftTree-ASD']
    assert actual['natural']['BudgetTree-ASD']['ratio'] == pytest.approx(.8)
    assert actual['prefix']['BudgetTree-ASD']['n'] == 1
    assert actual['natural']['identical_ar'] == 1


def test_missing_rows_and_changed_prompts_fail_instead_of_silently_filtering():
    rows, refs = matrix()
    with pytest.raises(ValueError,match='Incomplete'):
        summarize(rows[:-1],refs)
    rows[0]['prompt_sha256'] = 'changed'
    with pytest.raises(ValueError,match='Input'):
        summarize(rows,refs)


def test_short_eos_excluded_only_from_common128_and_quality_loss_kept():
    rows, refs = matrix()
    for row in rows:
        if row['benchmark']=='gsm8k' and row['arm']=='DriftTree-ASD':
            row['token_ids'] = [0, 1, 10]
            if row['stage']=='natural':
                row['quality'] = dict(correct=False)
    actual = summarize(rows,refs)['families']['gsm8k']['DriftTree-ASD']
    assert actual['prefix']['BudgetTree-ASD']['n'] == 0
    assert actual['natural']['identical_ar'] == 0
    assert actual['natural']['ar_prefix_fraction'] == pytest.approx(2/3)
    assert actual['natural']['BudgetTree-ASD']['lost'] == ['gsm8k:example']
