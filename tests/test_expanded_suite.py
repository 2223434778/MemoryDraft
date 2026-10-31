import pytest

from memorydraft.evaluation.expanded_suite import load_records,request_for,validate_generated_code,score
from scripts.run_expanded_suite import clipped


def test_new_data_counts_and_hidden_tests_not_in_prompt():
    import json
    from memorydraft.evaluation.expanded_suite import ROOT
    manifest=json.loads((ROOT/'data/research/expanded-suite/manifest.json').read_bytes())
    assert all('\\' not in p for u in manifest['upstream'].values() for p in u['files'])
    pool=load_records()
    assert len(pool)==182
    for r in pool:
        request=request_for(r,r['limit'])
        assert request.max_new_tokens==r['limit']
        if r['benchmark']=='canitedit':
            assert r['tests'] not in '\n'.join(m['content'] for m in request.messages)
            assert 'after' not in r or r['after'] not in '\n'.join(m['content'] for m in request.messages)
        if r['benchmark']=='livecodebench':assert r['input_output'] not in request.messages[0]['content']


def test_middle_clip_keeps_same_head_and_tail():
    class Tokens:
        def encode(self,text,**kwargs):return list(text)
        def decode(self,ids,**kwargs):return ''.join(ids)
    r=clipped(dict(prompt='abcdefghij',clip_input_tokens=6),Tokens())
    assert r['prompt']=='abchij' and r['input_clipped'] and r['original_prompt_tokens']==10


@pytest.mark.parametrize('code',['import os\nos.remove("x")','open("x","w")','import socket','eval("1")'])
def test_code_restrictions(code):
    with pytest.raises(ValueError):validate_generated_code(code)


def test_stdio_and_regular_edit_code_supported():
    validate_generated_code('import sys\nn=int(input())\nprint(n+1)')
    validate_generated_code('from dataclasses import dataclass\n@dataclass\nclass X:\n x:int')
    validate_generated_code('class X:\n def __eq__(self,__value):\n  return isinstance(__value,X)\n def __init__(self):\n  super().__init__()')


def test_aime_integer_equivalence():
    assert score(dict(benchmark='aime25',answer='7'),'The answer is \\boxed{007}.')['secondary_correct']
    assert not score(dict(benchmark='aime25',answer='7'),'\\boxed{8}')['correct']


def test_official_scorer_smoke_when_dependencies_prepared():
    from memorydraft.evaluation.expanded_suite import ROOT
    if not (ROOT/'.deps/langdetect').exists():pytest.skip('server-only optional scoring dependencies')
    record=dict(benchmark='ifeval',ifeval=dict(key=1,prompt='Say hello',instruction_id_list=['keywords:existence'],kwargs=[dict(keywords=['hello'])]))
    assert score(record,'hello')['correct']
    assert score(dict(benchmark='longbench-qasper',subset='qasper',answers=['hello world']),'hello world')['score']==1
    assert score(dict(benchmark='longbench-multi_news',subset='multi_news',answers=['hello world']),'hello world')['score']==pytest.approx(1)
    assert score(dict(benchmark='canitedit',tests='assert f(1)==2'),'def f(x):\n return x+1')['correct']
    import json
    io=json.dumps(dict(inputs=['1\n'],outputs=['2\n'],fn_name=None))
    assert score(dict(benchmark='livecodebench',input_output=io),'print(int(input())+1)')['correct']


def test_expanded_summary_pairs_quality_and_preserves_environment_errors():
    from scripts.report_expanded_suite import summarize,ARMS
    rows=[]
    for arm in ARMS:
        rows.append(dict(request_id='x',arm=arm,benchmark='ifeval',prompt_sha256='p',snapshot_id='s',
            latency_seconds=2 if arm=='DFlash' else 1,token_ids=[1],max_new_tokens=3,finish_reason='eos',
            quality=dict(correct=arm!='ASD',instructions=[arm!='ASD'],loose_correct=True,loose_instructions=[True]),
            rounds=[],ttft_seconds=.1,peak_gpu_bytes=1))
    result=summarize(rows)
    assert result['families']['ifeval']['BufferedKV-ASD']['DFlash']['paired_ratio']==.5
    assert result['families']['ifeval']['ASD']['DFlash']['lost']==['x']
    rows[1]['prompt_sha256']='different'
    with pytest.raises(ValueError,match='inputs'):summarize(rows)
