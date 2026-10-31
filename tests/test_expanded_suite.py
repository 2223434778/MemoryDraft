import pytest

from memorydraft.evaluation.expanded_suite import load_records,request_for,validate_generated_code,score
from scripts.run_expanded_suite import clipped


def test_new_data_counts_and_hidden_tests_not_in_prompt():
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


def test_aime_integer_equivalence():
    assert score(dict(benchmark='aime25',answer='7'),'The answer is \\boxed{007}.')['secondary_correct']
    assert not score(dict(benchmark='aime25',answer='7'),'\\boxed{8}')['correct']
