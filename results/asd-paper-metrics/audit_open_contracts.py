"""Post-hoc objective checks; never replaces the frozen main quality metrics."""
import ast
import hashlib
import json
from pathlib import Path
import random
import re

from memorydraft.evaluation.asd_paper_metrics import message_sha,open_units
from memorydraft.evaluation.asd_suite import code_score

ROOT=Path.cwd()


def main():
    out=ROOT/'results/asd-paper-metrics'
    cases=[];blind={}
    for unit in open_units(ROOT):
        grouped={}
        for arm,messages in unit['answers'].items():
            v=grouped.setdefault(message_sha(messages),dict(messages=messages,arms=[]));v['arms'].append(arm)
        ordered=list(grouped.items());random.Random('open-contract-audit:'+unit['request_id']).shuffle(ordered)
        for index,(digest,value) in enumerate(ordered):
            label=chr(65+index);blind[unit['request_id']+'/'+label]=dict(sha256=digest,arms=value['arms'])
            texts=[m['content'] for m in value['messages'] if m['role']=='assistant']
            checks=None;note=None
            if unit['request_id']=='mtbench:109':
                checks=dict(first_shadow_west=True,second_facing_south=label!='A')
                note='Reviewed anonymous A-D before reading the mapping. A ends both turns by incorrectly inferring north from right=west. The other completions infer south.'
                if label=='A':assert 'facing north' in texts[-1].replace('**','').lower()
                else:assert 'facing south' in texts[-1].replace('**','').lower()
            elif unit['request_id']=='mtbench:114':
                checks=dict(first_probability_35_over_36=True,second_probability_one=True)
                assert '\\frac{35}{36}' in texts[0] and '\\boxed{1}' in texts[1]
                note='Both final answers match enumeration: 35 of 36 dice outcomes have sum>=3, all 36 have an even sum or sum>=3.'
            elif unit['request_id']=='mtbench:130':
                checks=dict(second_turn_expected_linear_time=label=='C')
                if label=='C':assert 'set_arr2 = set(arr2)' in texts[1]
                else:assert '.sort()' in texts[1]
                note='Anonymous C supplies an expected O(n+m) hash-set approach. A/B/D retain sorting and overclaim best complexity. This criterion does not certify duplicate semantics or every space claim.'
            elif unit['request_id']=='mtbench:138':
                decoded=[]
                for text in texts:
                    try:decoded.append(json.loads(text))
                    except json.JSONDecodeError:decoded.append(None)
                first,second=decoded
                checks=dict(first_complete_json=isinstance(first,dict),
                    second_complete_json=isinstance(second,dict),
                    second_phone_names_uppercase=bool(second) and all(k==k.upper() for k in second))
                note='Ratings themselves are subjective. Check only complete JSON and the explicit all-capital phone-name constraint; no repairs or extraction of a partial object.'
            elif unit['request_id']=='alpaca:35230':
                checks=dict(passive_voice=texts[0].strip().rstrip('.').casefold()=='a house was built by my father')
                note='The complete text gives the requested passive form.'
            elif unit['request_id']=='alpaca:10294':
                numbers=re.findall(r'^\s*(\d+)\.\s',texts[0],re.M)
                checks=dict(exactly_five_numbered_questions=numbers==['1','2','3','4','5'])
                note='Structural count only; not an audit of medical claims. These outputs are questions.'
            elif unit['request_id']=='alpaca:12828':
                count=len([s for s in texts[0].splitlines() if s.strip()])
                checks=dict(fourteen_nonempty_poem_lines=count==14)
                note='Line count only, not a certification of iambic pentameter, originality or the full Shakespearean rhyme scheme.'
            elif unit['request_id']=='alpaca:46981':
                text=texts[0].lower();mammals=text.split('mammals',1)[-1].split('birds',1)[0]
                birds=text.split('birds',1)[-1]
                checks=dict(mammals_dog_tiger_whale_and_bird_eagle=all(a in mammals for a in ('dog','tiger','whale')) and 'eagle' in birds)
                note='Checks the explicitly requested four-animal classification, not other possible taxonomies.'
            elif unit['request_id']=='alpaca:42198':
                blocks=re.findall(r'```python\s*\n(.*?)```',texts[0],re.S)
                explicit=next(b for b in blocks if any(isinstance(n,ast.For) for n in ast.walk(ast.parse(b))))
                variable='my_list' if 'my_list' in explicit else 'numbers'
                task=dict(benchmark='mbpp',test_list=[f'assert {variable} == [2,4,6,8,10]'])
                result=code_score(task,'my_list=[1,2,3,4,5]\nnumbers=[1,2,3,4,5]\n'+explicit)
                checks=dict(first_explicit_loop_doubles_list=result['correct'])
                note='Executes the first explicitly presented for loop using the existing restricted, timed scorer. Optional alternatives and performance are not certified.'
            if checks is None:continue
            for arm in value['arms']:
                cases.append(dict(request_id=unit['request_id'],arm=arm,anonymous_label=label,
                    messages_sha256=digest,checks=checks,note=note))
    assert len(cases)==45
    result=dict(scope='Post-hoc nine task units with explicit checkable properties; not official open-task accuracy or a replacement for reward scoring.',
        reviewer='Codex; anonymous unique-response packets for direction and complexity review, deterministic checks elsewhere',
        unchanged_frozen_quality=True,cases=cases)
    (out/'open-contract-audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(dict(task_units=9,cells=len(cases),failed_checks=sum(not v for c in cases for v in c['checks'].values()))))


if __name__=='__main__':main()
