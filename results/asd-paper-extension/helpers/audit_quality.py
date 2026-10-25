"""Small semantic witnesses for changed code results; primary grades stay frozen."""
import ast
from copy import deepcopy
import hashlib,json
from pathlib import Path

from memorydraft.evaluation.asd_paper_extension import ARMS,load_records,normalize_fixed,read_rows
from memorydraft.evaluation.asd_bench import extract_code

ROOT=Path.cwd();OUT=ROOT/'results/asd-paper-extension'
records={r['id']:r for r in load_records(ROOT)};raw=read_rows(OUT,'rows');normalized,_=normalize_fixed(raw)
index={(r['request_id'],r['stage'],r['arm']):r for r in normalized if r['repeat']==0}
same=[]
for (identity,stage,arm),row in index.items():
    if arm!='BudgetTree':continue
    baseline=index[identity,stage,'DFlash']
    assert all(row[k]==baseline[k] for k in ('token_ids','text','finish_reason','prompt_sha256','snapshot_id'))
    same.append((identity,stage))
assert len(same)==530*3
examples=[('humaneval:91',['Is the sky blue?'],0,'The word I must be separated from Is.'),
    ('humaneval:99',['-15.5'],-16,'round(-15.5) is already -16; subtracting one gives -17.'),
    ('humaneval:103',[560,851],'0b1011000010','Mean705.5 rounds to706; integer floor returns705.'),
    ('mbpp:190',[1,1,4,4],4,'Strictly interior integer points: (3-1)*(3-1)=4, not3*3=9.'),
    ('mbpp:333',[[['a',10],['b',5]]],[['b',5],['a',10]],'Benchmark requires a returned sorted list; in-place sorting alone returns None.'),
    ('mbpp:74',[['red','green','greenn'],['a','b']],False,'Different input lengths must fail; zip alone silently truncates.')]
code=[]
for identity,args,expected,explanation in examples:
    outcomes={}
    for arm in ('ASD','BudgetTree-ASD'):
        row=index[identity,'natural',arm];text=extract_code(row['text'])
        assert hashlib.sha256(text.encode()).hexdigest()==row['quality']['code_sha256']
        parsed=ast.parse(text);assert all(isinstance(n,ast.FunctionDef) for n in parsed.body)
        namespace={};exec(compile(parsed,'<published-generated-function>','exec'),namespace)
        entry=records[identity].get('entry_point') or next(n.name for n in parsed.body if isinstance(n,ast.FunctionDef))
        actual_args=deepcopy(args);actual=namespace[entry](*actual_args);passed=actual==expected
        assert passed==row['quality']['correct'],(identity,arm)
        outcomes[arm]=dict(actual=actual,expected=expected,arguments_after_call=actual_args,
            witness_passed=passed,primary_quality=row['quality'],text_sha256=hashlib.sha256(row['text'].encode()).hexdigest())
    code.append(dict(request_id=identity,input=args,explanation=explanation,arms=outcomes))
description=json.loads((OUT/'descriptive.json').read_bytes());mmlu=description['mmlu_cases']
assert all(len(set(r['first_choices'].values()))==1 for r in mmlu)
mmlu_changes=[]
for case in mmlu:
    if case['primary_correct']['ASD']==case['primary_correct']['BudgetTree-ASD']:continue
    identity=case['request_id'];mmlu_changes.append(dict(case,answers={a:index[identity,'natural',a]['text'] for a in ARMS}))
math_cases=[]
for identity in ('math500:156','math500:224','math500:328','math500:213','math500:323'):
    math_cases.append(dict(request_id=identity,arms={a:dict(quality=index[identity,'natural',a]['quality'],
        output_tokens=len(index[identity,'natural',a]['token_ids']),finish_reason=index[identity,'natural',a]['finish_reason'],
        text_tail=index[identity,'natural',a]['text'][-700:],text_sha256=hashlib.sha256(index[identity,'natural',a]['text'].encode()).hexdigest())
        for a in ('ASD','BudgetTree-ASD')}))
result=dict(scope='Post-hoc quality evidence; no overwritten primary tests or grades. Witnesses cover six changed ASD/combined code cases, not every possible input.',
    strict_tree_equals_dflash=dict(matched_requests=530,stages=3,comparisons=len(same),tokens_text_stop_input_equal=True,
        AR_identity_scope='Both differ from tokenwise AR on math500:323 at602; actual-forward diagnostics are separately published.'),
    changed_code_witnesses=code,mmlu_leading_choices_equal_per_request=True,mmlu_primary_changes=mmlu_changes,
    math_gained_lost_examples=math_cases,all_changed_frozen_score_cases=description['quality_changed_cases'],
    frozen_score_definition='Primary for GSM/code/MMLU; MATH additionally retains the predeclared finite scalar-equivalence supplement. No after-the-fact grade changes.',
    primary_scores_unchanged=True)
path=OUT/'quality-audit.json';assert not path.exists();path.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(strict_tree_comparisons=len(same),code_cases=len(code),code_witness_evaluations=len(code)*2,
    mmlu_requests=len(mmlu),mmlu_primary_changes=len(mmlu_changes),all_changed_cases=len(description['quality_changed_cases']))))
