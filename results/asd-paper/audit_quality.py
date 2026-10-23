"""Supplement frozen grades with explicit counterexamples and format diagnostics."""
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

from memorydraft.evaluation.asd_paper import ARMS,load_records
from memorydraft.evaluation.asd_suite import code_score
from scripts.write_asd_paper_report import read_rows

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/asd-paper'


def main():
    paths=('analysis.json','report.md')
    frozen={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in paths}
    rows=[r for r in read_rows(OUT,'rows') if r['stage']=='natural' and r['repeat']==0]
    cells={(r['request_id'],r['arm']):r for r in rows}
    pool={r['id']:r for r in load_records(ROOT)}
    queue=json.loads((OUT/'quality-review-queue.json').read_bytes())
    ids={r['request_id'] for r in queue['cases']}
    assert len(ids)==24

    math_notes={
        'math500:154':'Squared lengths: reference 452, strict 520, ASD 40, combination 45; all unequal. Joint repeats the unfolding discussion until length=4096, without a final answer.',
        'math500:180':'Two-digit common divisors are 10,25,50. Combination alone returns 3; all others return 2.',
        'math500:264':'Reference mixed number is 9/5. Combination 5/3 and other completed answers 5 are unequal. ASD is unfinished at 4096.',
        'math500:33':'125/9*(3/5)^7 = 243/625, not strict 146/375. ASD and both combinations give the correct fraction.',
        'math500:355':'Put t=c/(a+b). Using ab<=(a+b)^2/4, E>=8/t+40+96t+64t^2. E-100>=(4t-1)^2*(4t+8)/t>=0; equality at a=b=2c. Joint is unfinished at 4096.',
        'math500:358':'Equation becomes (x-6)^2/40+y^2/10=1. Unequal positive denominators imply ellipse, not Joint circle.',
        'math500:421':'cos(A+B)=-3x gives 12x^3+14x^2-1=0. It is primitive and has no rational root, so minimal cubic coefficient L1 is 27, not Joint 22. ASD repeats an unfinished minimal-polynomial discussion.',
        'math500:425':'Centroid identity gives sum distances^2=(a^2+b^2+c^2)/3+3*GH^2/4=3R^2. On a 3-4-5 triangle, strict expression is 425/12, Joint 475/24, reference 75/4, so neither is an equivalent form. ASD repeats an unfinished vector identity.',
        'math500:485':'Enumerating the 24 open Hamiltonian paths with the six labelled edge lengths gives minimum 13, not the combination and Joint answer 19.',
    }
    assert set(math_notes)=={r['request_id'] for r in queue['cases'] if r['benchmark']=='math500'}
    divisors=[x for x in range(10,100) if 100%x==150%x==0]
    assert divisors==[10,25,50]
    assert Fraction(125,9)*Fraction(3,5)**7==Fraction(243,625)!=Fraction(146,375)
    assert Fraction(9,5)!=Fraction(5,3)!=5
    roots={sign*Fraction(1,d) for d in (1,2,3,4,6,12) for sign in (-1,1)}
    assert all(12*x**3+14*x**2-1!=0 for x in roots)
    strict=Fraction(50,3)+3*Fraction(25,4)
    joint=Fraction(9,2)*Fraction(25,4)-Fraction(50,6)
    assert strict==Fraction(425,12) and joint==Fraction(475,24) and strict!=joint!=Fraction(75,4)
    edges={frozenset(e):v for e,v in (('AB',6),('AC',6),('AD',5),('BC',5),('BD',3),('CD',4))}
    shortest=min(sum(edges[frozenset((a,b))] for a,b in zip(p,p[1:])) for p in itertools.permutations('ABCD'))
    assert shortest==13
    math_review=[]
    for identity,note in math_notes.items():
        math_review.append(dict(request_id=identity,review=note,additional_equivalent_answers=0,
            output_sha256={arm:hashlib.sha256(cells[identity,arm]['text'].encode()).hexdigest() for arm in ARMS}))

    code_notes={
        'humaneval:117':('assert select_words("Uncle sam",3)==["Uncle"]',
            'The combinations omit word.lower(), counting uppercase U as a consonant.'),
        'humaneval:83':('assert starts_one_ends(2)==18',
            'Combination enumerates and passes the source tests. Other formulas give 17/19/99 for n=2. Test passing does not establish acceptable asymptotic complexity.'),
        'humaneval:93':('assert encode("test")=="TGST"',
            'ASD handles both cases; other completions fail uppercase vowel replacement after swapcase.'),
        'mbpp:506':('assert permutation_coefficient(10,2)==90',
            'Strict and combination compute a binomial coefficient (45), instead of permutation coefficient (90). ASD and Joint use the passing recurrence.'),
        'mbpp:214':(None,
            'Both formulas implement radian-to-degree conversion. Original exact floating-point assertions distinguish multiplication/division association; this is not evidence of a mathematical mistake.'),
    }
    assert set(code_notes)=={r['request_id'] for r in queue['cases'] if r['benchmark'] in ('humaneval','mbpp')}
    code_review=[]
    for identity,(test,note) in code_notes.items():
        grades={}
        for arm in ARMS:
            row=cells[identity,arm]
            result=code_score(pool[identity],row['text'])
            assert result['correct']==row['quality']['correct'],(identity,arm)
            item=dict(original_tests_correct=result['correct'],original_error=result['error'])
            if test:
                task=dict(pool[identity],test_list=[test],benchmark='mbpp')
                item['counterexample_passed']=code_score(task,row['text'])['correct']
            grades[arm]=item
        code_review.append(dict(request_id=identity,review=note,counterexample=test,arms=grades))
    floating=[]
    for angle,expected in ((90,5156.620156177409),(60,3437.746770784939),(120,6875.493541569878)):
        left=angle*180/math.pi;grouped=angle*(180/math.pi)
        assert math.isclose(left,expected,rel_tol=1e-12) and grouped==expected
        floating.append(dict(radians=angle,expected=expected,left=left,grouped=grouped,
                             absolute_error=abs(left-expected),ulps=abs(left-expected)/math.ulp(expected)))
    assert any(v['left']!=v['expected'] for v in floating)

    choice=lambda text:re.match(r'\s*([ABCD])(?:[.\s]|$)',text)
    mmlu=[r for r in rows if r['benchmark']=='mmlu']
    mmlu_stats={}
    for arm in ARMS:
        selected=[r for r in mmlu if r['arm']==arm]
        mmlu_stats[arm]=dict(requests=len(selected),frozen_exact=sum(r['quality']['correct'] for r in selected),
            leading_choice_correct=sum(bool((m:=choice(r['text'])) and m[1]==r['quality']['expected_answer']) for r in selected),
            no_leading_choice=sum(not choice(r['text']) for r in selected))
    changed=[]
    for identity in {r['request_id'] for r in mmlu}:
        labels={arm:(choice(cells[identity,arm]['text'])[1] if choice(cells[identity,arm]['text']) else None) for arm in ARMS}
        if len(set(labels.values()))!=1:changed.append(dict(request_id=identity,labels=labels))
    assert changed==[] and all(v['leading_choice_correct']==38 for v in mmlu_stats.values())
    mmlu_review=[dict(request_id=r['request_id'],review='Changed frozen grade is due to extra explanation after the same correct leading option, not a changed leading choice.')
                 for r in queue['cases'] if r['benchmark']=='mmlu']
    assert len(mmlu_review)==10

    raw=gzip.decompress((OUT/'local-judge.jsonl.gz').read_bytes()).decode().splitlines()
    judge=[json.loads(x) for x in raw]
    pairs={}
    for r in judge:pairs.setdefault((r['request_id'],r['arm']),[]).append(r)
    nonidentical=[p for p in pairs.values() if all(r['status']=='judged' for r in p)]
    consistent=sum(p[0]['candidate_score']==p[1]['candidate_score'] for p in nonidentical)
    assert len(nonidentical)==88 and consistent==0
    judge_audit=dict(rows=len(judge),statuses=dict(Counter(r['status'] for r in judge)),
        labels=dict(Counter(r['label'] for r in judge if r['status']=='judged')),
        nonidentical_complete_pairs=len(nonidentical),order_consistent_pairs=consistent,
        conclusion='The first-token local judge fails the order-swap control. Do not use its approximately 50% preferences as quality preservation evidence.')
    outlier={arm:{k:cells['mbpp:163',arm][k] for k in ('finish_reason','quality','latency_seconds')}
             |dict(tokens=len(cells['mbpp:163',arm]['token_ids'])) for arm in ARMS}
    result=dict(frozen_primary_scores_unchanged=True,reviewed_cases=24,additional_math_equivalences=0,
        math=math_review,code=code_review,mbpp214_float_diagnostic=floating,mmlu=mmlu_review,
        mmlu_leading_choice_diagnostic=mmlu_stats,changed_leading_choices=changed,
        local_judge_audit=judge_audit,mbpp163_truncation=outlier,
        note='Post-hoc diagnostics only. No primary grade, timing sample, prompt, parameter or selection was changed.')
    assert frozen=={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in paths}
    (OUT/'quality-audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(dict(reviewed=24,additional_math_equivalences=0,mmlu_leading_choice=mmlu_stats,
        judge=judge_audit,frozen_primary_scores_unchanged=True)))


if __name__=='__main__':main()
