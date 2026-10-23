"""Record concrete quality checks without changing the frozen scorer."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from scripts.write_asd_tree_report import read_rows

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/asd-joint-tree'
ARMS=('ASD','BudgetTree-ASD','SpineTree-ASD','JointTree-ASD')


def main():
    cells={(r['request_id'],r['arm']):r for r in read_rows(OUT,'rows')
           if r['repeat']==0 and r['stage']=='natural'}
    cases=[]
    notes={
        'math500:359':'Fixed spine reaches 4096 without a final box and enumerates trial S values. '
            'Joint ends at 1291 with boxed 6, original combination at 711 with boxed 6. '
            'SP=24 and S^3-3SP=-8 give S=4,P=6. This checks the final value, not every generated step.',
        'math500:444':'All four frozen scores remain incorrect/needs_review. Explicit aspect-ratio '
            'counterexamples below rule out mere formatting differences.',
        'math500:448':'Correct inverse is 44. Joint emits 5; original combination and fixed spine emit 14. '
            'Both are arithmetically wrong. Linear ASD has the correct final value; its whole reasoning is not certified.',
        'humaneval:134':'All three tree-ASD variants fail the provided tests with identical code SHA. '
            'Linear ASD passes. This is an inherited quality difference relative to linear ASD.'}
    for identity,note in notes.items():
        selected={arm:cells[identity,arm] for arm in ARMS}
        cases.append(dict(request_id=identity,note=note,arms={arm:dict(
            tokens=len(r['token_ids']),calls=len(r['rounds']),finish_reason=r['finish_reason'],
            quality=r['quality'],repeat0_seconds=r['latency_seconds'],spent=r['asd']['spent'],
            text_sha256=hashlib.sha256(r['text'].encode()).hexdigest()) for arm,r in selected.items()}))
    full=cells['math500:359','SpineTree-ASD']
    assert len(full['token_ids'])==4096 and full['finish_reason']=='length' and not full['quality']['correct']
    assert all(cells['math500:359',arm]['quality']['parsed_answer']=='6' for arm in ARMS if arm!='SpineTree-ASD')
    residues={str(x):(17*x)%83 for x in (44,14,5)}
    assert residues=={'44':1,'14':72,'5':2}
    assert cells['math500:448','JointTree-ASD']['quality']['parsed_answer']=='5'
    assert all(cells['math500:448',arm]['quality']['parsed_answer']=='14' for arm in ('BudgetTree-ASD','SpineTree-ASD'))
    counterexamples=[dict(arm='ASD',k=1,value=str(Fraction(1+1,8))),
        dict(arm='BudgetTree-ASD',k=1,value=str(Fraction(1,4))),
        dict(arm='SpineTree-ASD',k=2,value=str(Fraction((4+1)**2,8*4))),
        dict(arm='JointTree-ASD',k=2,value=str(Fraction(4+1,4*2)))]
    assert all(Fraction(c['value'])!=Fraction(1,2) for c in counterexamples)
    hashes={cells['humaneval:134',arm]['quality']['code_sha256'] for arm in ARMS if arm!='ASD'}
    assert len(hashes)==1 and all(not cells['humaneval:134',arm]['quality']['correct'] for arm in ARMS if arm!='ASD')
    result=dict(scoring_unchanged=True,note='Post-hoc arithmetic/output review, no tuning or additional GPU generation.',
        cases=cases,inverse_residues=residues,rectangle_counterexamples=counterexamples)
    (OUT/'quality-review.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(dict(cases=len(cases),scoring_unchanged=True,arithmetic_checks=True)))


if __name__=='__main__':main()
