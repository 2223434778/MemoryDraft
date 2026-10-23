"""Collect all changed objective outcomes and unresolved math answers for review."""
import hashlib
import json
from pathlib import Path

from memorydraft.evaluation.asd_paper import ARMS,load_records,quality_ok
from scripts.write_asd_paper_report import read_rows

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/asd-paper'


def main():
    pool={r['id']:r for r in load_records(ROOT)}
    rows=[r for r in read_rows(OUT,'rows') if r['stage']=='natural' and r['repeat']==0]
    cells={(r['request_id'],r['arm']):r for r in rows}
    selected={r['request_id'] for r in rows if r['benchmark']=='math500' and r['quality'].get('needs_review')}
    for row in rows:
        if row['quality']['correct'] is None:continue
        for baseline in ('ASD','BudgetTree-ASD'):
            other=cells[row['request_id'],baseline]
            if quality_ok(row['quality'])!=quality_ok(other['quality']):selected.add(row['request_id'])
    cases=[]
    for identity in sorted(selected):
        task=pool[identity]
        case=dict(request_id=identity,benchmark=task['benchmark'],arms={})
        for key in ('question','problem','prompt','text','signature','answer','entry_point'):
            if key in task:case[key]=task[key]
        for arm in ARMS:
            row=cells[identity,arm]
            case['arms'][arm]=dict(quality=row['quality'],tokens=len(row['token_ids']),calls=len(row['rounds']),
                finish_reason=row['finish_reason'],spent=row['asd']['spent'] if row['asd'] else 0,
                text_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                final_excerpt=row['text'][-1800:] if task['benchmark'] in ('gsm8k','math500') else row['text'])
        cases.append(case)
    (OUT/'quality-review-queue.json').write_bytes((json.dumps(dict(scoring_unchanged=True,
        scope='All changed objective correct outcomes against ASD or original combination, plus unresolved MATH answers. '
        'This queue does not automatically change grades or certify reasoning.',cases=cases),indent=2)+'\n').encode())
    print(json.dumps(dict(review_cases=len(cases),by_benchmark={b:sum(c['benchmark']==b for c in cases)
        for b in sorted({c['benchmark'] for c in cases})})))


if __name__=='__main__':main()
