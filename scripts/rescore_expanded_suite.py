"""Uniform offline rescoring after fixing code-check false positives."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from memorydraft.evaluation.expanded_suite import load_records,score

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('generated',type=Path)
    parser.add_argument('--revision',type=int,choices=(2,3),default=2)
    args=parser.parse_args()
    generated=args.generated.resolve()
    info=json.loads((generated/'provenance.json').read_bytes())
    if info['status']!='completed':raise ValueError('Generation must finish before rescoring')
    out=generated/f'quality-v{args.revision}';out.mkdir(exist_ok=False)
    pool={r['id']:r for r in load_records(ROOT)}
    canonical=[]
    for record in pool.values():
        if record['benchmark']=='canitedit':
            canonical.append(dict(id=record['id'],canonical=score(record,record['canonical']),before=score(record,record['before'])))
    (out/'canonical-preflight.json').write_text(json.dumps(canonical,indent=2)+'\n')
    unavailable={r['id'] for r in canonical if r['canonical'].get('correct') is not True}
    cache={};count=0
    with (out/'scores.jsonl').open('x') as f:
        for line in (generated/'rows.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['benchmark']!='canitedit':continue
            key=row['request_id'],row['text']
            if key not in cache:
                cache[key]=dict(correct=None,evaluation_error='canonical tests unavailable') if key[0] in unavailable else score(pool[key[0]],key[1])
            f.write(json.dumps(dict(request_id=key[0],arm=row['arm'],quality=cache[key]))+'\n')
            count+=1
    manifest=dict(status='completed',revision=args.revision,rescored=count,canonical_scored=len(canonical)-len(unavailable),unavailable=sorted(unavailable),
        source={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['memorydraft/evaluation/expanded_suite.py','scripts/score_expanded_task.py','scripts/rescore_expanded_suite.py']},
        generation_rows_sha256=hashlib.sha256((generated/'rows.jsonl').read_bytes()).hexdigest(),
        rule=('Before examining generated code edits: allow ordinary object methods/identifiers and scientific imports; CPU only,8GiB AS. Same tests for all methods; no generation/timing changes.' if args.revision==2 else
              'Canonical-only environment repair: isolated trusted dependency path and file-based source inspection. Same hidden tests, CPU limits, all methods; generation/timing untouched. Earlier scoring retained.'))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':main()
