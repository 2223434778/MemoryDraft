"""Compact interpretation inputs from completed trajectories, not new inference."""
import json
from pathlib import Path
import statistics

from memorydraft.evaluation.adaptation_followup import STUDY, VARIANTS, previous_rows
from memorydraft.evaluation.runtime_migration import read_rows

root=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
base=root/f'results/{STUDY}'
analysis=json.loads((base/'analysis.json').read_text())
rows=[r for i in (0,1) for r in read_rows(base/f'eval-{i}/rows.jsonl')]
once={(r['request_id'],r['stage'],r['variant'],r['policy']):r for r in rows if r['repeat']==0}
points=previous_rows(root,'examples')
positions={f'{s}/{a}':dict(states=len(values),mean=statistics.mean(values),minimum=min(values),maximum=max(values))
           for s in ('train','calibration') for a in ('ordinary','asd-prefix')
           for values in [[r['generated_before'] for r in points if r['split']==s and r['source_policy']==a]]}
samples=[]
for sample in analysis['samples']:
    q=sample['request_id'];arms={}
    for p in ('DFlash','ASD'):
        for v in VARIANTS:
            row=once[q,'natural',v,p]
            item=sample['arms'][f'natural/{v}/{p}']
            arms[f'{v}/{p}']=dict(tokens=len(row['token_ids']),calls=len(row['rounds']),seconds=item['seconds'],
                base_ratio=item['latency_ratio'],position_ratio=item['position_ratio'],ordinary_ratio=item['ordinary_ratio'],
                quality=row['quality'],finish_reason=row['finish_reason'],
                same_tokens_as_base=row['token_ids']==once[q,'natural','base',p]['token_ids'],
                same_tokens_as_position=row['token_ids']==once[q,'natural','ordinary-position',p]['token_ids'],
                accepted=sum(r['accepted'] for r in row['rounds']),proposed=sum(r['proposed'] for r in row['rounds']),
                strict_accepted=sum(r.get('strict_accepted',r['accepted']) for r in row['rounds']),
                common_128=sample['arms'][f'prefix/{v}/{p}']['common_full_128'])
    samples.append(dict(request_id=q,study_split=sample['study_split'],arms=arms))
result=dict(positions=positions,samples=samples)
(base/'interpretation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(positions=positions,gate=analysis['content_increment_gate'],summaries=analysis['scopes'])))
