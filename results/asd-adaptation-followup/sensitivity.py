"""Posthoc quality/influence checks; never replace the frozen full-sample results."""
import json
from pathlib import Path
import statistics

from memorydraft.evaluation.adaptation_followup import STUDY, VARIANTS
from memorydraft.evaluation.runtime_migration import read_rows

root=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
base=root/f'results/{STUDY}'
a=json.loads((base/'analysis.json').read_text())
samples=[s for s in a['samples'] if s['study_split']=='independent']
correct=[s for s in samples if all(s['arms'][f'natural/{v}/ASD']['quality']['correct'] for v in VARIANTS)]
refs={r['request_id']:r for i in (0,1) for r in read_rows(base/f'eval-{i}/references.jsonl')}
subsets={}
for name,subset in (('all_16',samples),('shared_correct_only',correct),('without_450',[s for s in samples if s['request_id']!='gsm8k:450'])):
    summary={}
    for stage in ('natural','prefix'):
        for variant in VARIANTS:
            items=[s['arms'][f'{stage}/{variant}/ASD'] for s in subset]
            summary[f'{stage}/{variant}/ASD']=dict(requests=len(items),
                latency_ratio=statistics.mean(v['latency_ratio'] for v in items),
                position_ratio=statistics.mean(v['position_ratio'] for v in items),
                output_tokens=sum(v['tokens'] for v in items),target_calls=sum(v['rounds'] for v in items))
    subsets[name]=dict(ids=[s['request_id'] for s in subset],summary=summary)
influence={}
for stage,variant in (('natural','ordinary-position'),('prefix','ordinary-position'),('natural','asd-prefix'),('prefix','asd-prefix')):
    key=f'{stage}/{variant}/ASD'
    means={dropped['request_id']:statistics.mean(s['arms'][key]['latency_ratio'] for s in samples if s is not dropped) for dropped in samples}
    influence[key]=dict(minimum=min(means.values()),maximum=max(means.values()),leave_one_out=means)
errors=[]
for s in samples:
    q=s['request_id'];strict=refs[q]['quality'];approx=s['arms']['natural/base/ASD']['quality']
    if strict['correct'] != approx['correct']:
        errors.append(dict(request_id=q,strict=strict,base_asd=approx,
            variants={v:s['arms'][f'natural/{v}/ASD']['quality'] for v in VARIANTS}))
result=dict(posthoc=True,purpose='Sensitivity to already-wrong output paths; no removal from primary results or changes to gates',
            shared_correct_requests=len(correct),subsets=subsets,leave_one_out=influence,asd_quality_changes_vs_ar=errors)
(base/'sensitivity.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result))
