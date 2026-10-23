"""Summarize prefill/decoding cost and compare fresh same-card HTTP controls."""
import json
from pathlib import Path
from statistics import mean

root=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
base=root/'results/asd-shared-prefix'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
analysis=read(base/'analysis.json')
rows=[json.loads(s) for i in (0,1) for s in (base/f'eval-shared-{i}/rows.jsonl').read_text(encoding='utf-8').splitlines()]
workers=[]
for i in (0,1):
    for v in ('shared','whole'):
        d=base/f'eval-{v}-{i}'
        meta=read(d/'provenance.json')
        warm=[json.loads(s) for s in (d/'warmup.jsonl').read_text(encoding='utf-8').splitlines()]
        loads=[json.loads(s) for s in (d/'http-load.jsonl').read_text(encoding='utf-8').splitlines()]
        workers.append(dict(shard=i,variant=v,model_setup_seconds=meta['model_setup_seconds'],compiler=meta['compiler'],
            first_warmup=dict(seconds=warm[0]['latency_seconds'],ttft_seconds=warm[0]['ttft_seconds'],counters=warm[0]['counters']),
            peak_gpu_bytes=max(r['result']['peak_gpu_bytes'] for r in loads),
            cross_question=[dict(request_id=r['request_id'],hit=r['counters']['prefix_cache_hit'],
                prefix_tokens=r['counters']['prefix_cache_prefix_tokens']) for r in read(d/'cross-question.json')]))
natural=[r for r in rows if r['method']=='ASD' and r['stage']=='natural']
result=dict(ASD_natural_post_first_seconds={s:mean(r['latency_seconds']-r['ttft_seconds'] for r in natural if r['cache_state']==s)
    for s in ('off','cold','hit')},ASD_natural=analysis['summary']['4096/ASD/natural'],
    all_natural_eos=all(r['finish_reason']=='eos' for r in rows if r['stage']=='natural'),quality=analysis['quality'],
    shared_entry_bytes=sorted({r['counters']['prefix_cache_bytes'] for r in rows if r['cache_state']=='hit'}),
    max_spent_abs_delta=analysis['max_spent_abs_delta'],spent_bitwise_differences=analysis['spent_bitwise_differences'],
    formal_peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in rows),http_peak_gpu_bytes=max(w['peak_gpu_bytes'] for w in workers),
    workers=workers,http=analysis['http'])
(base/'interpretation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(natural=result['ASD_natural'],post_first=result['ASD_natural_post_first_seconds'],
    shared_entry_bytes=result['shared_entry_bytes'],all_eos=result['all_natural_eos'],quality=result['quality'],
    max_spent_abs_delta=result['max_spent_abs_delta'],
    http={i:{k:r[k] for k in ('shared_to_whole_wall_ratio','shared_to_off_wall_ratio')} for i,r in result['http'].items()},
    formal_peak_gib=result['formal_peak_gpu_bytes']/2**30,http_peak_gib=result['http_peak_gpu_bytes']/2**30)))
