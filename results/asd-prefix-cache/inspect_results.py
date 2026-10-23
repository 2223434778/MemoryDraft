"""Derive mechanism and capacity observations from the published raw rows."""
import json
from pathlib import Path
from statistics import mean

root=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
base=root/'results/asd-prefix-cache'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
analysis=read(base/'analysis.json')
rows=[json.loads(s) for i in (0,1) for s in (base/f'eval-{i}/rows.jsonl').read_text(encoding='utf-8').splitlines()]
output={}
for level in (512,2048,4096):
    subset=[r for r in rows if r['level']==level and r['method']=='ASD' and r['stage']=='natural']
    output[str(level)]=dict(
        mean_post_first_seconds={s:mean(r['latency_seconds']-r['ttft_seconds'] for r in subset if r['cache_state']==s)
                                 for s in ('off','cold','hit')},
        min_max_prefix_tokens=[min(r['counters']['prefix_cache_prefix_tokens'] for r in subset if r['cache_state']=='hit'),
                               max(r['counters']['prefix_cache_prefix_tokens'] for r in subset if r['cache_state']=='hit')],
        cache_entry_bytes=sorted({r['counters']['prefix_cache_bytes'] for r in subset if r['cache_state']=='hit'}),
        outputs_unchanged=all(not r['comparison_to_previous']['changed'] for r in subset))
workers=[]
for i in (0,1):
    d=base/f'eval-{i}'
    meta=read(d/'provenance.json')
    warm=[json.loads(s) for s in (d/'warmup.jsonl').read_text(encoding='utf-8').splitlines()]
    http=read(d/'http-summary.json')
    load=[json.loads(s) for s in (d/'http-load.jsonl').read_text(encoding='utf-8').splitlines()]
    cache=[r for r in load if r['state']=='cache']
    workers.append(dict(shard=i,model_setup_seconds=meta['model_setup_seconds'],compiler=meta['compiler'],
        cold_first_warmup=warm[0],http_wall_ratio=http['cache']['wall_seconds']/http['off']['wall_seconds'],
        http_hitrate=http['cache']['hits']/http['cache']['requests'],
        http_cache_evictions=sum(r['result']['counters']['prefix_cache_evictions'] for r in cache),
        http_cache_bytes=sorted({r['result']['counters']['prefix_cache_bytes'] for r in cache}),
        http_task_sequence=[r['task_id'] for r in cache],
        cross_question=[dict(request_id=r['request_id'],hit=r['counters']['prefix_cache_hit'],
                             prefix_tokens=r['counters']['prefix_cache_prefix_tokens']) for r in read(d/'cross-question.json')]))
result=dict(levels=output,workers=workers,
    all_natural_eos=all(r['finish_reason']=='eos' for r in rows if r['stage']=='natural'),
    maximum_peak_gib=max(r['peak_gpu_bytes'] for r in rows)/2**30,
    full_128={k:dict(n=v['common_full_128'],hit_ratio=v['common_full_128_hit_ratio']) for k,v in analysis['summary'].items() if k.endswith('/prefix')})
registry=read(root/'data/workloads/asd-prefix-cache/public-prefixes.json')
a,b=[p['token_ids'] for p in registry['prefixes'] if 4096 in p['levels']]
common=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
sizes=output['4096']['cache_entry_bytes']
per_token=max(sizes)/max(len(a),len(b))
result['public_prefix_overlap']=dict(prefix_tokens=[len(a),len(b)],common_tokens=common,
    whole_entries_bytes=sum(sizes),common_part_bytes=int(common*per_token),
    interpretation='Whole-prefix entries duplicate the shared beginning; not a measured block-cache implementation')
(base/'interpretation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(levels=result['levels'],http=[{k:w[k] for k in ('shard','http_wall_ratio','http_hitrate','http_cache_evictions')} for w in workers],peak_gib=result['maximum_peak_gib'],all_eos=result['all_natural_eos'])))
