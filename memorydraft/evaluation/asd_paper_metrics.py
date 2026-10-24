"""Supplement the frozen paper comparison without changing its data or policy."""
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_paper import ARMS,BENCHMARKS,conversation_record,load_records

STUDY='asd-paper-metrics'
ROLES=('DFlash-before','AR','DFlash-after')


def read_rows(root,name):
    for shard in (0,1):
        path=Path(root)/f'results/asd-paper/part-{shard}/{name}.jsonl.gz'
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for line in stream:yield json.loads(line)


def inputs(root,shard):
    refs={r['request_id']:r for r in read_rows(root,'references')}
    records=[]
    for benchmark in BENCHMARKS:
        selected=[r for r in load_records(root) if r['benchmark']==benchmark]
        for i,record in enumerate(selected):
            if i%2!=shard:continue
            if benchmark=='mtbench':
                records += [conversation_record(record,1),conversation_record(record,2,refs[record['id']+':turn1']['text'])]
            else:records.append(record)
    return records,refs


def validate_triplet(rows,old):
    if len(rows)!=3 or tuple(r['role'] for r in rows)!=ROLES:raise ValueError('Incomplete AR triplet')
    for r in rows:
        if r['request_id']!=old['request_id'] or r['benchmark']!=old['benchmark']:raise ValueError('Wrong task')
        if (r['prompt_sha256'],r['snapshot_id'])!=(old['prompt_sha256'],old['snapshot_id']):raise ValueError('Changed Target input')
        if len(r['token_ids'])!=256 or r['stop_policy']!='ignore-eos':raise ValueError('Wrong fixed work')
        if r['token_ids']!=old['token_ids']:raise ValueError('Fixed strict AR/token mismatch')
        if not math.isfinite(r['latency_seconds']) or r['latency_seconds']<=0:raise ValueError('Invalid wall time')
    return dict(request_id=old['request_id'],benchmark=old['benchmark'],
        ar_seconds=rows[1]['latency_seconds'],strict_seconds=statistics.mean(r['latency_seconds'] for r in (rows[0],rows[2])),
        old_strict_seconds=old['latency_seconds'],
        within_session_drift=rows[2]['latency_seconds']/rows[0]['latency_seconds'],
        cross_session_drift=statistics.mean(r['latency_seconds'] for r in (rows[0],rows[2]))/old['latency_seconds'])


def acceptance(row):
    blocks=row['rounds'];n=len(blocks)
    if not n:raise ValueError('Missing proposal blocks')
    tokens=len(row['token_ids'])
    # Logs retain an untrimmed bonus at the output limit. Count emitted draft tokens
    # by the actual generated offset so the last partial block cannot inflate tau.
    emitted_draft=sum(min(b['accepted'],max(0,tokens-b['generated_before'])) for b in blocks)
    nodes=sum(b.get('tree_nodes',b['proposed']+1) for b in blocks)
    return dict(rounds=n,proposed_draft=sum(b['proposed'] for b in blocks),
        selected_draft=sum(b['accepted'] for b in blocks),emitted_draft=emitted_draft,
        output_tokens=tokens,verification_nodes=nodes)


def open_units(root):
    pool={r['id']:r for r in load_records(root)};units={}
    for row in read_rows(root,'rows'):
        if row['benchmark']!='alpaca' or row['stage']!='natural' or row['repeat']!=0:continue
        task=pool[row['request_id']]
        question=task['instruction']+('\n\nInput:\n'+task['input'] if task['input'] else '')
        unit=units.setdefault(task['id'],dict(request_id=task['id'],benchmark='alpaca',answers={}))
        unit['answers'][row['arm']]=[dict(role='user',content=question),dict(role='assistant',content=row['text'])]
    for row in read_rows(root,'rollout'):
        task=pool[row['conversation_id']]
        unit=units.setdefault(task['id'],dict(request_id=task['id'],benchmark='mtbench',answers={}))
        unit['answers'][row['arm']]=[dict(role='user',content=task['turns'][0]),dict(role='assistant',content=row['first_text']),
            dict(role='user',content=task['turns'][1]),dict(role='assistant',content=row['text'])]
    if len(units)!=24 or any(set(u['answers'])!=set(ARMS) for u in units.values()):raise ValueError('Incomplete open coverage')
    return [units[k] for k in sorted(units)]


def message_sha(messages):
    return hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def aggregate(root,new_rows):
    groups=defaultdict(list)
    for row in new_rows:groups[row['request_id']].append(row)
    fixed={(r['request_id'],r['arm']):r for r in read_rows(root,'rows') if r['stage']=='fixed'}
    expected={identity for identity,arm in fixed if arm=='DFlash'}
    if set(groups)!=expected:raise ValueError('Incomplete fixed AR coverage')
    triplets={i:validate_triplet(groups[i],fixed[i,'DFlash']) for i in sorted(expected)}
    mechanisms=defaultdict(lambda:defaultdict(int))
    for r in read_rows(root,'rows'):
        if r['repeat']!=0:continue
        values=acceptance(r);key=r['benchmark'],r['stage'],r['arm']
        for name,value in values.items():mechanisms[key][name]+=value
        mechanisms[key]['requests']+=1
    families={}
    for benchmark in BENCHMARKS:
        chosen=[t for t in triplets.values() if t['benchmark']==benchmark]
        ar=sum(t['ar_seconds'] for t in chosen);strict=sum(t['strict_seconds'] for t in chosen)
        arms={}
        for arm in ARMS:
            old_seconds=sum(fixed[t['request_id'],arm]['latency_seconds'] for t in chosen)
            adjusted=sum(fixed[t['request_id'],arm]['latency_seconds']*t['cross_session_drift'] for t in chosen)
            arms[arm]=dict(old_fixed_tps=256*len(chosen)/old_seconds,
                split_session_speedup_vs_ar=ar/old_seconds,
                anchor_adjusted_estimated_speedup_vs_ar=ar/adjusted)
        families[benchmark]=dict(requests=len(chosen),ar_tps=256*len(chosen)/ar,
            measured_current_dflash_tps=256*len(chosen)/strict,measured_current_dflash_speedup=ar/strict,
            within_drift_over_3pct=sum(not .97<=t['within_session_drift']<=1.03 for t in chosen),
            cross_drift_over_3pct=sum(not .97<=t['cross_session_drift']<=1.03 for t in chosen),arms=arms)
    return dict(timed_generations=len(new_rows),requests=len(triplets),families=families,triplets=list(triplets.values()),
        acceptance=[dict(benchmark=b,stage=s,arm=a,**v,
            tau_selected=v['selected_draft']/v['rounds'],tau_emitted=v['emitted_draft']/v['rounds'],
            output_per_target_call=v['output_tokens']/v['rounds'],nodes_per_output_token=v['verification_nodes']/v['output_tokens'])
            for (b,s,a),v in sorted(mechanisms.items())],
        note='Old five-arm fixed timings are preserved. Split-session speedups and anchor-adjusted estimates are explicitly labelled, not new paired candidate timings. No drift-based task filtering.')
