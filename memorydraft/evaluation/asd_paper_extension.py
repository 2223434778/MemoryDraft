"""Frozen cross-task confirmation and matched-control frontier, without retuning."""
from collections import defaultdict
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import statistics

from memorydraft.backends.asd import ASDConfig
from memorydraft.evaluation.asd_comparison import bootstrap
from memorydraft.evaluation.asd_paper import (BENCHMARKS,conversation_record,expanded_ids,
    aggregate as paper_aggregate,request_for,sha)
from memorydraft.evaluation.asd_paper_metrics import acceptance

STUDY='asd-paper-extension'
SEED='MemoryDraft-asd-paper-extension-v1'
COUNTS=dict(gsm8k=128,math500=64,humaneval=64,mbpp=64,mmlu=114,mtbench=16,alpaca=64)
ARMS=('DFlash','ASD','BudgetTree','BudgetTree-ASD')
REPEATS=dict(natural=2,prefix=1,fixed=1)
CONTROLS={'B4':ASDConfig(budget=4),'B12':ASDConfig(budget=12),
    'g.125':ASDConfig(budget=8,gate=.125),'g.5':ASDConfig(budget=8,gate=.5),
    'M1':ASDConfig(budget=8,cap=1)}


def load_records(root):
    directory=Path(root)/'data/research'/STUDY
    manifest=json.loads((directory/'manifest.json').read_bytes());rows=[]
    if manifest['schema']!=1 or manifest['seed']!=SEED or manifest['counts']!=COUNTS:
        raise ValueError('Changed extension protocol')
    for benchmark in BENCHMARKS:
        name=benchmark+'.json';raw=(directory/name).read_bytes();selected=json.loads(raw)
        if sha(raw)!=manifest['files'][name] or len(selected)!=COUNTS[benchmark]:
            raise ValueError('Changed extension data: '+benchmark)
        if any(r['benchmark']!=benchmark for r in selected):raise ValueError('Wrong benchmark')
        rows.extend(selected)
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate extension task')
    excluded=json.loads((directory/'exclusions.json').read_bytes())
    if sha((directory/'exclusions.json').read_bytes())!=manifest['exclusions_sha256']:
        raise ValueError('Changed exposure exclusion manifest')
    if {r['id'] for r in rows}&set(excluded['excluded_ids']):raise ValueError('Prior exposure overlap')
    if manifest['controls']!={k:asdict(v) for k,v in CONTROLS.items()}:
        raise ValueError('Changed frontier controls')
    expected=[r['id'] for b in BENCHMARKS for r in [x for x in rows if x['benchmark']==b][:4]]
    if manifest['frontier_ids']!=expected:raise ValueError('Changed frontier selection')
    return rows


def frontier_ids(records):
    return {r['id'] for b in BENCHMARKS for r in [x for x in records if x['benchmark']==b][:4]}


def read_rows(directory,name):
    rows=[]
    for shard in (0,1):
        path=Path(directory)/f'part-{shard}/{name}.jsonl'
        raw=path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix('.jsonl.gz').read_bytes())
        rows.extend(json.loads(line) for line in raw.decode().splitlines())
    return rows


def signature(row):
    return tuple(row[k] for k in ('token_ids','rounds','asd','finish_reason','text'))


def normalize_fixed(rows):
    """Derive one strict timing per adjacent pair; retain every original role."""
    output=[r for r in rows if r['stage']!='fixed'];groups=defaultdict(dict);drifts=[]
    for row in rows:
        if row['stage']!='fixed':continue
        key=row['request_id'],row['arm'];role=row['role']
        if role in groups[key]:raise ValueError('Duplicate fixed role')
        groups[key][role]=row
    for (identity,arm),roles in sorted(groups.items()):
        strict=arm in ('DFlash','BudgetTree')
        if set(roles)!=({'before','after'} if strict else {'candidate'}):raise ValueError('Missing fixed role')
        chosen=roles['before'] if strict else roles['candidate']
        if any((r['prompt_sha256'],r['snapshot_id'])!=(chosen['prompt_sha256'],chosen['snapshot_id'])
               or len(r['token_ids'])!=256 or r['stop_policy']!='ignore-eos'
               or not math.isfinite(r['latency_seconds']) or r['latency_seconds']<=0 for r in roles.values()):
            raise ValueError('Invalid fixed role input, work or timing')
        if strict and signature(roles['before'])!=signature(roles['after']):raise ValueError('Strict fixed hash drift')
        result=dict(chosen,repeat=0,latency_seconds=statistics.mean(r['latency_seconds'] for r in roles.values()))
        output.append(result)
        if strict:
            drift=roles['after']['latency_seconds']/roles['before']['latency_seconds']
            drifts.append(dict(request_id=identity,benchmark=chosen['benchmark'],arm=arm,ratio=drift,
                within_3pct=.97<=drift<=1.03))
    return output,drifts


def validate_policy(row,config):
    summary=row['asd'];blocks=row['rounds']
    if not summary or not 0<=summary['spent']<=config.budget:raise ValueError('Control budget exceeded')
    if any(not 0<=b['relaxed_tokens']<=config.cap or not math.isfinite(b['regret']) or b['regret']<0 for b in blocks):
        raise ValueError('Invalid selected-path control accounting')
    if not math.isclose(sum(b['regret'] for b in blocks),summary['spent'],abs_tol=1e-8):
        raise ValueError('Selected-path regret differs from request ledger')


def aggregate(rows,references,records,frontier):
    normalized,drifts=normalize_fixed(rows)
    result=paper_aggregate(normalized,references,records,arms=ARMS,repeats=REPEATS)
    expected=set(expanded_ids(records))
    if len(rows)!=len(expected)*18:raise ValueError('Incomplete raw four-arm matrix')
    if len(drifts)!=len(expected)*2:raise ValueError('Incomplete fixed adjacent controls')
    result.update(raw_timed_generations=len(rows),fixed_drift=drifts,
        fixed_drift_over_3pct=sum(not r['within_3pct'] for r in drifts),drift_selection='All requests retained')
    indexed={(r['request_id'],r['stage'],r['arm']):r for r in normalized if r['repeat']==0}
    mechanism=defaultdict(lambda:defaultdict(int))
    for row in indexed.values():
        values=acceptance(row) if row['rounds'] else dict(rounds=0,proposed_draft=0,selected_draft=0,
            emitted_draft=0,output_tokens=len(row['token_ids']),verification_nodes=0)
        for key,value in values.items():mechanism[row['benchmark'],row['stage'],row['arm']][key]+=value
    result['acceptance']=[dict(benchmark=b,stage=s,arm=a,**v,
        tau_emitted=v['emitted_draft']/v['rounds'] if v['rounds'] else None,
        nodes_per_output_token=v['verification_nodes']/v['output_tokens'] if v['output_tokens'] else None)
        for (b,s,a),v in sorted(mechanism.items())]
    subset=[r for r in records if r['id'] in frontier_ids(records)];wanted=set(expanded_ids(subset))
    cells={}
    for row in frontier:
        key=row['request_id'],row['control'],row['stage'],row['arm']
        if key in cells:raise ValueError('Duplicate frontier cell')
        cells[key]=row
        if row['control'] not in CONTROLS or row['arm'] not in ('ASD','BudgetTree-ASD'):
            raise ValueError('Unknown frontier control')
        if row['asd_config']!=asdict(CONTROLS[row['control']]):raise ValueError('Changed control config')
        validate_policy(row,CONTROLS[row['control']])
        ref=indexed[row['request_id'],row['stage'],row['arm']]
        if (row['prompt_sha256'],row['snapshot_id'])!=(ref['prompt_sha256'],ref['snapshot_id']):
            raise ValueError('Control input mismatch')
        if not math.isfinite(row['latency_seconds']) or row['latency_seconds']<=0:raise ValueError('Bad control timing')
        if row['stage']=='fixed' and len(row['token_ids'])!=256:raise ValueError('Bad control workload')
    if set(cells)!={(i,c,s,a) for i in wanted for c in CONTROLS for s in ('natural','fixed') for a in ('ASD','BudgetTree-ASD')}:
        raise ValueError('Incomplete frontier matrix')
    summaries=[]
    for b in BENCHMARKS:
        identities=[i for i in wanted if indexed[i,'natural','ASD']['benchmark']==b]
        for control in ('B0','B8',*CONTROLS):
            for arm in ('ASD','BudgetTree-ASD'):
                strict='DFlash' if arm=='ASD' else 'BudgetTree'
                chosen=lambda i,s:(indexed[i,s,strict if control=='B0' else arm] if control in ('B0','B8')
                                   else cells[i,control,s,arm])
                natural=[chosen(i,'natural') for i in sorted(identities)]
                fixed=[chosen(i,'fixed') for i in sorted(identities)]
                ratios=[chosen(i,'natural')['latency_seconds']/indexed[i,'natural','ASD']['latency_seconds']
                        for i in sorted(identities)]
                known=[r['quality'] for r in natural if r['quality']['correct'] is not None]
                summaries.append(dict(benchmark=b,control=control,arm=arm,requests=len(identities),
                    natural_vs_asd=statistics.mean(ratios),ci95=bootstrap(ratios),
                    fixed_tps=256*len(fixed)/sum(r['latency_seconds'] for r in fixed),
                    correct=sum(q.get('secondary_correct',q['correct']) for q in known),scored=len(known),
                    truncated=sum(r['finish_reason']=='length' for r in natural),
                    tokens=sum(len(r['token_ids']) for r in natural),
                    source='primary repeat0 roles' if control in ('B0','B8') else 'one-at-a-time control run'))
    result.update(frontier=summaries,frontier_generations=len(frontier),frontier_requests=len(wanted))
    return result


def open_units(root,directory):
    pool={r['id']:r for r in load_records(root)};units={}
    for row in read_rows(directory,'rows'):
        if row['benchmark']!='alpaca' or row['stage']!='natural' or row['repeat']!=0:continue
        task=pool[row['request_id']]
        unit=units.setdefault(task['id'],dict(request_id=task['id'],benchmark='alpaca',answers={}))
        unit['answers'][row['arm']]=[*request_for(task,4096).messages,dict(role='assistant',content=row['text'])]
    for row in read_rows(directory,'rollout'):
        task=pool[row['conversation_id']]
        unit=units.setdefault(task['id'],dict(request_id=task['id'],benchmark='mtbench',answers={}))
        unit['answers'][row['arm']]=[*conversation_record(task,2,row['first_text'])['messages'],
                                   dict(role='assistant',content=row['text'])]
    if len(units)!=COUNTS['alpaca']+COUNTS['mtbench'] or any(set(u['answers'])!=set(ARMS) for u in units.values()):
        raise ValueError('Incomplete extension open quality matrix')
    return [units[k] for k in sorted(units)]
