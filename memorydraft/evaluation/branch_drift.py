"""Shared inputs, paired timing and AR identity diagnostics for branch drift."""
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_paper import BENCHMARKS, expanded_ids, quality_ok, sha
from memorydraft.evaluation.asd_paper_extension import load_records as extension_records

STUDY = 'branch-drift'
SEED = 'MemoryDraft-branch-drift-v1'
COUNTS = dict(gsm8k=16, math500=12, humaneval=16, mbpp=16, mmlu=57, mtbench=8, alpaca=16)
ARMS = ('DFlash', 'ASD', 'BudgetTree', 'BudgetTree-ASD', 'CoverageTree-ASD', 'DriftTree-ASD')
PREFIX_ARMS = ('ASD', 'BudgetTree', 'BudgetTree-ASD', 'CoverageTree-ASD', 'DriftTree-ASD')


def records(root):
    pool = extension_records(root)
    chosen = []
    for benchmark in BENCHMARKS:
        selected = sorted((r for r in pool if r['benchmark'] == benchmark),
                          key=lambda r: sha((SEED+':'+r['id']).encode()))
        if benchmark == 'mmlu':
            groups = defaultdict(list)
            for row in selected:
                groups[row['subject']].append(row)
            selected = [groups[k][0] for k in sorted(groups)]
        if len(selected) < COUNTS[benchmark]:
            raise ValueError('Insufficient benchmark coverage: '+benchmark)
        chosen.extend(selected[:COUNTS[benchmark]])
    return chosen


def references(root, selected):
    wanted = set(expanded_ids(selected))
    rows = []
    for shard in (0, 1):
        path = Path(root)/f'results/asd-paper-extension/part-{shard}/references.jsonl.gz'
        for line in gzip.decompress(path.read_bytes()).decode().splitlines():
            row = json.loads(line)
            if row['request_id'] in wanted:
                rows.append(row)
    if {r['request_id'] for r in rows} != wanted or len(rows) != len(wanted):
        raise ValueError('Incomplete frozen AR references')
    return rows


def prefix_length(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def summarize(rows, refs):
    indexed = {r['request_id']: r for r in refs}
    cells = {}
    for row in rows:
        key = row['request_id'], row['stage'], row['arm']
        if key in cells:
            raise ValueError('Duplicate benchmark cell')
        cells[key] = row
        ref = indexed[row['request_id']]
        if ((row['prompt_sha256'], row['snapshot_id']) != (ref['prompt_sha256'], ref['snapshot_id'])
                or not math.isfinite(row['latency_seconds']) or row['latency_seconds'] <= 0):
            raise ValueError('Input or timing mismatch')
        if row['asd'] and not 0 <= row['asd']['spent'] <= 8:
            raise ValueError('Invalid ASD ledger')
        if row['asd'] and not math.isclose(sum(b['regret'] for b in row['rounds']),
                                          row['asd']['spent'], abs_tol=1e-7):
            raise ValueError('Regret accounting mismatch')
    expected = {(i, s, a) for i in indexed for s, arms in (('natural', ARMS), ('prefix', PREFIX_ARMS)) for a in arms}
    if set(cells) != expected:
        raise ValueError('Incomplete benchmark matrix')
    families = {}
    for benchmark in BENCHMARKS:
        ids = [i for i, r in indexed.items() if r['benchmark'] == benchmark]
        family = {}
        for arm in ARMS:
            values = {}
            for stage, arms in (('natural', ARMS), ('prefix', PREFIX_ARMS)):
                if arm not in arms:
                    continue
                selected = [cells[i, stage, arm] for i in ids]
                tokens = sum(len(r['token_ids']) for r in selected)
                seconds = sum(r['latency_seconds'] for r in selected)
                quality = [quality_ok(r['quality']) for r in selected if stage == 'natural']
                v = dict(n=len(selected), seconds=seconds, tokens=tokens, tps=tokens/seconds,
                    calls=sum(len(r['rounds']) for r in selected),
                    scored=sum(q is not None for q in quality), correct=sum(q is True for q in quality),
                    truncated=sum(r['finish_reason'] == 'length' for r in selected),
                    identical_ar=sum(r['token_ids'] == indexed[r['request_id']]['token_ids'][:r['max_new_tokens']]
                                     for r in selected),
                    ar_prefix_fraction=statistics.mean(prefix_length(r['token_ids'], indexed[r['request_id']]['token_ids'])
                        /max(1, min(len(r['token_ids']), len(indexed[r['request_id']]['token_ids']))) for r in selected),
                    relaxed=sum((r['asd'] or {}).get('relaxed_tokens', 0) for r in selected),
                    spent=sum((r['asd'] or {}).get('spent', 0.) for r in selected))
                for baseline in arms:
                    pairs = [(r, cells[r['request_id'], stage, baseline]) for r in selected]
                    common = [(a, b) for a, b in pairs if stage == 'natural' or len(a['token_ids']) == len(b['token_ids']) == 128]
                    v[baseline] = dict(n=len(common), ratio=statistics.mean(a['latency_seconds']/b['latency_seconds']
                        for a, b in common) if common else None,
                        length_ratio=statistics.mean(len(a['token_ids'])/max(1, len(b['token_ids'])) for a, b in common) if common else None)
                    if stage == 'natural':
                        v[baseline].update(lost=[a['request_id'] for a, b in pairs
                            if quality_ok(b['quality']) is True and quality_ok(a['quality']) is False],
                            gained=[a['request_id'] for a, b in pairs
                            if quality_ok(a['quality']) is True and quality_ok(b['quality']) is False])
                counters = defaultdict(float)
                for r in selected:
                    for k, value in (r['asd'] or {}).get('branch_drift', {}).items():
                        if type(value) in (int, float) and k not in ('max_js', 'temperature'):
                            if k.endswith('_max'):
                                counters[k] = max(counters[k], value)
                            else:
                                counters[k] += value
                v['drift'] = dict(counters)
                values[stage] = v
            family[arm] = values
        families[benchmark] = family
    return dict(timed_generations=len(rows), reference_generations=len(refs),
        reference_policy='Frozen prior AR outputs; AR latency not compared across sessions', families=families)
