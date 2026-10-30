"""Attribute existing GSM latency without changing or filtering benchmark rows."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.diagnose_branch_drift_math import factors

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('ASD', 'BudgetTree', 'DriftTree-ASD', 'LazyDrift-ASD',
        'GateTree-ASD', 'TokenCap-ASD', 'GateCap-ASD')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'results/branch-drift-optimization'
    published = json.loads((source/'publication.json').read_bytes())['payloads']
    if any(digest(source/name) != sha for name, sha in published.items()):
        raise ValueError('Original publication changed')
    rows = [json.loads(line) for line in gzip.decompress((source/'rows.jsonl.gz').read_bytes()).decode().splitlines()]
    refs = {r['request_id']: r for r in json.loads(gzip.decompress((source/'references.json.gz').read_bytes()))}
    cells = {(r['request_id'], r['stage'], r['arm']): r for r in rows if r['benchmark'] == 'gsm8k'}
    ids = sorted({i for i, _, _ in cells})
    expected = {(i, stage, arm) for i in ids for stage in ('natural', 'prefix') for arm in ARMS}
    if len(ids) != 8 or set(cells) != expected:
        raise ValueError('Incomplete frozen GSM matrix')
    totals = {}
    for stage in ('natural', 'prefix'):
        totals[stage] = {}
        for arm in ARMS:
            selected = [cells[i, stage, arm] for i in ids]
            blocks = [b for r in selected for b in r['rounds']]
            tree = [b['tree_budget'] for b in blocks if b.get('tree_budget')]
            value = dict(seconds=sum(r['latency_seconds'] for r in selected),
                tokens=sum(len(r['token_ids']) for r in selected), rounds=len(blocks),
                relaxed=sum((r['asd'] or {}).get('relaxed_tokens', 0) for r in selected),
                correct=sum(r['quality']['correct'] is True for r in selected) if stage == 'natural' else None,
                identical_ar=sum(r['token_ids'] == refs[r['request_id']]['token_ids'][:r['max_new_tokens']] for r in selected),
                gate_seconds=sum((r['asd'] or {}).get('branch_drift', {}).get('gate_seconds', 0) for r in selected),
                js_seconds=sum((r['asd'] or {}).get('branch_drift', {}).get('js_seconds', 0) for r in selected),
                mean_nodes=statistics.mean(b['nodes'] for b in tree) if tree else None,
                mean_depth=statistics.mean(b['max_depth'] for b in tree) if tree else None)
            value['tps'] = value['tokens']/value['seconds']
            totals[stage][arm] = value
    cases = []
    for identity in ids:
        values = {}
        for arm in ARMS:
            r = cells[identity, 'natural', arm]
            values[arm] = dict(seconds=r['latency_seconds'], tokens=len(r['token_ids']),
                rounds=len(r['rounds']), relaxed=(r['asd'] or {}).get('relaxed_tokens', 0),
                identical_ar=r['token_ids'] == refs[identity]['token_ids'],
                correct=r['quality']['correct'], finish=r['finish_reason'])
        a, c = values['ASD'], values['GateCap-ASD']
        cases.append(dict(request_id=identity, arms=values, extra_seconds=c['seconds']-a['seconds'],
                          factors=factors(c, a)))
    paired = statistics.mean(c['factors']['latency_ratio'] for c in cases)
    family = json.loads((source/'analysis.json').read_bytes())['families']['gsm8k']
    if not math.isclose(paired, family['GateCap-ASD']['natural']['ASD']['ratio'], rel_tol=1e-12):
        raise ValueError('Published paired mean mismatch')
    same = [i for i in ids if cells[i, 'natural', 'GateCap-ASD']['token_ids']
            == cells[i, 'natural', 'BudgetTree']['token_ids']]
    allocation_cap_equal = all(cells[i, 'natural', 'GateTree-ASD']['token_ids']
        == cells[i, 'natural', 'GateCap-ASD']['token_ids']
        and len(cells[i, 'natural', 'GateTree-ASD']['rounds'])
        == len(cells[i, 'natural', 'GateCap-ASD']['rounds']) for i in ids)
    if not allocation_cap_equal:
        raise ValueError('The token-cap attribution control differs')
    extra = totals['natural']['GateCap-ASD']['seconds']-totals['natural']['ASD']['seconds']
    for case in cases:
        case['share_of_net_extra_wall'] = case['extra_seconds']/extra
    cases.sort(key=lambda c: c['extra_seconds'], reverse=True)
    result = dict(source_payloads_checked=len(published),
        source_publication_sha256=digest(source/'publication.json'),
        source_files_sha256={name: digest(source/name) for name in ('rows.jsonl.gz', 'references.json.gz', 'analysis.json')},
        requests=8, cases=cases, totals=totals, paired_mean_ratio=paired,
        factors=factors(totals['natural']['GateCap-ASD'], totals['natural']['ASD']),
        without_750_attribution_only=dict(n=7, ratio=statistics.mean(c['factors']['latency_ratio']
            for c in cases if c['request_id'] != 'gsm8k:750'), official_all_eight_retained=True),
        same_output_strict_tree_control=dict(ids=same, n=len(same),
            paired_ratio=statistics.mean(cells[i, 'natural', 'GateCap-ASD']['latency_seconds']
                /cells[i, 'natural', 'BudgetTree']['latency_seconds'] for i in same)),
        gate_allocation_and_cap_equal=allocation_cap_equal,
        common128_vs_asd=family['GateCap-ASD']['prefix']['ASD'],
        inference_changes=False, new_gpu_generations=0,
        limitations='Accounting factors are identities, not separate causal timings; tree-width mechanism needs a new controlled test')
    out = ROOT/'results/branch-drift-gsm-diagnosis'
    out.mkdir(parents=True, exist_ok=True)
    (out/'analysis.json').write_bytes((json.dumps(result, ensure_ascii=True, indent=2)+'\n').encode())
    (out/'publication.json').write_bytes((json.dumps(dict(
        script_sha256=digest(Path(__file__)), analysis_sha256=digest(out/'analysis.json'),
        source_payloads_checked=len(published)), indent=2)+'\n').encode())
    if any(digest(source/name) != sha for name, sha in published.items()):
        raise ValueError('Original publication changed during diagnosis')
    print(json.dumps(dict(requests=8, cells=len(cells), paired_mean_ratio=paired,
        total_factors=result['factors'], net_extra_seconds=extra,
        focal_share=cases[0]['share_of_net_extra_wall'], new_gpu_generations=0)))


if __name__ == '__main__':
    main()
