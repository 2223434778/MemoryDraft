"""Rebuild the matched engineering matrix and audit original publication bytes."""
import gzip
import hashlib
import importlib
import json
from pathlib import Path

from memorydraft.evaluation.asd_comparison import BASELINES,ADAPTERS
from memorydraft.evaluation.runtime_migration import read_rows
from scripts.run_asd_comparison_engineering import analyze,candidate


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    base=Path(__file__).resolve().parent
    root=base.parents[1]
    main_raw=(base.parent/'asd-comparison/analysis.json').read_bytes()
    execution=json.loads((base/'execution-verification.json').read_bytes())
    if (execution['status']!='complete' or execution['files_verified_before']!=execution['files_verified_after']
            or execution['comparison_analysis_sha256']!=digest(main_raw)):
        raise ValueError('Engineering source or upstream comparison changed')
    arms=(*BASELINES,candidate(json.loads(main_raw)))
    infos=[json.loads((base/f'eval-{i}/provenance.json').read_bytes()) for i in (0,1)]
    if any(p['status']!='complete' or not p['original_unchanged'] or tuple(p['arms'])!=arms for p in infos):
        raise ValueError('Incomplete or changed engineering workers')
    rows=[r for i in (0,1) for r in read_rows(base/f'eval-{i}/rows.jsonl')]
    refs=[r for i in (0,1) for r in read_rows(base/f'eval-{i}/references.jsonl')]
    expected={r['request_id']:r for r in refs}
    if len(rows)!=144 or len(refs)!=8 or len(expected)!=8:
        raise ValueError('Changed engineering coverage')
    for r in rows:
        ref=expected[r['request_id']]
        if (r['prompt_sha256'],r['snapshot_id'])!=(ref['prompt_sha256'],ref['snapshot_id']):
            raise ValueError('Actual engineering input mismatch')
        if r['arm'].endswith(('/DFlash','/tree32')) and (r['token_ids'],r['text'],r['finish_reason'])!=(ref['token_ids'],ref['text'],ref['finish_reason']):
            raise ValueError('Actual engineering strict output mismatch')
    rebuilt=analyze(rows,arms)
    resource=execution['resources']
    if (max(r['peak_gpu_bytes'] for r in rows)!=resource['peak_gpu_bytes'] or resource['peak_gpu_bytes']>46*2**30
            or sum(r['counters']['compile_graphs_after']-r['counters']['compile_graphs_before'] for r in rows)!=0
            or resource['formal_compile_events']!=0
            or any(v>4*2**30 for v in resource['cache_bytes'].values())):
        raise ValueError('Engineering resources changed')
    rebuilt['resources']=resource
    if rebuilt!=json.loads((base/'analysis.json').read_bytes()):raise ValueError('Engineering statistics changed')
    for v,(_,sha) in ADAPTERS.items():
        if execution['previous_adapters_unchanged'][v]!=sha:raise ValueError('Previous adapter changed')
    publication=json.loads((base/'publication.json').read_bytes())
    for entry in publication['files']:
        data=(base/entry['published']).read_bytes()
        if digest(data)!=entry['published_sha256']:raise ValueError('Engineering publication changed')
        raw=gzip.decompress(data) if entry['published'].endswith('.gz') else data
        if digest(raw)!=entry['raw_sha256']:raise ValueError('Engineering original payload changed')
    module=importlib.import_module('results.asd-comparison.audit_postrun')
    source=module.git_source(root,json.loads((base/'ASD_COMPARISON_ENGINEERING_SOURCE_MANIFEST.json').read_bytes()))
    (root/'artifacts/comparison-engineering-source-audit.json').write_bytes((json.dumps(source,indent=2)+'\n').encode())
    print(json.dumps(dict(rebuilt_exact=True,rows=len(rows),requests=len(expected),source={k:v for k,v in source.items() if k!='line_endings'})))
    print(json.dumps(rebuilt['summary'],sort_keys=True))


if __name__=='__main__':main()
