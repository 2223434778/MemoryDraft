"""Source/evidence audit; historical publications are checked at immutable commits."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from memorydraft.evaluation.asd_suite import score
from memorydraft.evaluation.long_context import load_data
from memorydraft.evaluation.prefix_cache import analyze

root=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
base=root/'results/asd-shared-prefix'
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))


def blobs(commit):
    entries=[]
    for record in filter(None,subprocess.check_output(['git','ls-tree','-rz',commit],cwd=root).split(b'\0')):
        meta,name=record.split(b'\t',1);mode,kind,oid=meta.split()
        assert mode in (b'100644',b'100755') and kind==b'blob'
        entries.append((name.decode(),oid.decode()))
    raw=subprocess.check_output(['git','cat-file','--batch'],input=''.join(oid+'\n' for _,oid in entries).encode(),cwd=root)
    offset,contents=0,{}
    for name,oid in entries:
        end=raw.index(b'\n',offset);header=raw[offset:end].split()
        assert header[:2]==[oid.encode(),b'blob']
        size=int(header[2]);contents[name]=raw[end+1:end+1+size];offset=end+size+2
    return contents


manifest=read(base/'ASD_SHARED_PREFIX_SOURCE_MANIFEST.json')
canonical=blobs(manifest['commit'])
assert {n:sha(b) for n,b in canonical.items()}==manifest['files']
assert sha(json.dumps(manifest['files'],sort_keys=True).encode())==manifest['source_sha256']
legacy,protected,frozen=[],0,0
for n,data in canonical.items():
    if n.startswith('results/'):
        actual=(root/n).read_bytes()
        if actual!=data:
            assert n in {'results/research-audit/model_preflight.json','results/research-audit/pilot_summary.json'}
            assert actual.replace(b'\r\n',b'\n')==data;legacy.append(n)
        protected+=1
    if (n.startswith('memorydraft/') or n.startswith('data/workloads/')
            or n.startswith('scripts/') and ('shared_prefix' in n or n.endswith('run_asd_prefix_cache.py'))
            or n.startswith('tests/test_shared_prefix') or n=='docs/ASD_SHARED_PREFIX_PROTOCOL.md'
            or n=='configs/runtime-dflash-asd-shared-prefix.json'):
        assert sha((root/n).read_bytes())==manifest['files'][n],n
        frozen+=1
prior=blobs('de1d4e43452cf5441115a2880eff718e22ebfca7')
publication=json.loads(prior['results/asd-prefix-cache/publication.json'])
assert all(sha(prior[n])==h for n,h in publication['files'].items())
for n in ('memorydraft/runtime/prefix_cache.py','memorydraft/runtime/engine.py','memorydraft/backends/dflash_reuse.py',
          'memorydraft/backends/asd.py','data/workloads/asd-prefix-cache/public-prefixes.json',
          'configs/runtime-dflash-asd-prefix-cache.json'):
    assert canonical[n]==prior[n],n
execution=read(base/'execution-verification.json')
assert execution['status']=='complete'
assert execution['files_verified_before']==execution['files_verified_after']==len(canonical)
tasks,items,_=load_data(root)
tasks={r['id']:r for r in tasks}
items=[r for r in items if r['level']==4096]
rows=[json.loads(s) for i in (0,1) for s in (base/f'eval-shared-{i}/rows.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(rows)==96
assert all(r['quality']==score(tasks[r['task_id']],r['text']) for r in rows if r['stage']=='natural')
recomputed=analyze(rows,items)
analysis=read(base/'analysis.json')
assert all(analysis[k]==v for k,v in recomputed.items())
assert analysis['all_strict_match_previous_ar'] and not analysis['comparison_to_previous_differences']
assert analysis['all_outputs_and_decisions_equal']
assert (base/'analysis.json').read_bytes()==(base/'analysis-server.json').read_bytes()
http_count=0
for i in (0,1):
    for v in ('shared','whole'):
        d=base/f'eval-{v}-{i}'
        meta=read(d/'provenance.json')
        assert meta['status']=='complete' and meta['rows']==(48 if v=='shared' else 0)
        config=meta['runtime']['engine']['config']
        assert config['prefix_cache']['max_bytes']==2*2**30 and config['prefix_cache']['max_entries']==3
        assert config['asd']==dict(budget=8,gate=.25,cap=2)
        assert meta['compiler']['graph_count']<=8
        load=[json.loads(s) for s in (d/'http-load.jsonl').read_text(encoding='utf-8').splitlines()]
        http_count+=len(load)
        assert all(r['result']['peak_gpu_bytes']<=46*2**30 for r in load)
        assert analysis['http'][str(i)]['outputs_and_aggregate_counters_equal']
        http=read(d/'http-summary.json')
        assert http['status']=='complete' and all(http[k] for k in ('json_and_sse_match','disconnect_cancelled','recovered'))
        cross=read(d/'cross-question.json')
        assert len(cross)==4 and not cross[0]['counters']['prefix_cache_hit']
        if v=='shared':assert sum(r['counters']['prefix_cache_hit'] for r in cross)==3
assert http_count==120
archive=root/'artifacts/asd-shared-prefix-results.zip'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all((base/n).read_bytes()==z.read(n) for n in z.namelist())
    members=len(z.namelist())
infra={n:subprocess.check_output(['git','rev-parse',n],cwd=root).decode().strip() for n in
       ('main','codex/infra-interview-v0.3','infra-v0.3-interview^{}')}
assert set(infra.values())=={'27e36199f8a65acdb9a9c42ccb744ddfa9a9402b'}
result=dict(status='complete',execution_commit=manifest['commit'],canonical_git_blobs=len(canonical),frozen_files=frozen,
    previous_result_files=protected,previous_publication_files=len(publication['files']),legacy_crlf=legacy,
    formal_rows=len(rows),outputs_and_aggregate_counters_equal=True,all_strict_match_previous_ar=True,
    http_load_requests=http_count,server_local_analysis_equal=True,online_runtime_unchanged=True,infra_refs=infra,
    archive_bytes=archive.stat().st_size,archive_sha256=sha(archive.read_bytes()),archive_members=members)
(base/'postrun-audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result))
