"""Recompute evidence, verify immutable historical publications and protected refs."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from memorydraft.evaluation.long_context import load_data
from memorydraft.evaluation.prefix_cache import analyze
from memorydraft.evaluation.asd_suite import score

root = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
base = root/'results/asd-prefix-cache'
sha = lambda b:hashlib.sha256(b).hexdigest()
read = lambda p:json.loads(p.read_text(encoding='utf-8'))


def blobs(commit):
    entries=[]
    for record in filter(None,subprocess.check_output(['git','ls-tree','-rz',commit],cwd=root).split(b'\0')):
        meta,name=record.split(b'\t',1)
        mode,kind,oid=meta.split()
        assert mode in (b'100644',b'100755') and kind==b'blob'
        entries.append((name.decode(),oid.decode()))
    raw=subprocess.check_output(['git','cat-file','--batch'],input=''.join(oid+'\n' for _,oid in entries).encode(),cwd=root)
    offset,contents=0,{}
    for name,oid in entries:
        end=raw.index(b'\n',offset);header=raw[offset:end].split()
        assert header[:2]==[oid.encode(),b'blob']
        size=int(header[2]);contents[name]=raw[end+1:end+1+size];offset=end+size+2
    return contents


manifest=read(base/'ASD_PREFIX_CACHE_SOURCE_MANIFEST.json')
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
            or n.startswith('scripts/') and ('prefix_cache' in n or n.endswith('prepare_public_prefixes.py'))
            or n.startswith('tests/test_prefix_cache') or n=='docs/ASD_PREFIX_CACHE_PROTOCOL.md'):
        assert sha((root/n).read_bytes())==manifest['files'][n],n
        frozen+=1
prior=blobs('c303bc852ab822fbfc01e9a884df2aaa65341ceb')
publication=json.loads(prior['results/asd-long-context/publication.json'])
assert all(sha(prior[n])==h for n,h in publication['files'].items())
execution=read(base/'execution-verification.json')
assert execution['status']=='complete'
assert execution['files_verified_before']==execution['files_verified_after']==len(canonical)
tasks,items,_=load_data(root)
tasks={r['id']:r for r in tasks}
rows=[json.loads(line) for i in (0,1) for line in (base/f'eval-{i}/rows.jsonl').read_text(encoding='utf-8').splitlines()]
assert all(r['quality']==score(tasks[r['task_id']],r['text']) for r in rows if r['stage']=='natural')
recomputed=analyze(rows,items)
analysis=read(base/'analysis.json')
assert all(analysis[k]==v for k,v in recomputed.items())
assert analysis['all_strict_match_previous_ar'] and not analysis['comparison_to_previous_differences']
assert recomputed['all_outputs_and_decisions_equal']
assert (base/'analysis.json').read_bytes()==(base/'analysis-server.json').read_bytes()
assert read(base/'analysis-local.json')==analysis
for i in (0,1):
    directory=base/f'eval-{i}'
    meta=read(directory/'provenance.json')
    assert meta['status']=='complete' and meta['rows']==144
    http=read(directory/'http-summary.json')
    assert http['status']=='complete' and all(http[k] for k in ('json_and_sse_match','disconnect_cancelled','recovered'))
    assert http['off']['requests']==http['cache']['requests']==20
    load=[json.loads(line) for line in (directory/'http-load.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(load)==40
    assert http['cache']['hits']==sum(r['result']['counters'].get('prefix_cache_hit',0) for r in load if r['state']=='cache')
    assert all(r['result']['counters'].get('prefix_cache_bytes',0)<=2*2**30 for r in load)
    cross=read(directory/'cross-question.json')
    assert len(cross)==4 and not cross[0]['counters']['prefix_cache_hit']
archive_path=root/'artifacts/asd-prefix-cache-v2-results.zip'
with zipfile.ZipFile(archive_path) as z:
    assert z.testzip() is None
    assert all((base/n).read_bytes()==z.read(n) for n in z.namelist())
    members=len(z.namelist())
infra={n:subprocess.check_output(['git','rev-parse',n],cwd=root).decode().strip()
       for n in ('main','codex/infra-interview-v0.3','infra-v0.3-interview^{}')}
assert set(infra.values())=={'27e36199f8a65acdb9a9c42ccb744ddfa9a9402b'}
result=dict(status='complete',execution_commit=manifest['commit'],canonical_git_blobs=len(canonical),frozen_files=frozen,
    previous_result_files=protected,previous_publication_files=len(publication['files']),legacy_crlf=legacy,
    formal_rows=len(rows),outputs_and_aggregate_counters_equal=True,all_strict_match_previous_ar=True,
    http_requests=80,cross_question_and_cancellation=True,server_local_analysis_equal=True,infra_refs=infra,
    archive_bytes=archive_path.stat().st_size,archive_sha256=sha(archive_path.read_bytes()),archive_members=members)
(base/'postrun-audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result))
