"""Postrun public audit; does not generate tokens, train, or alter prior evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from memorydraft.evaluation.adaptation_followup import STUDY, ADAPTER_SHA, aligned_controls, previous_rows, records
from memorydraft.evaluation.runtime_migration import read_rows
from scripts.write_asd_adaptation_followup_report import analyze

root=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
base=root/f'results/{STUDY}'
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((base/'ASD_ADAPTATION_FOLLOWUP_SOURCE_MANIFEST.json').read_text())
assert sha(json.dumps(manifest['files'],sort_keys=True).encode())==manifest['source_sha256']
listing=subprocess.check_output(['git','ls-tree','-rz',manifest['commit']],cwd=root).split(b'\0')
entries=[]
for entry in filter(None,listing):
    meta,name=entry.split(b'\t',1)
    mode,kind,oid=meta.split()
    assert kind==b'blob' and mode in (b'100644',b'100755')
    entries.append((name.decode(),oid.decode()))
assert {n for n,_ in entries}==set(manifest['files'])
raw=subprocess.check_output(['git','cat-file','--batch'],input=''.join(oid+'\n' for _,oid in entries).encode(),cwd=root)
offset=0
for name,oid in entries:
    end=raw.index(b'\n',offset)
    header=raw[offset:end].split()
    assert header[:2]==[oid.encode(),b'blob']
    size=int(header[2]);content=raw[end+1:end+1+size];offset=end+size+2
    assert sha(content)==manifest['files'][name],name
reproduction=json.loads((base/'reproduction-fix.json').read_text())
assert reproduction['execution_commit']==manifest['commit']
for n,h in reproduction['files'].items():
    assert sha(subprocess.check_output(['git','show',reproduction['commit']+':'+n],cwd=root))==h
    assert sha((root/n).read_bytes())==h
frozen={n:h for n,h in manifest['files'].items() if n.startswith(('data/research/asd-adaptation-followup/','scripts/')) and 'adaptation_followup' in n
        or n in {'memorydraft/evaluation/adaptation_followup.py','tests/test_adaptation_followup.py','docs/ASD_ADAPTATION_FOLLOWUP_PROTOCOL.md'}}
frozen.update({n:h for n,h in manifest['files'].items() if n.startswith('data/research/asd-adaptation-followup/')})
for n in reproduction['files']:frozen.pop(n,None)
for n,h in frozen.items():assert sha((root/n).read_bytes())==h,n
legacy=[];protected=0
for n,h in manifest['files'].items():
    if not n.startswith('results/'):continue
    data=(root/n).read_bytes()
    if sha(data)!=h:
        assert n in {'results/research-audit/model_preflight.json','results/research-audit/pilot_summary.json'}
        assert sha(data.replace(b'\r\n',b'\n'))==h,n
        legacy.append(n)
    protected+=1
execution=json.loads((base/'execution-verification.json').read_text())
assert execution['status']=='complete' and execution['files_verified_before']==execution['files_verified_after']==len(entries)
assert execution['previous_private_assets_unchanged']
examples=[r for i in (0,1) for r in read_rows(base/f'collect-{i}/examples.jsonl')]
controls=aligned_controls(previous_rows(root,'examples'),previous_rows(root,'sources'))
assert len(examples)==len(controls)==80
idx={(r['request_id'],r['source_round']):r for r in examples}
assert len(idx)==80
for point in controls:
    new=idx[point['request_id'],point['source_round']]
    assert all(new[k]==v for k,v in point.items())
    assert new['context_tokens']==len(new['prefix_ids'])-1 and 1<=len(new['teacher_tokens'])<=15
train=json.loads((base/'train/provenance.json').read_text())
curve=list(read_rows(base/'train/metrics.jsonl'))
assert [r['update'] for r in curve]==[0,32,64,96,128]
selected=min(curve[1:],key=lambda r:r['calibration']['loss'])
assert selected['update']==train['selected_step'] and selected['calibration']['loss']==train['selected_calibration_loss']
assert train['teacher_parameter_versions_unchanged'] and train['teacher_gradients_absent'] and train['draft_base_versions_unchanged']
oldtrain=json.loads((root/'results/asd-adaptation/train-asd-prefix/provenance.json').read_text())
assert train['train_order_keys']==oldtrain['train_order_keys'] and train['updates']==128 and train['accumulation']==2
assert train['train_states']==64 and train['train_requests']==32 and train['calibration_states']==32
oldcal=[r for r in previous_rows(root,'examples') if r['split']=='calibration']
assert train['calibration_identity_sha256']==sha(json.dumps(oldcal,sort_keys=True).encode())
refs={r['request_id']:r for i in (0,1) for r in read_rows(base/f'eval-{i}/references.jsonl')}
rows=[r for i in (0,1) for r in read_rows(base/f'eval-{i}/rows.jsonl')]
assert len(rows)==960 and len(refs)==24 and set(refs)=={r['id'] for r in records(root)}
assert not {r['request_id'] for r in previous_rows(root,'examples')} & set(refs)
for row in rows:
    ref=refs[row['request_id']]
    assert (row['prompt_sha256'],row['snapshot_id'])==(ref['prompt_sha256'],ref['snapshot_id'])
    if row['policy']=='DFlash':
        assert row['token_ids']==ref['token_ids'][:row['max_new_tokens']]
        if row['stage']=='natural':assert (row['text'],row['finish_reason'])==(ref['text'],ref['finish_reason'])
    if row['variant'] in ADAPTER_SHA:assert row['adapter_sha256']==ADAPTER_SHA[row['variant']]
    if row['variant']=='ordinary-position':assert row['adapter_sha256']==train['adapter_sha256']
    for step in row['rounds']:
        frontier=row['prompt_tokens']+step['generated_before']-1
        assert step['draft_cache_before']+step['pending_before']==frontier
        assert step['target_cache_after']==frontier+step['advance']
analysis=analyze(rows)
assert analysis['all_safe']
zipfile_path=root/f'artifacts/{STUDY}-results.zip'
with zipfile.ZipFile(zipfile_path) as archive:
    assert archive.testzip() is None
    for name in archive.namelist():assert (base/name).read_bytes()==archive.read(name),name
references={r:subprocess.check_output(['git','rev-parse',r],cwd=root).decode().strip() for r in ('main','codex/infra-interview-v0.3','infra-v0.3-interview^{}')}
assert set(references.values())=={'27e36199f8a65acdb9a9c42ccb744ddfa9a9402b'}
result=dict(status='complete',execution_commit=manifest['commit'],git_blobs=len(entries),source_sha256=manifest['source_sha256'],frozen_files=frozen,
            reproduction_fix_commit=reproduction['commit'],reproduction_fix_files=reproduction['files'],
            previous_result_files_verified=protected,legacy_crlf=legacy,matched_states=80,train_states=64,formal_rows=len(rows),ar_references=len(refs),
            all_safe=True,frontiers_valid=True,training_evaluation_ids_disjoint=True,calibration_identical=True,previous_adapters_identical=True,
            zip_bytes=zipfile_path.stat().st_size,zip_sha256=sha(zipfile_path.read_bytes()),infra_refs=references)
(base/'postrun-audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result))
