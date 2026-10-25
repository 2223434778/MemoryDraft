"""Resume complete request groups; keep numerical geometry differences visible."""
import hashlib
import json
from pathlib import Path

from memorydraft.evaluation.asd_paper import expanded_ids
from memorydraft.evaluation.asd_paper_extension import ARMS,CONTROLS,frontier_ids,load_records

ORIGINAL_COMMIT='6d89480c52e0d89714778668c25ef7027a9f0ac8'
ORIGINAL_SOURCE='d39de7b926d6d49ca3750374ddfd52b012e12ee9e92bdfaa13623a41d3b81029'


def identity_difference(left,right,comparison):
    a,b=left['token_ids'],right['token_ids']
    if a==b:return None
    index=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
    digest=lambda ids:hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    return dict(request_id=left['request_id'],arm=left.get('arm'),right_arm=right.get('arm','AR'),stage=left.get('stage'),
        repeat=left.get('repeat'),role=left.get('role'),comparison=comparison,index=index,
        left_id=a[index] if index<len(a) else None,right_id=b[index] if index<len(b) else None,
        left_tokens=len(a),right_tokens=len(b),left_sha256=digest(a),right_sha256=digest(b),
        interpretation='Identity failure recorded; numerical cause requires separate diagnosis')


def completed_payloads(directory,selected,completed,*,frontier=None):
    """Do not import an unfinished request, even if some of its rows exist."""
    directory=Path(directory)
    if type(completed) is not int or not 0<=completed<=len(selected):raise ValueError('Invalid completed prefix')
    tasks=selected[:completed];wanted=set(expanded_ids(tasks));payloads={};digests={}
    for name in ('rows','references','frontier','rollout'):
        path=directory/(name+'.jsonl');data=path.read_bytes() if path.exists() else b''
        digests[name]=hashlib.sha256(data).hexdigest();kept=[];rows=[]
        for raw in data.splitlines(keepends=True):
            row=json.loads(raw)
            if row['request_id'] in wanted:kept.append(raw);rows.append(row)
        payloads[name]=(b''.join(kept),rows)
    rows=payloads['rows'][1]
    def key(row):return row['request_id'],row['stage'],row['arm'],row['repeat'],row.get('role')
    expected={(i,s,a,r,None) for i in wanted for s,n in (('natural',2),('prefix',1)) for a in ARMS for r in range(n)}
    expected|={(i,'fixed',a,r,role) for i in wanted for a in ARMS
        for r,role in (enumerate(('before','after')) if a in ('DFlash','BudgetTree') else [(0,'candidate')])}
    if len(rows)!=len(expected) or {key(r) for r in rows}!=expected:raise ValueError('Incomplete resumed main group')
    refs=payloads['references'][1]
    if len(refs)!=len(wanted) or {r['request_id'] for r in refs}!=wanted:raise ValueError('Incomplete resumed references')
    frontier=frontier_ids(selected) if frontier is None else frontier
    frontier_wanted=set(expanded_ids([t for t in tasks if t['id'] in frontier]))
    controls=payloads['frontier'][1]
    control_keys={(i,c,s,a) for i in frontier_wanted for c in CONTROLS for s in ('natural','fixed') for a in ('ASD','BudgetTree-ASD')}
    if len(controls)!=len(control_keys) or {(r['request_id'],r['control'],r['stage'],r['arm']) for r in controls}!=control_keys:
        raise ValueError('Incomplete resumed control group')
    rollout=payloads['rollout'][1];dialogs={t['id'] for t in tasks if t['benchmark']=='mtbench'}
    if len(rollout)!=len(dialogs)*len(ARMS) or {(r['conversation_id'],r['arm']) for r in rollout}!={(i,a) for i in dialogs for a in ARMS}:
        raise ValueError('Incomplete resumed dialogue group')
    return payloads,digests


def resume_completed(root,original,out,selected,shard):
    root,original,out=map(Path,(root,original,out))
    manifest=json.loads((original/'ASD_PAPER_EXTENSION_SOURCE_MANIFEST.json').read_bytes())
    if manifest['commit']!=ORIGINAL_COMMIT or manifest['source_sha256']!=ORIGINAL_SOURCE:
        raise ValueError('Unexpected original source')
    for name,digest in manifest['files'].items():
        if hashlib.sha256((original/name).read_bytes()).hexdigest()!=digest:raise ValueError('Original source changed: '+name)
        if name.startswith(('memorydraft/backends/','memorydraft/runtime/')) or name in (
            'configs/runtime-dflash-rmsnorm.json','scripts/run_asd_tree.py','scripts/run_asd_joint_tree.py',
            'memorydraft/evaluation/asd_joint_tree.py','memorydraft/evaluation/asd_paper_extension.py'):
            # The extension aggregation can change, but generation/data helpers cannot.
            if name.endswith('asd_paper_extension.py'):continue
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Generation source changed: '+name)
    for name in ('memorydraft/evaluation/asd_paper.py','memorydraft/evaluation/asd_paper_extension.py'):
        if (root/name).read_bytes().split(b'\ndef aggregate(')[0]!=(original/name).read_bytes().split(b'\ndef aggregate(')[0]:
            raise ValueError('Generation/data helper changed: '+name)
    previous=original/f'results/raw/asd-paper-extension/part-{shard}'
    info=json.loads((previous/'provenance.json').read_bytes())
    if info['status']!='failed' or not info['original_unchanged'] or info['task_ids']!=[r['id'] for r in selected]:
        raise ValueError('Unexpected failed worker prefix')
    payloads,digests=completed_payloads(previous,selected,info['completed_tasks'],frontier=frontier_ids(load_records(root)))
    for name,(data,_) in payloads.items():
        if data:(out/(name+'.jsonl')).write_bytes(data)
    for name in ('count','frontier_count','rollout_count'):
        field={'count':'rows','frontier_count':'frontier','rollout_count':'rollout'}[name]
        if info.get(name,0)!=len(payloads[field][1]):raise ValueError('Resumed count mismatch')
    proof=dict(original_root=str(original),original_commit=manifest['commit'],original_source_sha256=manifest['source_sha256'],
        original_provenance=info,completed_tasks=info['completed_tasks'],source_files_verified=len(manifest['files']),
        original_payload_sha256=digests,selection='Whole completed prefix only; incomplete rows retained in original failure archive')
    (out/'reuse.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof
