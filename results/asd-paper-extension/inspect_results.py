"""Independent checks of immutable inputs, raw outputs, controls and resources."""
from pathlib import Path
import hashlib,json,math,subprocess
import gzip
import xml.etree.ElementTree as ET

from memorydraft.evaluation.asd_paper import expanded_ids
from memorydraft.backends.asd import ASDConfig
from memorydraft.evaluation.asd_paper_extension import (ARMS,CONTROLS,aggregate,frontier_ids,load_records,
    normalize_fixed,read_rows,validate_policy)
from memorydraft.evaluation.asd_paper_metrics import message_sha

ROOT=Path.cwd();OUT=ROOT/'results/asd-paper-extension'


def git_source(manifest):
    assert hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()==manifest['source_sha256']
    entries=[]
    for raw in subprocess.check_output(['git','ls-tree','-r','-z',manifest['commit']]).split(b'\0'):
        if raw:
            info,name=raw.split(b'\t',1);entries.append((name.decode(),info.split()[2]))
    assert {name for name,_ in entries}==set(manifest['files'])
    raw=subprocess.run(['git','cat-file','--batch'],input=b''.join(oid+b'\n' for _,oid in entries),stdout=subprocess.PIPE,check=True).stdout
    offset=0
    for name,oid in entries:
        end=raw.index(b'\n',offset);header=raw[offset:end].split();assert header[:2]==[oid,b'blob']
        size=int(header[2]);data=raw[end+1:end+size+1];offset=end+size+2
        assert hashlib.sha256(data).hexdigest()==manifest['files'][name],name
    assert offset==len(raw)
    return len(entries)


def emitted_positions(row):
    total=len(row['token_ids']);draft=set();target={0} if total else set()
    for block in row['rounds']:
        start=block['generated_before'];accepted=block['accepted']
        positions=set(range(start,min(total,start+accepted)))
        assert not draft&positions;draft|=positions
        recovery=start+accepted
        if recovery<total:target.add(recovery)
    assert not draft&target and draft|target==set(range(total)),(row['request_id'],row['arm'],row['stage'])
    assert len(draft)==sum(min(b['accepted'],max(0,total-b['generated_before'])) for b in row['rounds'])


def close(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(close(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b


def main():
    publication=json.loads((OUT/'publication.json').read_bytes())
    for name,item in publication['files'].items():
        data=(OUT/name).read_bytes()
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],name
    manifest=json.loads((OUT/'ASD_PAPER_EXTENSION_CONTINUATION_SOURCE_MANIFEST.json').read_bytes());source_count=git_source(manifest)
    original=json.loads((OUT/'interruption/ASD_PAPER_EXTENSION_SOURCE_MANIFEST.json').read_bytes())
    original_source_count=git_source(original)
    verify=json.loads((OUT/'execution-verification.json').read_bytes())
    assert verify['status']=='complete' and verify['files_verified_before']==verify['files_verified_after']==source_count
    exclusion=json.loads((ROOT/'data/research/asd-paper-extension/exclusions.json').read_bytes())
    for name,digest in exclusion['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    records=load_records(ROOT);raw=read_rows(OUT,'rows');refs=read_rows(OUT,'references');frontier=read_rows(OUT,'frontier')
    rebuilt=aggregate(raw,refs,records,frontier,strict_token_identity=False);saved=json.loads((OUT/'analysis.json').read_bytes())
    assert all(close(saved[k],v) for k,v in rebuilt.items())
    normalized,_=normalize_fixed(raw);by_id={r['request_id']:r for r in refs};audited=0;observed=[]
    for row in normalized:
        if row['repeat']!=0:continue
        if row['asd'] is not None:validate_policy(row,ASDConfig(budget=8))
        emitted_positions(row);audited+=1
        if row['stage']!='fixed' and row['arm'] in ('DFlash','BudgetTree'):
            limit=4096 if row['stage']=='natural' else 128
            from memorydraft.evaluation.asd_extension_resume import identity_difference
            difference=identity_difference(row,dict(by_id[row['request_id']],token_ids=by_id[row['request_id']]['token_ids'][:limit]),'strict-vs-AR')
            if difference:observed.append(difference)
    for row in frontier:emitted_positions(row)
    resources=[]
    for shard in (0,1):
        p=json.loads((OUT/f'part-{shard}/provenance.json').read_bytes());resources.append(p)
        assert p['status']=='complete' and p['original_unchanged'] and p['count']==4770
        assert p['frontier_count']==320 and p['rollout_count']==32 and p['completed_tasks']==257
        assert p['reused_completed_tasks']==(83 if shard==0 else 89) and p['reuse_payloads_unchanged']
        reuse=json.loads((OUT/f'part-{shard}/reuse.json').read_bytes())
        assert reuse['original_commit']==original['commit'] and reuse['original_source_sha256']==original['source_sha256']
        assert reuse['source_files_verified']==original_source_count and reuse['completed_tasks']==p['reused_completed_tasks']
        oldp=json.loads((OUT/f'interruption/part-{shard}/provenance.json').read_bytes())
        assert reuse['original_provenance']==oldp and oldp['status']=='failed' and oldp['original_unchanged']
        from memorydraft.evaluation.asd_paper import BENCHMARKS
        selected=[r for b in BENCHMARKS for i,r in enumerate(r for r in records if r['benchmark']==b) if i%2==shard]
        wanted=set(expanded_ids(selected[:p['reused_completed_tasks']]))
        for name,digest in reuse['original_payload_sha256'].items():
            oldpath=OUT/f'interruption/part-{shard}/{name}.jsonl.gz'
            data=gzip.decompress(oldpath.read_bytes()) if oldpath.exists() else b''
            assert hashlib.sha256(data).hexdigest()==digest
            expected=b''.join(line for line in data.splitlines(keepends=True) if json.loads(line)['request_id'] in wanted)
            path=OUT/f'part-{shard}/{name}.jsonl.gz'
            new=gzip.decompress(path.read_bytes()) if path.exists() else b''
            kept=b''.join(line for line in new.splitlines(keepends=True) if json.loads(line)['request_id'] in wanted)
            assert kept==expected,(shard,name)
        assert p['cost_table_read_seconds']>=0 and p['load_seconds']>0
        assert p['peak_gpu_bytes']<=46*2**30 and p['cache_bytes']<=4*2**30
        assert p['compiler']['graph_count']==p['compiler_after_warmup']['graph_count']<=8
        assert p['compiler']['float32_matmul_precision']=='highest' and not p['compiler']['allow_tf32']
        engine=p['runtime']['engine']
        assert engine['dtype']=='torch.float32' and engine['gpu']=='NVIDIA RTX A6000'
        assert engine['execution_profile']=='rmsnorm-inductor' and engine['draft_adapter'] is None and engine['prefix_cache'] is None
    protected='27e36199f8a65acdb9a9c42ccb744ddfa9a9402b'
    for ref in ('main','codex/infra-interview-v0.3','infra-v0.3-interview^{}'):
        assert subprocess.check_output(['git','rev-parse',ref],text=True).strip()==protected
    tree=ET.parse(OUT/'tests-server.xml');suites=tree.getroot().findall('testsuite')
    assert sum(int(s.attrib['tests']) for s in suites)==181
    assert all(int(s.attrib['failures'])==int(s.attrib['errors'])==0 for s in suites)
    reward=OUT/'reward/provenance.json'
    if reward.exists():
        from memorydraft.evaluation.asd_paper_extension import open_units
        info=json.loads(reward.read_bytes());assert info['status']=='complete' and info['rows']==320
        assert info['controls_passed']>=7 and info['repeat_max_abs_error']<=1e-5 and info['peak_gpu_bytes']<=22*2**30
        expected={(u['request_id'],arm):message_sha(messages) for u in open_units(ROOT,OUT) for arm,messages in u['answers'].items()}
        rows=[json.loads(line) for line in gzip.decompress((OUT/'reward/rows.jsonl.gz').read_bytes()).decode().splitlines()]
        assert len(rows)==320 and {(r['request_id'],r['arm']) for r in rows}==set(expected)
        assert all(r['messages_sha256']==expected[r['request_id'],r['arm']] for r in rows)
        execution=json.loads((OUT/'reward/execution.json').read_bytes())
        assert execution['status']=='complete' and execution['source_files_verified_before']==execution['source_files_verified_after']==original_source_count
    for name,digest in original['files'].items():
        if name.startswith(('memorydraft/backends/','memorydraft/runtime/')) or name in (
            'configs/runtime-dflash-rmsnorm.json','scripts/run_asd_tree.py','scripts/run_asd_joint_tree.py','memorydraft/evaluation/asd_joint_tree.py'):
            assert manifest['files'][name]==digest,('generation core changed',name)
    diagnostic=json.loads((OUT/'strict-diagnosis/provenance.json').read_bytes())
    assert diagnostic['status']=='complete' and diagnostic['ar_matches_original']['equal']
    assert diagnostic['numeric']['strict_repeat_matches']['equal'] and diagnostic['numeric']['shortened_ar_matches_full']
    geometry=[]
    for shard in (0,1):
        path=OUT/f'part-{shard}/geometry-differences.jsonl.gz'
        if path.exists():geometry.extend(json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines())
    for value in observed:
        assert any(all(value[k]==g.get(k) for k in ('request_id','arm','stage','repeat','comparison','left_sha256','right_sha256')) for g in geometry)
    numeric=[];covered=set()
    for shard in (0,1):
        directory=OUT/f'geometry-diagnosis/part-{shard}'
        info=json.loads((directory/'provenance.json').read_bytes())
        assert info['status']=='complete' and info['source_commit']==manifest['commit']
        assert info['files_verified_before']==info['files_verified_after']==source_count
        assert info['helper_sha256']==hashlib.sha256((OUT/'geometry-diagnosis/helpers/diagnose_extension_geometry.py').read_bytes()).hexdigest()
        assert info['planned_comparisons']==len(info['comparisons'])
        for number,value in enumerate(info['comparisons']):
            g=value['observation'];assert g in geometry
            covered.add((g['request_id'],g['arm'],g['stage'],g['comparison']))
            assert value['left_replay_matches_original'] and value['right_replay_matches_original_prefix'] and value['same_confirmed_prefix']
            assert value['left_pass']['argmax_id']==g['left_id'] and value['right_pass']['argmax_id']==g['right_id']
            left=json.loads((directory/f'{number:03d}-left.json').read_bytes())
            assert hashlib.sha256(json.dumps(left['token_ids'],separators=(',',':')).encode()).hexdigest()==value['left_output_ids_sha256']
            numeric.append(value)
    assert covered=={(g['request_id'],g['arm'],g['stage'],g['comparison']) for g in geometry if g['comparison']!='different-own-history-inputs'}
    post=json.loads((OUT/'auxiliary/POSTPROCESS.json').read_bytes())
    assert post['status']=='complete' and post['files_verified_before']==post['files_verified_after']==source_count
    result=dict(source_commit=manifest['commit'],source_files=source_count,original_payloads=len(publication['files']),
        old_inputs_and_payloads_verified=len(exclusion['files']),primary_raw=len(raw),references=len(refs),frontier=len(frontier),
        emitted_position_trajectories=audited+len(frontier),resources_and_old_infra_preserved=True,reconstruction_numerically_equal=True,
        original_source_commit=original['commit'],original_source_files=original_source_count,reused_complete_sources=172,
        cross_geometry_observations=len(geometry),strict_vs_ar_first_repeat_differences=len(observed),
        strict_AR_bit_identity_claim=False,actual_pass_diagnosed_comparisons=len(numeric),
        maximum_first_difference_logit_delta=max((v['max_abs_difference'] for v in numeric),default=0.),
        numerical_diagnosis_scope='Actual-pass first-divergence replay for every recorded token-identity case; no whole-sequence bit-identity claim')
    (OUT/'audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))


if __name__=='__main__':main()
