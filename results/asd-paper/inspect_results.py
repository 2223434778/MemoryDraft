"""Independent public-payload, Git-source, data, execution and ledger checks."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from memorydraft.evaluation.asd_joint_tree import frozen_config
from memorydraft.evaluation.asd_paper import ARMS,COUNTS,aggregate,expanded_ids,load_records
from scripts.write_asd_paper_report import judge_stats,read_rows

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/asd-paper'


def verify_source(manifest):
    assert hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()==manifest['source_sha256']
    raw=subprocess.check_output(['git','ls-tree','-r','-z',manifest['commit']],cwd=ROOT)
    entries=[]
    for entry in raw.split(b'\0'):
        if entry:
            metadata,name=entry.split(b'\t',1);entries.append((name.decode(),metadata.split()[2]))
    assert set(manifest['files'])=={n for n,_ in entries}
    raw=subprocess.run(['git','cat-file','--batch'],cwd=ROOT,input=b''.join(h+b'\n' for _,h in entries),
        stdout=subprocess.PIPE,check=True).stdout
    offset=0
    for name,oid in entries:
        end=raw.index(b'\n',offset);header=raw[offset:end].split();assert header[:2]==[oid,b'blob']
        size=int(header[2]);content=raw[end+1:end+1+size]
        assert hashlib.sha256(content).hexdigest()==manifest['files'][name],name
        offset=end+size+2
    return len(entries)


def ledger(row):
    assert row['approximate']==(row['arm'] not in ('DFlash','BudgetTree'))
    if row['asd']:
        assert row['asd']['budget']==8 and row['asd']['blocks']==len(row['rounds'])
        assert 0<=row['asd']['spent']<=8
        assert abs(sum(b['regret'] for b in row['rounds'])-row['asd']['spent'])<1e-9
        assert sum(b['relaxed_tokens'] for b in row['rounds'])==row['asd']['relaxed_tokens']
    for b in row['rounds']:
        assert b['advance']==b['accepted']+1 and 0<=b['accepted']<=15
        if row['asd']:assert 0<=b['relaxed_tokens']<=2 and b['regret']>=0
        if 'tree_nodes' in b:
            assert 1<=b['tree_nodes']<=32
            assert b['tree_path'][0]==0 and len(b['tree_path'])==b['advance']
            assert len(set(b['tree_path']))==len(b['tree_path']) and max(b['tree_path'])<b['tree_nodes']


def main():
    publication=json.loads((OUT/'publication.json').read_bytes())
    for name,info in publication['files'].items():
        raw=(OUT/name).read_bytes()
        assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256'],name
    manifest=json.loads((OUT/'ASD_PAPER_SOURCE_MANIFEST.json').read_bytes());files=verify_source(manifest)
    directory=ROOT/'data/research/asd-paper'
    data_manifest=json.loads((directory/'manifest.json').read_bytes())
    excluded=json.loads((directory/'exclusions.json').read_bytes())
    assert hashlib.sha256((directory/'exclusions.json').read_bytes()).hexdigest()==data_manifest['excluded_files_sha256']
    for name,digest in excluded['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    pool=load_records(ROOT)
    assert not {r['id'] for r in pool}&set(excluded['excluded_ids'])
    assert len(pool)==241 and len(expanded_ids(pool))==249
    rows,refs=read_rows(OUT,'rows'),read_rows(OUT,'references');rollout=read_rows(OUT,'rollout')
    assert len(rows)==4980 and len(refs)==249 and len(rollout)==40
    rebuilt=json.loads(json.dumps(aggregate(rows,refs,pool)));saved=json.loads((OUT/'analysis.json').read_bytes())
    assert all(saved[k]==v for k,v in rebuilt.items())
    assert saved['local_judge']==judge_stats(OUT)
    for row in [*rows,*rollout]:
        ledger(row)
        if row['stage']=='fixed':assert len(row['token_ids'])==256 and row['stop_policy']=='ignore-eos'
    assert max(r['peak_gpu_bytes'] for r in [*rows,*refs,*rollout])<=46*2**30
    primary={(r['request_id'],r['arm']):r for r in rows if r['stage']=='natural' and r['repeat']==0}
    for row in rollout:
        first=primary[row['conversation_id']+':turn1',row['arm']]
        assert (row['first_text'],row['first_token_ids'],row['first_latency_seconds'])==(first['text'],first['token_ids'],first['latency_seconds'])
        assert row['context_policy']=='own realized first-turn history'
    execution=json.loads((OUT/'execution-verification.json').read_bytes())
    assert execution['status']=='complete' and execution['commit']==manifest['commit']
    assert execution['files_verified_before']==execution['files_verified_after']==files
    for shard,expected_count in ((0,2500),(1,2480)):
        part=OUT/f'part-{shard}';info=json.loads((part/'provenance.json').read_bytes())
        assert info['status']=='complete' and info['original_unchanged'] and info['count']==expected_count
        assert info['arms']==list(ARMS) and info['canonical_failed_ids']==[]
        assert info['peak_gpu_bytes']<=46*2**30 and info['cache_bytes']<=4*2**30
        assert info['compiler']['graph_count']==info['compiler_after_warmup']['graph_count']
        assert info['compiler']['float32_matmul_precision']=='highest' and not info['compiler']['allow_tf32']
        source=info['runtime']['source']
        assert source['sha256']==hashlib.sha256(json.dumps(source['files'],sort_keys=True).encode()).hexdigest()
        assert all(manifest['files']['memorydraft/'+name]==digest for name,digest in source['files'].items())
        _,source=frozen_config(ROOT,shard);assert json.loads((part/'cost-table.json').read_bytes())==source
        assert json.loads((part/'math-complete.json').read_bytes())==dict(count=960,status='complete',parameter_changes=0)
        assert len(read_rows(OUT,'warmup'))==18
    completion=json.loads((OUT/'completion.json').read_bytes());deployment=json.loads((OUT/'deployment.json').read_bytes())
    assert completion['verified_canonical_files']==files
    assert completion['source_sha256']==deployment['source_sha256']==manifest['source_sha256']
    judge=json.loads((OUT/'local-judge-provenance.json').read_bytes())
    assert judge['status']=='complete' and judge['rows']==192 and judge['max_input_tokens']==4096
    assert judge['peak_gpu_bytes']<=46*2**30 and judge['cache_bytes']<=4*2**30
    protected='27e36199f8a65acdb9a9c42ccb744ddfa9a9402b'
    for ref in ('main','codex/infra-interview-v0.3','infra-v0.3-interview^{}'):
        assert subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip()==protected
    suite=ET.parse(OUT/'tests-server.xml').getroot().find('testsuite')
    assert suite.attrib['failures']==suite.attrib['errors']=='0'
    for name,info in json.loads((OUT/'local-checks/manifest.json').read_bytes()).items():
        raw=(OUT/'local-checks'/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==info['sha256'] and len(raw)==info['bytes']
    result=dict(source_commit=manifest['commit'],canonical_files=files,public_payloads=len(publication['files']),
        timed=len(rows),ar=len(refs),realized_second_turns=len(rollout),old_files=len(excluded['files']),
        selected_unused_source_tasks=len(pool),all_source_data_input_repeat_strict_budget_resource_publication_checks=True,
        allocation=saved['allocation_audit'],protected_refs=True)
    (OUT/'audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(result))


if __name__=='__main__':main()
