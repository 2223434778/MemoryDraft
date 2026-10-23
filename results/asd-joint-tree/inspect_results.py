"""Independent post-run checks for the frozen second-stage comparison."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from memorydraft.evaluation.asd_joint_tree import ARMS, LegacyCheck, aggregate, frozen_config
from memorydraft.evaluation.asd_tree import records
from scripts.write_asd_tree_report import read_rows

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/asd-joint-tree'


def main():
    publication=json.loads((OUT/'publication.json').read_text())
    for name,info in publication['files'].items():
        data=(OUT/name).read_bytes()
        assert len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256'],name
    manifest=json.loads((OUT/'ASD_JOINT_TREE_SOURCE_MANIFEST.json').read_text())
    assert hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()==manifest['source_sha256']
    commit=manifest['commit']
    raw=subprocess.check_output(['git','ls-tree','-r','-z',commit],cwd=ROOT)
    entries=[]
    for entry in raw.split(b'\0'):
        if entry:
            meta,name=entry.split(b'\t',1)
            entries.append((name.decode(),meta.split()[2]))
    assert set(manifest['files'])=={n for n,_ in entries}
    data=subprocess.run(['git','cat-file','--batch'],cwd=ROOT,input=b''.join(h+b'\n' for _,h in entries),stdout=subprocess.PIPE,check=True).stdout
    offset=0
    for name,oid in entries:
        end=data.index(b'\n',offset)
        header=data[offset:end].split()
        assert header[:2]==[oid,b'blob']
        size=int(header[2]);blob=data[end+1:end+1+size]
        assert hashlib.sha256(blob).hexdigest()==manifest['files'][name],name
        offset=end+size+2
    rows,refs=read_rows(OUT,'rows'),read_rows(OUT,'references')
    saved=json.loads((OUT/'analysis.json').read_text())
    rebuilt=json.loads(json.dumps(aggregate(rows,refs)))
    assert all(rebuilt[k]==saved[k] for k in rebuilt)
    assert len(rows)==1440 and len(refs)==48 and {r['request_id'] for r in refs}=={r['id'] for r in records(ROOT)}
    legacy=LegacyCheck(ROOT)
    for row in rows:
        legacy(row)
        assert row['approximate']==(row['arm'] not in ('DFlash','BudgetTree'))
        if row['asd']:
            assert row['asd']['budget']==8 and row['asd']['blocks']==len(row['rounds'])
            assert abs(sum(r['regret'] for r in row['rounds'])-row['asd']['spent'])<1e-10
            assert sum(r['relaxed_tokens'] for r in row['rounds'])==row['asd']['relaxed_tokens']
        for r in row['rounds']:
            assert r['advance']==r['accepted']+1
            if row['asd']:assert 0<=r['relaxed_tokens']<=2 and r['regret']>=0
            if 'tree_nodes' in r:
                assert 1<=r['tree_nodes']<=32 and len(r['tree_path'])==r['advance']
                assert r['tree_path'][0]==0 and len(set(r['tree_path']))==len(r['tree_path'])
                assert max(r['tree_path'])<r['tree_nodes'] and r['accepted']<=15
    assert legacy.checked==960
    execution=json.loads((OUT/'execution-verification.json').read_text())
    assert execution['status']=='complete' and execution['commit']==commit
    assert execution['files_verified_before']==execution['files_verified_after']==len(manifest['files'])
    for shard in (0,1):
        part=OUT/f'part-{shard}'
        assert json.loads((part/'pilot-check.json').read_text())['status']=='passed'
        p=json.loads((part/'provenance.json').read_text())
        assert p['status']=='complete' and p['original_unchanged'] and p['count']==720
        assert p['legacy_checked_rows']==480
        assert p['peak_gpu_bytes']<=46*2**30 and p['cache_bytes']<=4*2**30
        assert p['compiler']['graph_count']==p['compiler_after_warmup']['graph_count']
        assert p['compiler']['float32_matmul_precision']=='highest' and not p['compiler']['allow_tf32']
        config,source=frozen_config(ROOT,shard)
        actual=json.loads((part/'cost-table.json').read_text())
        assert all(actual[k]==v for k,v in source.items())
        assert p['tree_config']['costs']==json.loads((ROOT/f'results/asd-budget-tree/part-{shard}/calibration.json').read_text())['costs']
    completion=json.loads((OUT/'completion.json').read_text())
    deployment=json.loads((OUT/'deployment.json').read_text())
    assert completion['source_sha256']==deployment['source_sha256']==manifest['source_sha256']
    assert deployment['commit']==commit and completion['verified_canonical_files']==len(manifest['files'])
    protected='27e36199f8a65acdb9a9c42ccb744ddfa9a9402b'
    for ref in ('main','codex/infra-interview-v0.3','infra-v0.3-interview^{}'):
        assert subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip()==protected
    xml=ET.parse(OUT/'tests-server.xml').getroot().find('testsuite')
    assert xml.attrib['failures']==xml.attrib['errors']=='0'
    for name,info in json.loads((OUT/'local-checks/manifest.json').read_text()).items():
        data=(OUT/'local-checks'/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==info['sha256'] and len(data)==info['bytes']
        suite=ET.fromstring(data).find('testsuite')
        assert (int(suite.attrib['failures']),int(suite.attrib['errors']))==(info['failures'],info['errors'])
    print(json.dumps(dict(source_commit=commit,canonical_files=len(manifest['files']),
        public_payloads=len(publication['files']),timed_generations=len(rows),ar_generations=len(refs),
        legacy_checked_rows=legacy.checked,allocation=saved['allocation_audit'],
        all_publication_source_statistics_ledger_resource_checks=True,server_tests=xml.attrib['tests'],protected_refs=True)))


if __name__=='__main__':main()
