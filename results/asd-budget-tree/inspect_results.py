"""Post-run publication, source, ledger and protected-ref audit. No GPU needed."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from memorydraft.evaluation.asd_tree import ARMS, REPEATS, aggregate, records
from scripts.write_asd_tree_report import read_rows

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/asd-budget-tree'


def main():
    publication=json.loads((OUT/'publication.json').read_text())
    for name,info in publication['files'].items():
        data=(OUT/name).read_bytes()
        assert len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256'],name
    manifest=json.loads((OUT/'ASD_TREE_SOURCE_MANIFEST.json').read_text())
    assert hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()==manifest['source_sha256']
    commit=manifest['commit']
    names=list(manifest['files'])
    objects=subprocess.check_output(['git','ls-tree','-r','-z',commit],cwd=ROOT)
    entries=[]
    for item in objects.split(b'\0'):
        if item:
            meta,name=item.split(b'\t',1)
            entries.append((name.decode(),meta.split()[2]))
    assert set(names)=={n for n,_ in entries}
    data=subprocess.run(['git','cat-file','--batch'],cwd=ROOT,input=b''.join(h+b'\n' for _,h in entries),stdout=subprocess.PIPE,check=True).stdout
    offset=0
    for name,oid in entries:
        end=data.index(b'\n',offset)
        header=data[offset:end].split()
        assert header[:2]==[oid,b'blob']
        size=int(header[2])
        blob=data[end+1:end+1+size]
        assert hashlib.sha256(blob).hexdigest()==manifest['files'][name],name
        offset=end+size+2
    rows,refs=read_rows(OUT,'rows'),read_rows(OUT,'references')
    result=json.loads(json.dumps(aggregate(rows,refs)))
    saved=json.loads((OUT/'analysis.json').read_text())
    assert all(result[k]==saved[k] for k in result)
    assert len(rows)==960 and len(refs)==48 and {r['request_id'] for r in refs}=={r['id'] for r in records(ROOT)}
    for row in rows:
        assert row['approximate']==(row['arm'] in ('ASD','BudgetTree-ASD'))
        if row['asd']:
            assert row['asd']['budget']==8 and row['asd']['blocks']==len(row['rounds'])
            assert abs(sum(r['regret'] for r in row['rounds'])-row['asd']['spent'])<1e-10
            assert sum(r['relaxed_tokens'] for r in row['rounds'])==row['asd']['relaxed_tokens']
        for r in row['rounds']:
            assert r['advance']==r['accepted']+1
            if row['asd']:assert 0<=r['relaxed_tokens']<=2 and r['regret']>=0
            if 'tree_nodes' in r:
                assert 1<=r['tree_nodes']<=32
                assert r['tree_path'][0]==0 and len(set(r['tree_path']))==len(r['tree_path'])==r['advance']
                assert max(r['tree_path'])<r['tree_nodes'] and r['accepted']<=15
    execution=json.loads((OUT/'execution-verification.json').read_text())
    assert execution['status']=='complete' and execution['commit']==commit
    assert execution['files_verified_before']==execution['files_verified_after']==len(manifest['files'])
    for shard in (0,1):
        part=OUT/f'part-{shard}'
        assert json.loads((part/'pilot-check.json').read_text())['status']=='passed'
        p=json.loads((part/'provenance.json').read_text())
        assert p['status']=='complete' and p['original_unchanged'] and p['count']==480
        assert p['peak_gpu_bytes']<=46*2**30 and p['cache_bytes']<=4*2**30
        assert p['compiler']['graph_count']==p['compiler_after_warmup']['graph_count']
    completion=json.loads((OUT/'completion.json').read_text())
    deployment=json.loads((OUT/'deployment.json').read_text())
    assert completion['source_sha256']==deployment['source_sha256']==manifest['source_sha256']
    assert deployment['commit']==commit and completion['verified_canonical_files']==len(names)
    failure=OUT/'startup-eos'
    for name,info in json.loads((failure/'publication.json').read_text())['files'].items():
        content=(failure/name).read_bytes()
        assert len(content)==info['bytes'] and hashlib.sha256(content).hexdigest()==info['sha256']
    protected='27e36199f8a65acdb9a9c42ccb744ddfa9a9402b'
    for ref in ('main','codex/infra-interview-v0.3','infra-v0.3-interview^{}'):
        assert subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip()==protected
    xml=ET.parse(OUT/'tests-server.xml').getroot().find('testsuite')
    assert xml.attrib['failures']==xml.attrib['errors']=='0'
    for name,info in json.loads((OUT/'local-checks/manifest.json').read_text()).items():
        content=(OUT/'local-checks'/name).read_bytes()
        assert hashlib.sha256(content).hexdigest()==info['sha256'] and len(content)==info['bytes']
    print(json.dumps(dict(source_commit=commit,canonical_files=len(names),public_payloads=len(publication['files']),
        timed_generations=len(rows),ar_generations=len(refs),all_publication_source_statistics_ledger_resource_checks=True,
        server_tests=xml.attrib['tests'],protected_refs=True)))


if __name__=='__main__':main()
