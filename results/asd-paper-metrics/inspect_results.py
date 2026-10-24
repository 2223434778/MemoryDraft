"""Independent source, token accounting, fixed-AR and reward provenance audit."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from memorydraft.evaluation.asd_paper_metrics import aggregate,inputs,message_sha,open_units,read_rows
from scripts.write_asd_paper_metrics_report import read,rewards

ROOT=Path.cwd()
OUT=ROOT/'results/asd-paper-metrics'


def git_source(manifest):
    entries=[]
    for raw in subprocess.check_output(['git','ls-tree','-r','-z',manifest['commit']],cwd=ROOT).split(b'\0'):
        if raw:
            metadata,name=raw.split(b'\t',1);entries.append((name.decode(),metadata.split()[2]))
    assert set(manifest['files'])=={n for n,_ in entries}
    raw=subprocess.run(['git','cat-file','--batch'],input=b''.join(oid+b'\n' for _,oid in entries),
        stdout=subprocess.PIPE,cwd=ROOT,check=True).stdout
    offset=0
    for name,oid in entries:
        end=raw.index(b'\n',offset);header=raw[offset:end].split();assert header[:2]==[oid,b'blob']
        size=int(header[2]);content=raw[end+1:end+size+1];offset=end+size+2
        assert hashlib.sha256(content).hexdigest()==manifest['files'][name],name
    return len(entries)


def main():
    publication=json.loads((OUT/'publication.json').read_bytes())
    for name,item in publication['files'].items():
        content=(OUT/name).read_bytes();assert len(content)==item['bytes'] and hashlib.sha256(content).hexdigest()==item['sha256']
    manifest=json.loads((OUT/'ASD_PAPER_METRICS_SOURCE_MANIFEST.json').read_bytes());count=git_source(manifest)
    verification=json.loads((OUT/'execution-verification.json').read_bytes())
    assert verification['status']=='complete' and verification['files_verified_before']==verification['files_verified_after']==count
    rows=[r for shard in (0,1) for r in read(OUT/f'part-{shard}/rows.jsonl')]
    assert len(rows)==747
    saved=json.loads((OUT/'analysis.json').read_bytes());rebuilt=aggregate(ROOT,rows)
    assert all(saved[k]==v for k,v in rebuilt.items()) and saved['reward']==rewards(OUT)
    for shard,expected in ((0,375),(1,372)):
        info=json.loads((OUT/f'part-{shard}/provenance.json').read_bytes())
        assert info['status']=='complete' and info['count']==expected and info['original_unchanged']
        assert info['peak_gpu_bytes']<=46*2**30 and info['cache_bytes']<=4*2**30
        assert info['compiler']['graph_count']==info['compiler_after_warmup']['graph_count']
        assert info['compiler']['float32_matmul_precision']=='highest' and not info['compiler']['allow_tf32']
        engine=info['runtime']['engine']
        assert engine['dtype']=='torch.float32' and engine['gpu']=='NVIDIA RTX A6000'
        assert engine['execution_profile']=='rmsnorm-inductor' and engine['draft_adapter'] is None and engine['prefix_cache'] is None
        assert len(read(OUT/f'part-{shard}/warmup.jsonl'))==2
        records,refs=inputs(ROOT,shard)
        assert len(records)*3==expected
    # Independently label every emitted position as either draft or Target-produced.
    audited=0
    for row in read_rows(ROOT,'rows'):
        if row['repeat']!=0:continue
        total=len(row['token_ids']);draft=set();target={0} if total else set()
        for block in row['rounds']:
            start=block['generated_before'];accepted=block['accepted']
            positions=set(range(start,min(total,start+accepted)))
            assert not draft&positions;draft|=positions
            recovery=start+accepted
            if recovery<total:target.add(recovery)
        assert not draft&target and draft|target==set(range(total)),(row['request_id'],row['arm'],row['stage'])
        assert len(draft)==sum(min(b['accepted'],max(0,total-b['generated_before'])) for b in row['rounds'])
        audited+=1
    reward=saved['reward']
    if reward['status']=='complete':
        info=reward['provenance'];assert info['rows']==120 and info['controls_passed']>=7 and info['repeat_max_abs_error']<=1e-5
        assert info['peak_gpu_bytes']<=22*2**30
        messages={(u['request_id'],arm):message_sha(m) for u in open_units(ROOT) for arm,m in u['answers'].items()}
        for row in read(OUT/'reward/rows.jsonl'):assert messages[row['request_id'],row['arm']]==row['messages_sha256']
        spec=json.loads((ROOT/'configs/asd-reward-model.json').read_bytes())
        assert info['revision']==spec['revision'] and info['model']==spec['repo']
        execution=json.loads((OUT/'reward/execution.json').read_bytes())
        assert execution['status']=='complete' and execution['exit_code']==0
        assert execution['source_files_verified_before']==execution['source_files_verified_after']==count
        assert execution['source_sha256']==manifest['source_sha256']
        downloaded=json.loads((OUT/'reward/download-verification.json').read_bytes())
        assert downloaded['all_verified'] and downloaded['files']==len(spec['files'])==14
        assert downloaded['bytes']==sum(r['bytes'] for r in spec['files'])
        assert (OUT/'reward/model-manifest.json').read_bytes()==(ROOT/'configs/asd-reward-model.json').read_bytes()
    previous=json.loads((ROOT/'results/asd-paper/publication.json').read_bytes())
    for name,info in previous['files'].items():
        assert hashlib.sha256((ROOT/'results/asd-paper'/name).read_bytes()).hexdigest()==info['sha256']
    protected='27e36199f8a65acdb9a9c42ccb744ddfa9a9402b'
    for ref in ('main','codex/infra-interview-v0.3','infra-v0.3-interview^{}'):
        assert subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip()==protected
    suite=ET.parse(OUT/'tests-server.xml').getroot().find('testsuite')
    assert suite.attrib['failures']==suite.attrib['errors']=='0'
    assert int(suite.attrib['tests'])==28
    for name,tests,skipped in (('paper-metrics-tests-final-local.xml',28,0),('paper-metrics-full-v2-local.xml',1047,7)):
        suite=ET.parse(OUT/'local-checks'/name).getroot().find('testsuite')
        assert suite.attrib['failures']==suite.attrib['errors']=='0'
        assert int(suite.attrib['tests'])==tests and int(suite.attrib['skipped'])==skipped
    result=dict(source_commit=manifest['commit'],source_files=count,public_payloads=len(publication['files']),
        timed=747,requests=249,emitted_position_audits=audited,reward_status=reward['status'],
        source_data_input_token_time_resource_old_evidence_and_infra_checks=True)
    (OUT/'audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))


if __name__=='__main__':main()
