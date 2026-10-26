"""Verify raw ZIP-member hashes, public payloads, report bytes and optional Git blobs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--git',action='store_true');args=parser.parse_args()
    root=Path.cwd();out=root/'results/dspark-tree-opt'
    manifest=json.loads((out/'publication.json').read_bytes());sha=lambda data:hashlib.sha256(data).hexdigest()
    for name,item in manifest['files'].items():
        path=out/name;assert path.resolve().is_relative_to(out.resolve())
        data=path.read_bytes();assert len(data)==item['bytes'] and sha(data)==item['sha256'],name
    for name,item in manifest['original_archive_members'].items():
        assert sha((out/name).read_bytes())==item['sha256'],name
    for name,digest in manifest['reports'].items():assert sha((root/name).read_bytes())==digest,name
    protected=json.loads((out/'local-checks/preservation.json').read_bytes())
    entries={}
    for raw in subprocess.check_output(['git','ls-tree','-r','-z','HEAD']).split(b'\0'):
        if raw:
            metadata,name=raw.split(b'\t',1);entries[name.decode()]=metadata.split()[2]
    for name,oid in protected['protected_git_objects'].items():assert entries[name].decode()==oid,name
    for ref,oid in protected['protected_refs'].items():
        assert subprocess.check_output(['git','rev-parse',ref],text=True).strip()==oid,ref
    count=0
    if args.git:
        expected={'results/dspark-tree-opt/'+name:item for name,item in manifest['files'].items()}
        data=(out/'publication.json').read_bytes()
        expected['results/dspark-tree-opt/publication.json']={'bytes':len(data),'sha256':sha(data)}
        expected.update({name:{'bytes':(root/name).stat().st_size,'sha256':digest} for name,digest in manifest['reports'].items()})
        names=sorted(expected);oids=[entries[name] for name in names]
        raw=subprocess.run(['git','cat-file','--batch'],input=b''.join(oid+b'\n' for oid in oids),stdout=subprocess.PIPE,check=True).stdout
        offset=0
        for name,oid in zip(names,oids):
            end=raw.index(b'\n',offset);head=raw[offset:end].split();assert head[:2]==[oid,b'blob']
            size=int(head[2]);data=raw[end+1:end+1+size];offset=end+size+2;item=expected[name]
            assert size==item['bytes'] and sha(data)==item['sha256'],name;count+=1
        assert offset==len(raw)
    print(json.dumps(dict(payloads=len(manifest['files']),original_archive_members=len(manifest['original_archive_members']),
        reports=len(manifest['reports']),git_blobs_verified=count,protected_old_files=len(protected['protected_git_objects']),protected_refs=True)))


if __name__=='__main__':main()
