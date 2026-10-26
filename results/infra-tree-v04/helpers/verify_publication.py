"""Byte-exact evidence and frozen runtime source checks, with optional Git audit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--git',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[3];out=root/'results/infra-tree-v04'
    publication=json.loads((out/'publication.json').read_bytes());sha=lambda value:hashlib.sha256(value).hexdigest()
    expected={}
    for name,item in publication['files'].items():
        path=out/name;assert path.resolve().is_relative_to(out.resolve())
        value=path.read_bytes();assert len(value)==item['bytes'] and sha(value)==item['sha256'],name
        expected['results/infra-tree-v04/'+name]=item
    for name,item in publication['original_archive_members'].items():assert sha((out/name).read_bytes())==item['sha256'],name
    for name,digest in publication['reports'].items():
        value=(root/name).read_bytes();assert sha(value)==digest,name
        expected[name]=dict(bytes=len(value),sha256=digest)
    value=(out/'publication.json').read_bytes()
    expected['results/infra-tree-v04/publication.json']=dict(bytes=len(value),sha256=sha(value))
    manifest=json.loads((out/'INFRA_TREE_SOURCE_MANIFEST.json').read_bytes())
    assert manifest['source_sha256']==sha(json.dumps(manifest['files'],sort_keys=True).encode())
    core={n:h for n,h in manifest['files'].items() if n.startswith('memorydraft/') and n.endswith(('.py','.json','.txt'))}
    entries={}
    for entry in subprocess.check_output(['git','ls-tree','-r','-z',manifest['commit'],'--','memorydraft'],cwd=root).split(b'\0'):
        if entry:
            meta,name=entry.split(b'\t',1);name=name.decode()
            if name in core:entries[name]=meta.split()[2]
    def check_blobs(names,oids,wanted):
        raw=subprocess.run(['git','cat-file','--batch'],cwd=root,input=b''.join(oid+b'\n' for oid in oids),stdout=subprocess.PIPE,check=True).stdout
        offset=0
        for name,oid in zip(names,oids):
            end=raw.index(b'\n',offset);header=raw[offset:end].split();assert header[:2]==[oid,b'blob']
            size=int(header[2]);data=raw[end+1:end+1+size];offset=end+size+2
            assert sha(data)==wanted[name]['sha256'],name
            if 'bytes' in wanted[name]:assert size==wanted[name]['bytes'],name
        assert offset==len(raw)
    assert set(entries)==set(core)
    names=sorted(core);check_blobs(names,[entries[n] for n in names],{n:dict(sha256=h) for n,h in core.items()})
    count=0
    if args.git:
        entries={}
        for entry in subprocess.check_output(['git','ls-tree','-r','-z','HEAD'],cwd=root).split(b'\0'):
            if entry:
                meta,name=entry.split(b'\t',1);entries[name.decode()]=meta.split()[2]
        assert {n for n in entries if n.startswith('results/infra-tree-v04/')}=={n for n in expected if n.startswith('results/')}
        names=sorted(expected);check_blobs(names,[entries[n] for n in names],expected);count=len(names)
    print(json.dumps(dict(payloads=len(publication['files']),original_members=len(publication['original_archive_members']),
        reports=len(publication['reports']),frozen_package_files=len(core),git_public_files=count)))


if __name__=='__main__':main()
