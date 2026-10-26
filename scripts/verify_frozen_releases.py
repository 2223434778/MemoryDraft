"""Verify archived payloads and source at frozen refs while the runtime evolves."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REFS={'codex/infra-interview-v0.3':'27e36199f8a65acdb9a9c42ccb744ddfa9a9402b',
      'infra-v0.3-interview^{}':'27e36199f8a65acdb9a9c42ccb744ddfa9a9402b',
      'codex/asd-splice':'8e7a80b5b1af3b09886659e3549691be3ce6493e',
      'codex/dspark-tree':'96773223039b1faaa61c823d1b54da4b2edec34b',
      'codex/dspark-tree-opt':'f56561438a68f7362ffc8cec61f409b248c5f4f0'}


def verify():
    for reference,oid in REFS.items():
        actual=subprocess.check_output(['git','rev-parse',reference],cwd=ROOT,text=True).strip()
        if actual!=oid:raise ValueError('Frozen reference moved: '+reference)
    reports=[]
    for study,reference in (('dspark-tree','codex/dspark-tree'),('dspark-tree-opt','codex/dspark-tree-opt'),
                             ('asd-paper-extension','codex/asd-splice')):
        out=ROOT/'results'/study;manifest=json.loads((out/'publication.json').read_bytes())
        for name,item in manifest['files'].items():
            path=out/name
            if not path.resolve().is_relative_to(out.resolve()):raise ValueError('Unsafe archived path')
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=item['sha256'] or len(raw)!=item['bytes']:
                raise ValueError('Archived payload changed: '+str(path))
        for name,digest in manifest.get('reports',{}).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Archived report changed: '+name)
        paths=['results/'+study,*manifest.get('reports',{})]
        if study=='asd-paper-extension':
            paths+=['docs/ASD_PAPER_EXTENSION_FINDINGS.md','docs/ASD_PAPER_EXTENSION_RESULTS.md']
        archived=[]
        for entry in subprocess.check_output(['git','ls-tree','-r','-z',reference,'--',*paths],cwd=ROOT).split(b'\0'):
            if entry:
                metadata,name=entry.split(b'\t',1);archived.append((name.decode(),metadata.split()[2]))
        raw=subprocess.run(['git','cat-file','--batch'],cwd=ROOT,input=b''.join(oid+b'\n' for _,oid in archived),
                           stdout=subprocess.PIPE,check=True).stdout
        offset=0
        for name,oid in archived:
            end=raw.index(b'\n',offset);header=raw[offset:end].split();size=int(header[2])
            if header[:2]!=[oid,b'blob'] or (ROOT/name).read_bytes()!=raw[end+1:end+1+size]:
                raise ValueError('Archived Git payload changed: '+name)
            offset=end+size+2
        if offset!=len(raw):raise ValueError('Truncated archived Git stream')
        count=0;preservation=out/'local-checks/preservation.json'
        if preservation.exists():
            protected=json.loads(preservation.read_bytes())['protected_git_objects']
            entries={}
            for entry in subprocess.check_output(['git','ls-tree','-r','-z',reference],cwd=ROOT).split(b'\0'):
                if entry:
                    metadata,name=entry.split(b'\t',1);entries[name.decode()]=metadata.split()[2].decode()
            for name,oid in protected.items():
                if entries.get(name)!=oid:raise ValueError('Frozen source differs: '+name)
            count=len(protected)
        reports.append(dict(study=study,payloads=len(manifest['files']),archived_git_files=len(archived),
                            frozen_reference=reference,protected_source_objects=count))
    return dict(frozen_refs=True,scope='published payloads unchanged; source checked at archived refs, not evolving HEAD',studies=reports)


if __name__=='__main__':print(json.dumps(verify(),indent=2))
