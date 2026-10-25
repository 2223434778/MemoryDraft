"""Verify every public payload, optionally against canonical HEAD Git blobs."""
import argparse
import hashlib,json,subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--git',action='store_true');args=parser.parse_args()
    root=Path.cwd();out=root/'results/asd-paper-extension'
    publication=json.loads((out/'publication.json').read_bytes())
    derived=json.loads((out/'derived-publication.json').read_bytes())
    digest=lambda data:hashlib.sha256(data).hexdigest()
    assert digest((out/'publication.json').read_bytes())==derived['input_publication_sha256']
    assert not set(publication['files'])&set(derived['files'])
    files={**publication['files'],**derived['files']}
    for name,item in files.items():
        path=out/name;assert path.resolve().is_relative_to(out.resolve())
        data=path.read_bytes();assert len(data)==item['bytes'] and digest(data)==item['sha256'],name
    for name in ('publication.json','derived-publication.json'):
        data=(out/name).read_bytes();files[name]=dict(bytes=len(data),sha256=digest(data))
    for name,sha in derived['reports'].items():assert digest((root/name).read_bytes())==sha,name
    lock=json.loads((out/'contracts/review/review-lock.json').read_bytes())
    assert digest((out/'contracts/review/reviews.json').read_bytes())==lock['reviews_sha256']
    committed=digest(subprocess.check_output(['git','show',derived['anonymous_review_commit']+':results/asd-paper-extension/contracts/review/reviews.json']))
    assert committed==lock['reviews_sha256']
    count=0
    if args.git:
        entries={}
        for raw in subprocess.check_output(['git','ls-tree','-r','-z','HEAD','--','results/asd-paper-extension']).split(b'\0'):
            if raw:
                info,name=raw.split(b'\t',1);entries[name.decode()]=info.split()[2]
        names=['results/asd-paper-extension/'+n for n in sorted(files)]
        oids=[entries[n] for n in names]
        raw=subprocess.run(['git','cat-file','--batch'],input=b''.join(oid+b'\n' for oid in oids),stdout=subprocess.PIPE,check=True).stdout
        offset=0
        for name,oid in zip(names,oids):
            end=raw.index(b'\n',offset);header=raw[offset:end].split();assert header[:2]==[oid,b'blob']
            size=int(header[2]);data=raw[end+1:end+size+1];offset=end+size+2
            item=files[name.removeprefix('results/asd-paper-extension/')]
            assert size==item['bytes'] and digest(data)==item['sha256'],name;count+=1
        assert offset==len(raw)
    for ref in ('main','codex/infra-interview-v0.3','infra-v0.3-interview^{}'):
        assert subprocess.check_output(['git','rev-parse',ref],text=True).strip()=='27e36199f8a65acdb9a9c42ccb744ddfa9a9402b'
    print(json.dumps(dict(original_payloads=len(publication['files']),derived_payloads=len(derived['files']),
        git_payloads_verified=count,report_hashes_verified=len(derived['reports']),anonymous_review_frozen=True,infra_refs_preserved=True)))


if __name__=='__main__':main()
