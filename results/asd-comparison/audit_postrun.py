"""Audit published bytes, Git content/EOL conversion and independent infra refs."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def digest(data):return hashlib.sha256(data).hexdigest()


def git_source(root,manifest):
    commit=manifest['commit']
    names=list(manifest['files'])
    output=subprocess.check_output(['git','cat-file','--batch'],cwd=root,
        input=('\n'.join(commit+':'+n for n in names)+'\n').encode())
    position=0
    line_endings={}
    for name in names:
        end=output.index(b'\n',position)
        header=output[position:end].split()
        if len(header)!=3 or header[1]!=b'blob':raise ValueError('Missing frozen Git blob: '+name)
        size=int(header[2]);position=end+1
        blob=output[position:position+size];position+=size+1
        if digest(blob)!=manifest['files'][name]:
            lf=blob.replace(b'\r\n',b'\n')
            variants={'lf':lf,'crlf':lf.replace(b'\n',b'\r\n')}
            kind=next((k for k,v in variants.items() if digest(v)==manifest['files'][name]),None)
            if kind is None:raise ValueError('Frozen source content changed: '+name)
            line_endings[name]=dict(git_sha256=digest(blob),deployed_sha256=manifest['files'][name],archive_eol=kind)
    if position!=len(output):raise ValueError('Unexpected Git batch output')
    return dict(source_commit=commit,frozen_source_files=len(names),exact_git_blobs=len(names)-len(line_endings),
        archive_line_ending_only_files=len(line_endings),line_endings=line_endings)


def main():
    base=Path(__file__).resolve().parent
    root=base.parents[1]
    manifest=json.loads((base/'ASD_COMPARISON_SOURCE_MANIFEST.json').read_bytes())
    source=git_source(root,manifest)
    entries=0
    for folder in (base,base/'startup-failure'):
        publication=json.loads((folder/'publication.json').read_bytes())
        for item in publication['files']:
            data=(folder/item['published']).read_bytes()
            if digest(data)!=item['published_sha256'] or len(data)!=item['published_bytes']:
                raise ValueError('Published bytes changed: '+item['published'])
            raw=gzip.decompress(data) if item['published'].endswith('.gz') else data
            if digest(raw)!=item['raw_sha256'] or len(raw)!=item['raw_bytes']:
                raise ValueError('Original payload changed: '+item['source'])
            entries+=1
    execution=json.loads((base/'execution-verification.json').read_bytes())
    if (execution['status']!='complete' or execution['files_verified_before']!=execution['files_verified_after']
            or execution['files_verified_before']!=len(manifest['files'])):
        raise ValueError('Execution source verification failed')
    expected='27e36199f8a65acdb9a9c42ccb744ddfa9a9402b'
    for ref in ('main','codex/infra-interview-v0.3','infra-v0.3-interview^{}'):
        if subprocess.check_output(['git','rev-parse',ref],cwd=root,text=True).strip()!=expected:
            raise ValueError('Independent infra reference changed: '+ref)
    changed=subprocess.check_output(['git','diff','--name-only','89a57b5bb775a4ca9a98e6e8f02d2608cda4f55c','--','results'],cwd=root,text=True).splitlines()
    if any(not p.startswith(('results/asd-comparison/','results/asd-comparison-engineering/')) for p in changed):
        raise ValueError('Older published results changed')
    local=json.loads((base/'local-checks/manifest.json').read_bytes())
    for entry in local['files']:
        data=(base/'local-checks'/entry['published']).read_bytes()
        if len(data)!=entry['bytes'] or digest(data)!=entry['sha256']:
            raise ValueError('Local verification evidence changed')
    result=dict(**source,published_payloads=entries,local_check_payloads=len(local['files']),
        infra_refs_unchanged=True,older_results_unchanged=True)
    destination=root/'artifacts/comparison-source-audit.json'
    destination.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in result.items() if k!='line_endings'}))


if __name__=='__main__':main()
