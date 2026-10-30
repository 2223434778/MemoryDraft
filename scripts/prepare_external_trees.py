"""Prepare pinned, unmodified official sources from two existing local clones."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.run_external_trees import UPSTREAMS


def published_bytes(raw, expected):
    if expected is None or hashlib.sha256(raw).hexdigest()==expected:
        return raw
    lf=raw.replace(b'\r\n',b'\n')
    for candidate in (lf,lf.replace(b'\n',b'\r\n')):
        if hashlib.sha256(candidate).hexdigest()==expected:
            return candidate
    raise ValueError('Upstream bytes differ beyond checkout line endings')


def prepare(root, clones):
    manifest={}
    published=root/'results/external-trees/upstreams.json'
    expected=json.loads(published.read_bytes()) if published.exists() else {}
    for name,(commit,url) in UPSTREAMS.items():
        clone=clones[name].resolve()
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=clone,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=clone,text=True)
        if actual!=commit or dirty:
            raise ValueError('Expected clean pinned upstream: '+name)
        paths=subprocess.check_output(['git','ls-files'],cwd=clone,text=True).splitlines()
        selected=[p for p in paths if p.endswith('.py') or p in ('LICENSE','README.md','pyproject.toml','requirements.txt')]
        hashes={}
        for name_in_repo in selected:
            source=clone/name_in_repo
            target=root/'third_party'/name/name_in_repo
            if not source.resolve().is_relative_to(clone) or not target.resolve().is_relative_to(root.resolve()):
                raise ValueError('Unsafe upstream path')
            raw=published_bytes(source.read_bytes(),expected.get(name,{}).get('files',{}).get(name_in_repo))
            if target.exists() and target.read_bytes()!=raw:
                raise ValueError('Refuse to overwrite different upstream file: '+str(target))
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
            hashes[name_in_repo]=hashlib.sha256(raw).hexdigest()
        manifest[name]=dict(commit=commit,url=url,license='MIT',files=hashes)
    if expected and manifest!=expected:
        raise ValueError('Prepared source differs from published GPU source')
    output=root/'UPSTREAM_TREES.json'
    if output.exists() and json.loads(output.read_bytes())!=manifest:
        raise ValueError('Refuse to replace a different upstream manifest')
    output.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bastion',type=Path,required=True)
    parser.add_argument('--ddtree',type=Path,required=True)
    args=parser.parse_args()
    manifest=prepare(Path(__file__).resolve().parents[1],dict(bastion=args.bastion,ddtree=args.ddtree))
    print(json.dumps({n:dict(commit=v['commit'],files=len(v['files'])) for n,v in manifest.items()},indent=2))


if __name__=='__main__':main()
