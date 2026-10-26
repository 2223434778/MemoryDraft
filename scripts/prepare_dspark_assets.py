"""Download the pinned public DSpark checkpoint without touching DFlash assets."""
import argparse
import hashlib
import json
from pathlib import Path
from importlib import resources
import requests

ASSETS=json.loads(resources.files('memorydraft').joinpath('assets/dspark-manifest.json').read_text())
REPOSITORY=ASSETS['repository']
REVISION=ASSETS['revision']
WEIGHT_SHA256=ASSETS['files']['model.safetensors']['sha256']
WEIGHT_BYTES=ASSETS['files']['model.safetensors']['bytes']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest={'repository':REPOSITORY,'revision':REVISION,'files':{}}
    for name in ('config.json','model.safetensors'):
        path=args.output/name
        if path.exists():
            if name=='model.safetensors' and (path.stat().st_size!=WEIGHT_BYTES or sha(path)!=WEIGHT_SHA256):
                raise ValueError('Existing DSpark weight checksum mismatch; refusing overwrite')
        else:
            temporary=path.with_suffix(path.suffix+'.partial')
            with requests.get(f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}',
                              stream=True,timeout=(30,120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(8*1024*1024):stream.write(chunk)
            if name=='model.safetensors' and (temporary.stat().st_size!=WEIGHT_BYTES or sha(temporary)!=WEIGHT_SHA256):
                raise ValueError('Downloaded DSpark weight checksum mismatch')
            temporary.rename(path)
        manifest['files'][name]={'bytes':path.stat().st_size,'sha256':sha(path)}
        if manifest['files'][name]!=ASSETS['files'][name]:raise ValueError('Pinned asset differs: '+name)
        print(json.dumps({'verified':name,**manifest['files'][name]}),flush=True)
    (args.output/'VERIFIED.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
