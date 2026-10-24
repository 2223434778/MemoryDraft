"""Download a pinned public judge checkpoint; verify every payload before use."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request


def verify(path,item):
    if not path.is_file() or path.stat().st_size!=item['bytes']:return False
    digest=hashlib.sha256() if item.get('sha256') else hashlib.sha1()
    if not item.get('sha256'):digest.update(b'blob '+str(item['bytes']).encode()+b'\0')
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(8*2**20),b''):digest.update(data)
    return digest.hexdigest()==(item.get('sha256') or item['git_blob'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'configs/asd-reward-model.json').read_bytes())
    out=Path(args.directory).resolve()
    if not out.is_relative_to(root.resolve()):raise ValueError('Judge download must remain inside workspace')
    out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    def ranged(item,path,url):
        chunk_size=16*2**20;chunks=path.with_name(path.name+'.chunks');chunks.mkdir(exist_ok=True)
        offsets=list(range(0,item['bytes'],chunk_size))
        def chunk(start):
            end=min(start+chunk_size,item['bytes'])-1
            dest=chunks/str(start);temporary=dest.with_suffix('.part')
            if dest.is_file() and dest.stat().st_size==end-start+1:return dest
            for retry in range(4):
                try:
                    request=urllib.request.Request(url+'?download=true',headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request,timeout=60) as response:
                        expected=f'bytes {start}-{end}/{item["bytes"]}'
                        if response.status!=206 or response.headers.get('Content-Range')!=expected:
                            raise ValueError('Unexpected checkpoint chunk range')
                        with temporary.open('wb') as stream:shutil.copyfileobj(response,stream,1024*1024)
                    if temporary.stat().st_size!=end-start+1:raise ValueError('Truncated checkpoint chunk')
                    temporary.replace(dest)
                    if start%(chunk_size*32)==0:print('chunk',item['filename'],start,flush=True)
                    return dest
                except Exception:
                    if retry==3:raise RuntimeError('Public checkpoint chunk failed at '+str(start)) from None
                    time.sleep(2)
        with ThreadPoolExecutor(max_workers=4) as pool:parts=list(pool.map(chunk,offsets))
        temporary=path.with_name(path.name+'.part')
        with temporary.open('wb') as stream:
            for part in parts:
                with part.open('rb') as source:shutil.copyfileobj(source,stream,8*2**20)
        if not verify(temporary,item):raise ValueError('Chunked checkpoint SHA mismatch')
        temporary.replace(path)
        for part in parts:part.unlink()
        chunks.rmdir();print('verified',item['filename'],item['bytes'],flush=True)
    def download(item):
        path=out/item['filename'];part=path.with_name(path.name+'.part')
        if verify(path,item):return
        if path.exists():raise ValueError('Unexpected invalid existing judge file')
        url=f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{item['filename']}"
        if item['bytes']>100*2**20:
            ranged(item,path,url);return
        for retry in range(4):
            try:
                offset=part.stat().st_size if part.exists() else 0
                headers={'User-Agent':'MemoryDraft-public-research'}
                if offset:headers['Range']=f'bytes={offset}-'
                with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as response:
                    if offset and response.status!=206:raise ValueError('Server did not honor resume range')
                    with part.open('ab' if offset else 'wb') as stream:
                        for data in iter(lambda:response.read(8*2**20),b''):stream.write(data)
                if not verify(part,item):raise ValueError('Downloaded checkpoint hash mismatch')
                part.replace(path);print('verified',item['filename'],item['bytes'],flush=True);return
            except Exception as error:
                print('download retry',item['filename'],retry,type(error).__name__,flush=True)
                if retry==3:raise RuntimeError('Public judge download failed: '+item['filename']) from None
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(download,manifest['files']))
    result=dict(repo=manifest['repo'],revision=manifest['revision'],files=len(manifest['files']),
        bytes=sum(v['bytes'] for v in manifest['files']),seconds=time.perf_counter()-started,all_verified=True)
    (out/'DOWNLOAD_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
