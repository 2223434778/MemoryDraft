"""Download official public data and scorer sources before offline evaluation."""
import hashlib,json,time,zipfile,io
from pathlib import Path
import requests


def main():
    root=Path(__file__).resolve().parents[1]
    cache=root/'artifacts/expanded-cache';cache.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    def get(url):
        r=session.get(url,timeout=90);r.raise_for_status();return r
    def save(name,url):
        p=cache/name
        if not p.exists():p.write_bytes(get(url).content)
        return p.read_bytes()
    pin='83f1e7b59e762648c953de751d31f75720683622'
    directory=root/'artifacts/ifeval-source';directory.mkdir(exist_ok=True)
    for n in ['instructions.py','instructions_registry.py','instructions_util.py','evaluation_lib.py','data/input_data.jsonl']:
        p=directory/n;p.parent.mkdir(exist_ok=True,parents=True)
        p.write_bytes(save('ifeval-'+n.replace('/','-'),'https://raw.githubusercontent.com/google-research/google-research/'+pin+'/instruction_following_eval/'+n))
    (directory/'LICENSE').write_bytes(save('ifeval-LICENSE','https://raw.githubusercontent.com/google-research/google-research/'+pin+'/LICENSE'))
    for ds in ['nuprl/CanItEdit','math-ai/aime25','livecodebench/code_generation_lite','THUDM/LongBench']:
        pins={'nuprl/CanItEdit':'3c07f38b1f9385f3214fcea94d4664c79df0d36a','math-ai/aime25':'563bb8404243c5f09de6ec262f2db674fe5bce9b',
            'livecodebench/code_generation_lite':'0fe84c3912ea0c4d4a78037083943e8f0c4dd505','THUDM/LongBench':'5e628be450b7e67fb7ae6e201bd6d8f7056f7672'}
        info=get('https://huggingface.co/api/datasets/'+ds+'/revision/'+pins[ds]).json()
        (cache/(ds.replace('/','-')+'-info.json')).write_text(json.dumps(info,indent=2))
        print(ds,info['sha'],[s['rfilename'] for s in info['siblings']],flush=True)
        data_path=cache/(ds.replace('/','-')+'.json')
        if ds in ['nuprl/CanItEdit','math-ai/aime25'] and not data_path.exists():
            if ds=='nuprl/CanItEdit':
                import pyarrow.parquet as pq
                rows=pq.read_table(io.BytesIO(save('canitedit.parquet','https://huggingface.co/datasets/'+ds+'/resolve/'+pins[ds]+'/data/test-00000-of-00001.parquet'))).to_pylist()
            else:rows=[json.loads(s) for s in save('aime25.jsonl','https://huggingface.co/datasets/'+ds+'/resolve/'+pins[ds]+'/test.jsonl').decode().splitlines()]
            data_path.write_text(json.dumps(rows),encoding='utf-8')
        if ds=='livecodebench/code_generation_lite':
            (cache/'lcb-loader.py').write_bytes(save('lcb-loader.py','https://huggingface.co/datasets/'+ds+'/resolve/'+info['sha']+'/code_generation_lite.py'))
            save('lcb-test6.jsonl','https://huggingface.co/datasets/'+ds+'/resolve/'+info['sha']+'/test6.jsonl?download=true')
        if ds=='THUDM/LongBench':
            raw=save('longbench-data.zip','https://huggingface.co/datasets/THUDM/LongBench/resolve/'+info['sha']+'/data.zip?download=true&x='+str(time.time()))
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for subset in ['qasper','multifieldqa_en','multi_news','gov_report']:
                    names=[n for n in z.namelist() if n.endswith('/'+subset+'.jsonl') or n==subset+'.jsonl']
                    assert len(names)==1,names
                    (cache/(subset+'.jsonl')).write_bytes(z.read(names[0]))
    print('download complete',flush=True)

if __name__=="__main__":main()
