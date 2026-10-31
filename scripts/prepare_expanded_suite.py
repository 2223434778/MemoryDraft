"""Freeze deterministic new-task subsets from downloaded official data."""
import base64
import gzip
import hashlib
import io
import json
import pickle
from pathlib import Path
import shutil
import subprocess
import zlib

ROOT=Path(__file__).resolve().parents[1]
SEED='MemoryDraft-expanded-v1'


class StringUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        raise ValueError('Only literal test payloads are supported')


def ordered(rows,key):
    return sorted(rows,key=lambda r:hashlib.sha256((SEED+':'+str(r[key])).encode()).hexdigest())


def main():
    cache=ROOT/'artifacts/expanded-cache'
    out=ROOT/'data/research/expanded-suite';out.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((out/'manifest.json').read_bytes()) if (out/'manifest.json').exists() else None
    records=[]
    examples=[json.loads(s) for s in (cache/'ifeval-data-input_data.jsonl').read_text(encoding='utf-8').splitlines()]
    for r in ordered(examples,'key')[:64]:
        records.append(dict(id='ifeval:'+str(r['key']),benchmark='ifeval',prompt=r['prompt'],ifeval=r,limit=2048))
    examples=[json.loads(s) for s in (cache/'lcb-test6.jsonl').read_text(encoding='utf-8').splitlines()]
    for difficulty in ('easy','medium','hard'):
        for r in ordered([x for x in examples if x['difficulty']==difficulty],'question_id')[:8]:
            public=json.loads(r['public_test_cases'])
            try:private=json.loads(r['private_test_cases'])
            except (ValueError,TypeError):
                private=json.loads(StringUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(r['private_test_cases'])))).load())
            tests=public+private
            meta=json.loads(r['metadata']) if isinstance(r['metadata'],str) else r['metadata']
            prompt='Write a complete Python solution. Return only the solution code.\n\n'+r['question_content']
            if r['starter_code']:prompt+='\n\nRequired starter code:\n'+r['starter_code']
            records.append(dict(id='livecodebench:'+r['platform']+':'+r['question_id'],benchmark='livecodebench',
                prompt=prompt,limit=2048,difficulty=difficulty,contest_date=r['contest_date'],
                input_output=json.dumps(dict(inputs=[t['input'] for t in tests],outputs=[t['output'] for t in tests],fn_name=meta.get('func_name')))))
    for r in ordered(json.loads((cache/'nuprl-CanItEdit.json').read_bytes()),'full_name')[:32]:
        messages=[dict(role='system',content='You are PythonEditGPT. You will be provided the original code snippet and an instruction that specifies the changes you need to make. You will produce the changed code, based on the original code and the instruction given. Only produce the code, do not include any additional prose.'),
            dict(role='user',content='## Code Before\n```py\n'+r['before']+'\n```\n\n## Instruction\n'+r['instruction_descriptive']+'\n\n## Code After')]
        records.append(dict(id='canitedit:'+r['full_name'],benchmark='canitedit',messages=messages,
            tests=r['tests'],canonical=r['after'],before=r['before'],limit=2048))
    templates=json.loads((ROOT/'artifacts/upstream-longbench/LongBench/config/dataset2prompt.json').read_bytes())
    for subset in ('qasper','multifieldqa_en','multi_news','gov_report'):
        examples=[json.loads(s) for s in (cache/(subset+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        for index,r in enumerate(examples):r['_index']=index
        for r in ordered(examples,'_index')[:8]:
            records.append(dict(id='longbench-'+subset+':'+str(r['_index']),benchmark='longbench-'+subset,
                subset=subset,prompt=templates[subset].format(**r),answers=r['answers'],
                limit=512 if subset in ('multi_news','gov_report') else 128,clip_input_tokens=3840))
    for r in json.loads((cache/'math-ai-aime25.json').read_bytes()):
        records.append(dict(id='aime25:'+r['id'],benchmark='aime25',answer=r['answer'],limit=4096,
            prompt=r['problem']+'\nPlease reason step by step, and put your final answer within \\boxed{}.'))
    records.sort(key=lambda r:(list(['ifeval','livecodebench','canitedit','longbench-qasper','longbench-multifieldqa_en','longbench-multi_news','longbench-gov_report','aime25']).index(r['benchmark']),r['id']))
    raw=bytearray(gzip.compress(json.dumps(records,ensure_ascii=False,sort_keys=True).encode(),mtime=0))
    raw[9]=255
    raw=bytes(raw)
    if frozen and hashlib.sha256(raw).hexdigest()!=frozen['tasks_sha256']:
        raise ValueError('Prepared tasks differ from the frozen selection')
    (out/'tasks.json.gz').write_bytes(raw)
    third=ROOT/'third_party/expanded';third.mkdir(parents=True,exist_ok=True)
    upstream={}
    for name,source,files,commit,license in [
        ('instruction_following_eval',ROOT/'artifacts/ifeval-source',['instructions.py','instructions_registry.py','instructions_util.py','evaluation_lib.py','LICENSE'],'83f1e7b59e762648c953de751d31f75720683622','Apache-2.0'),
        ('longbench',ROOT/'artifacts/upstream-longbench',['LongBench/metrics.py','LICENSE'],'2e00731f8d0bff23dc4325161044d0ed8af94c1e','MIT'),
        ('lcb',ROOT/'artifacts/upstream-lcb',['lcb_runner/evaluation/testing_util.py','LICENSE'],'28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24','MIT'),
        ('canitedit',ROOT/'artifacts/upstream-canitedit',['LICENSE','benchmark/generate_completions.py'],'74d15ea7e6207cb9fdeeecd761907371d4cc2e26','BSD-3-Clause with no-training restriction')]:
        target=third/name;target.mkdir(exist_ok=True)
        digests={}
        for path in files:
            dest=target/Path(path).name
            from scripts.prepare_external_trees import published_bytes
            relative=dest.relative_to(ROOT).as_posix()
            expected=frozen['upstream'][name]['files'].get(relative) if frozen else None
            dest.write_bytes(published_bytes((source/path).read_bytes(),expected))
            digests[dest.relative_to(ROOT).as_posix()]=hashlib.sha256(dest.read_bytes()).hexdigest()
        upstream[name]=dict(commit=commit,license=license,files=digests)
    counts={b:sum(r['benchmark']==b for r in records) for b in dict.fromkeys(r['benchmark'] for r in records)}
    manifest=dict(seed=SEED,count=len(records),counts=counts,tasks_sha256=hashlib.sha256(raw).hexdigest(),upstream=upstream,
        datasets={p.name:json.loads(p.read_bytes())['sha'] for p in cache.glob('*-info.json')},
        raw_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.iterdir() if p.is_file() and p.name!='longbench-data.zip'},
        selection='SHA order before model outputs; CanItEdit descriptive; LiveCodeBench v6-only 8 easy/medium/hard; LongBench8/subset with official middle clipping at3840 prompt tokens; full AIME25')
    if frozen:
        if manifest['upstream']!=frozen['upstream']:raise ValueError('Scorer source differs from frozen source')
    else:(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(counts=counts,total=len(records)),indent=2))


if __name__=='__main__':main()
