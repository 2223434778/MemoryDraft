"""Freeze independent seven-task records using previously pinned public sources."""
from collections import defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import re
import shutil
import subprocess

from memorydraft.evaluation.asd_paper import BENCHMARKS,expanded_ids,mbpp_interface,sha
from memorydraft.evaluation.asd_paper_extension import CONTROLS,COUNTS,SEED,STUDY

ROOT=Path(__file__).resolve().parents[1]


def rank(row):return sha((SEED+':'+row['id']).encode())


def balanced(rows,key,count):
    groups=defaultdict(list)
    for row in rows:groups[key(row)].append(row)
    names=sorted(groups,key=lambda name:sha((SEED+str(name)).encode()))
    for values in groups.values():values.sort(key=rank)
    return [groups[name][i] for i in range(max(map(len,groups.values())))
            for name in names if i<len(groups[name])][:count]


def exposures(root):
    identities=set();files={}
    paths=subprocess.check_output(['git','ls-files','-z','--','data/research','results'],cwd=root).split(b'\0')
    pattern=r'(?:gsm8k|math500|mbpp|humaneval|mmlu|mtbench|alpaca):\d+'
    for name in paths:
        if not name:continue
        relative=name.decode();path=root/relative
        if path.suffix not in ('.json','.jsonl','.gz') or relative.startswith(f'data/research/{STUDY}/'):continue
        raw=path.read_bytes();files[relative]=sha(raw)
        if path.suffix=='.gz':raw=gzip.decompress(raw)
        identities.update(re.findall(pattern,raw.decode('utf-8')))
    for benchmark,count in (('gsm8k',32),('math500',24),('mbpp',19),('humaneval',16)):
        identities.update(f'{benchmark}:{i}' for i in range(count))
    return identities,files


def source_pools(root):
    import pyarrow.parquet as pq
    old=root/'artifacts/asd-evaluation-sources';new=root/'artifacts/asd-paper-sources'
    sources=json.loads((new/'manifest.json').read_bytes())
    for name,entry in sources.items():
        if sha((new/name).read_bytes())!=entry['sha256']:raise ValueError('Changed pinned source '+name)
    legacy=json.loads((old/'manifest.json').read_bytes())
    for benchmark in ('mbpp','humaneval'):
        entry=legacy[benchmark]
        if sha((old/entry['filename']).read_bytes())!=entry['sha256']:raise ValueError('Changed code source')
        sources[benchmark]=entry
    gsm=root/'artifacts/asd-attribution-gsm-source.jsonl';math=root/'artifacts/asd-budget-math500-source.jsonl'
    if sha(gsm.read_bytes())!='3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14':
        raise ValueError('Changed GSM source')
    if sha(math.read_bytes())!='35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132':
        raise ValueError('Changed MATH source')
    prior=json.loads((root/'data/research/asd-paper/manifest.json').read_bytes())
    sources.update({b:prior['sources'][b] for b in ('gsm8k','math500')})
    lines=lambda data:[json.loads(line) for line in data.decode().splitlines()]
    pools=dict(
        gsm8k=[dict(r,id=f'gsm8k:{i}',benchmark='gsm8k',source_index=i) for i,r in enumerate(lines(gsm.read_bytes()))],
        math500=[dict(r,id=f'math500:{i}',benchmark='math500',source_index=i) for i,r in enumerate(lines(math.read_bytes()))],
        humaneval=[dict(r,id='humaneval:'+r['task_id'].split('/')[-1],benchmark='humaneval')
                   for r in lines(gzip.decompress((old/'humaneval.jsonl.gz').read_bytes()))],
        mbpp=[dict(r,id=f"mbpp:{r['task_id']}",benchmark='mbpp') for r in lines((old/'mbpp.jsonl').read_bytes())
              if 11<=r['task_id']<=510],
        mmlu=[dict(r,id=f'mmlu:{i}',benchmark='mmlu',source_index=i)
              for i,r in enumerate(pq.read_table(new/'mmlu.parquet').to_pylist())],
        mtbench=[dict(r,id=f"mtbench:{r['question_id']}",benchmark='mtbench')
                 for r in lines((new/'mtbench.jsonl').read_bytes())],
        alpaca=[dict(r,id=f'alpaca:{i}',benchmark='alpaca',source_index=i)
                for i,r in enumerate(json.loads((new/'alpaca.json').read_bytes()))])
    return pools,sources


def main():
    out=ROOT/'data/research'/STUDY
    if out.exists():raise FileExistsError('Refusing to overwrite frozen selection')
    pools,sources=source_pools(ROOT);excluded,files=exposures(ROOT);chosen={};sizes={}
    for benchmark in BENCHMARKS:
        candidates=[r for r in pools[benchmark] if r['id'] not in excluded];sizes[benchmark]=len(candidates)
        if benchmark in ('math500','mmlu','mtbench','alpaca'):
            key=(lambda r:bool(r['input'])) if benchmark=='alpaca' else (
                lambda r:r['category'] if benchmark=='mtbench' else r['subject'])
            selected=balanced(candidates,key,COUNTS[benchmark])
        else:selected=sorted(candidates,key=rank)[:COUNTS[benchmark]]
        if len(selected)!=COUNTS[benchmark]:raise ValueError('Insufficient unused '+benchmark+': '+str(sizes[benchmark]))
        chosen[benchmark]=[dict(r,signature=mbpp_interface(r)) if benchmark=='mbpp' else r for r in selected]
    out.mkdir(parents=True);hashes={}
    for benchmark,rows in chosen.items():
        raw=(json.dumps(rows,ensure_ascii=False,indent=2)+'\n').encode();name=benchmark+'.json'
        (out/name).write_bytes(raw);hashes[name]=sha(raw)
    prior=ROOT/'data/research/asd-paper'
    for path in prior.iterdir():
        if path.is_file() and (path.suffix in ('.md','.txt') or 'LICENSE' in path.name):shutil.copyfile(path,out/path.name)
    exclusion_bytes=(json.dumps(dict(files=files,excluded_ids=sorted(excluded)),indent=2)+'\n').encode()
    (out/'exclusions.json').write_bytes(exclusion_bytes)
    frontier=[r['id'] for b in BENCHMARKS for r in chosen[b][:4]]
    manifest=dict(schema=1,seed=SEED,counts=COUNTS,files=hashes,sources=sources,pool_sizes=sizes,
        ids={b:[r['id'] for r in rows] for b,rows in chosen.items()},exclusions_sha256=sha(exclusion_bytes),
        frontier_ids=frontier,controls={k:asdict(v) for k,v in CONTROLS.items()},
        selection='SHA rank of unused public IDs; balanced subject/category/input presence; no output filtering',
        scope='Cross-task DFlash extension; not an exact DSpark/L20 or official GPT-4 quality reproduction')
    (out/'manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
    rows=[r for b in BENCHMARKS for r in chosen[b]]
    print(json.dumps(dict(source_tasks=len(rows),requests=len(expanded_ids(rows)),unused_pool_sizes=sizes,
        frontier_requests=len(expanded_ids([r for r in rows if r['id'] in set(frontier)])),
        excluded_ids=len(excluded),prior_files=len(files),manifest_sha256=sha((out/'manifest.json').read_bytes()))))


if __name__=='__main__':main()
