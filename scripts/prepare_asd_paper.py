"""Select unused public source IDs before any new model generation."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
import shutil

from memorydraft.evaluation.asd_paper import BENCHMARKS, COUNTS, SEED, mbpp_interface, sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/research/asd-paper'


def rank(row):return sha((SEED+':'+row['id']).encode())


def balanced(rows,key,count):
    groups=defaultdict(list)
    for row in rows:groups[key(row)].append(row)
    names=sorted(groups,key=lambda name:sha((SEED+str(name)).encode()))
    for rows in groups.values():rows.sort(key=rank)
    ordered=[groups[name][i] for i in range(max(map(len,groups.values()))) for name in names if i<len(groups[name])]
    return ordered[:count]


def exclusions():
    identities=set();files={}
    for directory in (ROOT/'data/research',ROOT/'results'):
        for path in sorted(directory.rglob('*')):
            if (not path.is_file() or path.suffix not in ('.json','.jsonl','.gz')
                    or path.is_relative_to(OUT) or 'raw' in path.relative_to(ROOT).parts):continue
            raw=path.read_bytes()
            files[path.relative_to(ROOT).as_posix()]=sha(raw)
            if path.suffix=='.gz':raw=gzip.decompress(raw)
            identities.update(re.findall(r'(?:gsm8k|math500|mbpp|humaneval):\d+',raw.decode('utf-8')))
    for b,n in (('gsm8k',32),('math500',24),('mbpp',19),('humaneval',16)):
        identities.update(f'{b}:{i}' for i in range(n))
    return identities,files


def read_jsonl(path,compressed=False):
    raw=path.read_bytes();raw=gzip.decompress(raw) if compressed else raw
    return [json.loads(line) for line in raw.decode().splitlines()]


def main():
    import pyarrow.parquet as pq
    old=ROOT/'artifacts/asd-evaluation-sources';new=ROOT/'artifacts/asd-paper-sources'
    sources=json.loads((new/'manifest.json').read_text())
    for name,entry in sources.items():
        if sha((new/name).read_bytes())!=entry['sha256']:raise ValueError('Changed downloaded source '+name)
    legacy=json.loads((old/'manifest.json').read_text())
    for b in ('mbpp','humaneval'):
        entry=legacy[b];path=old/entry['filename']
        if sha(path.read_bytes())!=entry['sha256']:raise ValueError('Changed old public source '+b)
        sources[b]=entry
    gsm=ROOT/'artifacts/asd-attribution-gsm-source.jsonl';math=ROOT/'artifacts/asd-budget-math500-source.jsonl'
    if sha(gsm.read_bytes())!='3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14' or sha(math.read_bytes())!='35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132':
        raise ValueError('Changed pinned math source')
    origins=json.loads((ROOT/'data/research/asd-budget/records.json').read_text())['sources']
    sources.update(gsm8k=dict(origins['gsm8k'],sha256=sha(gsm.read_bytes())),math500=origins['math500'])
    pools=dict(gsm8k=[dict(r,id=f'gsm8k:{i}',benchmark='gsm8k',source_index=i) for i,r in enumerate(read_jsonl(gsm))],
        math500=[dict(r,id=f'math500:{i}',benchmark='math500',source_index=i) for i,r in enumerate(read_jsonl(math))],
        humaneval=[dict(r,id='humaneval:'+r['task_id'].split('/')[-1],benchmark='humaneval') for r in read_jsonl(old/'humaneval.jsonl.gz',True)],
        mbpp=[dict(r,id=f"mbpp:{r['task_id']}",benchmark='mbpp') for r in read_jsonl(old/'mbpp.jsonl') if 11<=r['task_id']<=510],
        mmlu=[dict(r,id=f'mmlu:{i}',benchmark='mmlu',source_index=i) for i,r in enumerate(pq.read_table(new/'mmlu.parquet').to_pylist())],
        mtbench=[dict(r,id=f"mtbench:{r['question_id']}",benchmark='mtbench') for r in read_jsonl(new/'mtbench.jsonl')],
        alpaca=[dict(r,id=f'alpaca:{i}',benchmark='alpaca',source_index=i) for i,r in enumerate(json.loads((new/'alpaca.json').read_bytes()))])
    excluded,files=exclusions()
    chosen={};sizes={}
    for b in BENCHMARKS:
        candidates=[r for r in pools[b] if r['id'] not in excluded];sizes[b]=len(candidates)
        if b in ('math500','mmlu','mtbench'):
            key='category' if b=='mtbench' else 'subject'
            selected=balanced(candidates,lambda r:r[key],COUNTS[b])
        elif b=='alpaca':selected=balanced(candidates,lambda r:bool(r['input']),COUNTS[b])
        else:selected=sorted(candidates,key=rank)[:COUNTS[b]]
        if len(selected)!=COUNTS[b]:raise ValueError('Insufficient unused tasks '+b)
        if b=='mbpp':
            selected=[dict(r,signature=mbpp_interface(r)) for r in selected]
        chosen[b]=selected
    OUT.mkdir(parents=True,exist_ok=False)
    hashes={}
    for b,rows in chosen.items():
        raw=(json.dumps(rows,ensure_ascii=False,indent=2)+'\n').encode();name=b+'.json'
        (OUT/name).write_bytes(raw);hashes[name]=sha(raw)
    for source,name in ((new,'mmlu-LICENSE'),(new,'fastchat-LICENSE'),(new,'alpaca-DATA_LICENSE'),
        (new,'fastchat-judge-README.md'),(new,'alpaca-README.md'),(old,'mbpp-LICENSE'),(old,'humaneval-LICENSE'),
        (ROOT/'data/research/asd-suite','GSM8K_LICENSE.txt'),(ROOT/'data/research/asd-suite','MATH_SOURCE_LICENSE.txt')):
        shutil.copyfile(source/name,OUT/name)
    reference=read_jsonl(new/'mtbench-reference.jsonl')
    keep={r['question_id'] for r in chosen['mtbench']}
    (OUT/'mtbench-reference.json').write_bytes((json.dumps([r for r in reference if r['question_id'] in keep],indent=2)+'\n').encode())
    (OUT/'exclusions.json').write_bytes((json.dumps(dict(files=files,excluded_ids=sorted(excluded)),indent=2)+'\n').encode())
    manifest=dict(schema=1,seed=SEED,counts=COUNTS,files=hashes,sources=sources,pool_sizes=sizes,
        ids={b:[r['id'] for r in rows] for b,rows in chosen.items()},excluded_files_sha256=sha((OUT/'exclusions.json').read_bytes()),
        selection='SHA256 rank after conservative exclusion of all prior data/results IDs; subject/category/input-presence round robin; no generated outcome filtering',
        scope='DFlash-based ASD comparison on named paper tasks; not original DSpark/L20 prompt/split reproduction')
    (OUT/'manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
    (OUT/'ATTRIBUTION.md').write_bytes(('# Public research-only benchmark subset\n\nGSM8K/HumanEval/MMLU: MIT notices retained. '
        'MBPP/FastChat repository: Apache-2.0 notices retained; MT-Bench question authors credited to LMSYS/FastChat. '
        'MATH-500 retains original-source MIT notice and source-card qualification. '
        'Alpaca selected data is CC BY-NC 4.0, research/non-commercial only; its data license overrides any project-code license for these records. '
        'Original sources, immutable revisions, hashes and selection are in manifest.json. Generated answers and standard/reference solutions are never retrieval memory.\n').encode())
    print(json.dumps(dict(records=sum(COUNTS.values()),primary_requests=sum(COUNTS.values())+COUNTS['mtbench'],counts=COUNTS,
        excluded=len(excluded),prior_files=len(files),manifest_sha256=sha((OUT/'manifest.json').read_bytes()))))


if __name__=='__main__':main()
