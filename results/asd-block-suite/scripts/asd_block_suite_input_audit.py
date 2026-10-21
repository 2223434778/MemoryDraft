from pathlib import Path
import hashlib,json,os,sys
ROOT=Path('/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-block-suite')
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from transformers import AutoTokenizer
from memorydraft.evaluation.asd_suite import request_for
from memorydraft.memory.store import Snapshot
from scripts.run_asd_block_suite import records_and_positions
model=ROOT.parent/'models/qwen3-8b';tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
records,positions=records_and_positions();checks=[];rows=[];mechanism=[]
for i in (0,1):
 directory=ROOT/f'results/raw/asd-block-suite/part-{i}'
 p=json.loads((directory/'provenance.json').read_text());assert p['status']=='complete'
 rows.extend(json.loads(s) for s in (directory/'rows.jsonl').read_text().splitlines())
 mechanism.extend(json.loads(s) for s in (directory/'mechanism.jsonl').read_text().splitlines())
assert len(rows)==3120 and len(mechanism)==144
for record in records:
 req=request_for(record)
 ids=tokenizer.apply_chat_template(list(req.messages),add_generation_prompt=True,enable_thinking=False)
 prompt=hashlib.sha256(json.dumps(ids).encode()).hexdigest();snapshot=Snapshot(req.user_id,req.order,()).snapshot_id
 selected=[r for r in rows if r['request_id']==record['id']]
 events=[r for r in mechanism if r['request_id']==record['id']]
 assert len(selected)==65 and len(events)==(12 if positions[record['id']]<2 else 0)
 assert all(r['prompt_sha256']==prompt and r['snapshot_id']==snapshot and r['prompt_tokens']==len(ids) for r in selected+events),record['id']
 checks.append(dict(request_id=record['id'],benchmark=record['benchmark'],prompt_tokens=len(ids),prompt_sha256=prompt,snapshot_id=snapshot,timed_rows=len(selected),mechanism_rows=len(events)))
result=dict(status='complete',split='development',timed_rows=len(rows),mechanism_rows=len(mechanism),questions=len(checks),checks=checks,
 tokenizer_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(model.glob('*token*')) if p.is_file()},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 note='CPU-only postrun tokenization. No validation/reserve model generations.')
(ROOT.parent/'asd-block-suite-input-audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(status='complete',timed_rows=len(rows),mechanism_rows=len(mechanism),questions=len(checks))))
