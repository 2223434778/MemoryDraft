from pathlib import Path
import hashlib,json,os,sys
ROOT=Path('/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-suite-stage1')
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from transformers import AutoTokenizer
from memorydraft.evaluation.asd_suite import load_suite,request_for
from memorydraft.memory.store import Snapshot
model=ROOT.parent/'models/qwen3-8b'
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
records=load_suite(ROOT/'data/research/asd-suite'); checks=[]
rows=[]
for p in (ROOT/'results/raw/asd-suite-stage1').glob('part-*/rows.jsonl'):
 rows.extend(json.loads(s) for s in p.read_text().splitlines())
for record in records:
 req=request_for(record)
 ids=tokenizer.apply_chat_template(list(req.messages),add_generation_prompt=True,enable_thinking=False)
 prompt=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
 snapshot=Snapshot(req.user_id,req.order,()).snapshot_id
 selected=[r for r in rows if r['request_id']==record['id']]
 assert len(selected)==25
 assert all(r['prompt_sha256']==prompt and r['snapshot_id']==snapshot and r['prompt_tokens']==len(ids) for r in selected),record['id']
 checks.append(dict(request_id=record['id'],benchmark=record['benchmark'],prompt_tokens=len(ids),prompt_sha256=prompt,snapshot_id=snapshot,verified_rows=len(selected)))
result=dict(status='complete',split='development',timed_rows=len(rows),questions=len(checks),checks=checks,
 tokenizer_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(model.glob('*token*')) if p.is_file()},
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),note='CPU-only postrun tokenization. No validation/reserve model generations.')
out=ROOT.parent/'asd-suite-stage1-input-audit.json';out.write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(status='complete',rows=len(rows),questions=len(checks))))
