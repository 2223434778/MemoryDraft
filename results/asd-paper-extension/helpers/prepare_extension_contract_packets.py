"""Prepare all planned property cells; display only anonymous manual packets."""
from pathlib import Path
import hashlib,json

from memorydraft.evaluation.asd_extension_contracts import evaluate,load_plan
from memorydraft.evaluation.asd_paper_extension import ARMS,open_units
from memorydraft.evaluation.asd_paper_metrics import message_sha

ROOT=Path.cwd();DATA=ROOT/'results/asd-paper-extension';OUT=ROOT/'artifacts/asd-extension-contract-review'
plan=load_plan(ROOT);units={u['request_id']:u for u in open_units(ROOT,DATA)}
OUT.mkdir(exist_ok=False);(OUT/'packets').mkdir();cells=[];mapping={}
for number,rule in enumerate(plan['rules'],1):
    identity=rule['id'];answers=units[rule['task_id']]['answers']
    questions=[m['content'] for m in next(iter(answers.values())) if m['role']=='user'][:rule['turn']]
    packet=dict(rule=rule,questions=questions,answers={})
    order=sorted(ARMS,key=lambda a:hashlib.sha256(('MemoryDraft-contract-label-v1:'+identity+':'+a).encode()).hexdigest())
    mapping[identity]={label:arm for label,arm in zip('ABCD',order)}
    for label,arm in zip('ABCD',order):
        messages=answers[arm];index=2*rule['turn']-1;text=messages[index]['content']
        digest=hashlib.sha256(text.encode()).hexdigest();result=evaluate(rule,text)
        cells.append(dict(property_id=identity,task_id=rule['task_id'],turn=rule['turn'],check=rule['check'],arm=arm,
            answer_sha256=digest,messages_sha256=message_sha(messages[:index+1]),**result))
        packet['answers'][label]=dict(text=text,answer_sha256=digest,automatic_verdict=result['verdict'],
            prior_answer=messages[1]['content'] if rule['turn']==2 else None)
    (OUT/'packets'/f'{number:02d}.json').write_bytes((json.dumps(packet,ensure_ascii=False,indent=2)+'\n').encode())
(OUT/'cells-private.json').write_bytes((json.dumps(cells,ensure_ascii=False,indent=2)+'\n').encode())
(OUT/'mapping-private.json').write_bytes((json.dumps(mapping,indent=2)+'\n').encode())
metadata=dict(plan_sha256=hashlib.sha256((ROOT/'configs/asd-extension-contracts.json').read_bytes()).hexdigest(),
    check_module_sha256=hashlib.sha256((ROOT/'memorydraft/evaluation/asd_extension_contracts.py').read_bytes()).hexdigest(),
    packet_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    primary_generation_sources=['6d89480c52e0d89714778668c25ef7027a9f0ac8','1d39c56e5a81c896077258433b476af1a934ce6f'],
    check_plan_commit='7cf0daf',property_cells=len(cells),manual_cells=sum(r['check']=='manual' for r in cells),
    automatic_cells=sum(r['check']!='manual' for r in cells),
    scope='Explicit partial properties; method labels hidden in displayed packets. Agent-assisted review, not independent human quality grading.')
(OUT/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
assert metadata['property_cells']==136 and metadata['manual_cells']==100 and metadata['automatic_cells']==36
print(json.dumps(metadata))
