"""Bind supplementary property verdicts to the actual published conversations."""
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path

from memorydraft.evaluation.asd_extension_contracts import evaluate,load_plan
from memorydraft.evaluation.asd_paper_extension import ARMS,open_units
from memorydraft.evaluation.asd_paper_metrics import message_sha

ROOT=Path.cwd();OUT=ROOT/'results/asd-paper-extension'
plan=load_plan(ROOT);rules={r['id']:r for r in plan['rules']}
audit=json.loads((OUT/'open-contract-audit.json').read_bytes());cells=audit['rows']
units={u['request_id']:u for u in open_units(ROOT,OUT)}
assert len(cells)==136 and {(r['property_id'],r['arm']) for r in cells}=={(r,a) for r in rules for a in ARMS}
assert audit['metadata']['plan_sha256']==hashlib.sha256((ROOT/'configs/asd-extension-contracts.json').read_bytes()).hexdigest()
assert audit['primary_scores_unchanged']
groups=defaultdict(Counter);manual=auto=0
for cell in cells:
    rule=rules[cell['property_id']];messages=units[rule['task_id']]['answers'][cell['arm']]
    index=2*rule['turn']-1;text=messages[index]['content'];result=evaluate(rule,text)
    assert cell['answer_sha256']==hashlib.sha256(text.encode()).hexdigest()
    assert cell['messages_sha256']==message_sha(messages[:index+1])
    assert cell['check']==rule['check'] and cell['turn']==rule['turn'] and cell['task_id']==rule['task_id']
    raw=cell.get('raw_automatic_verdict',cell['verdict']);assert raw==result['verdict']
    if rule['check']=='manual' or raw=='fail':
        review=cell['agent_review'];assert review['answer_sha256']==cell['answer_sha256']
        assert review['reason'].strip() and review['verdict']==cell['verdict'];manual+=1
    else:auto+=1
    assert cell['verdict'] in ('pass','fail','uncertain')
    groups[cell['task_id'].split(':')[0],cell['arm']][cell['verdict']]+=1
summary=[dict(benchmark=b,arm=a,**counts) for (b,a),counts in sorted(groups.items())]
assert summary==audit['summary']
result=dict(bound_property_cells=len(cells),rules=len(rules),manual_or_automatic_failure_reviews=manual,
    untouched_automatic_passes=auto,conversation_and_answer_hashes_verified=True,primary_scores_unchanged=True,
    scope='Selected explicit properties, not full-answer accuracy; agent-assisted, not human expert judgment')
path=OUT/'contract-binding-audit.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
