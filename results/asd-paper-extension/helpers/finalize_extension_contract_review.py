"""Bind every review to its displayed answer and preserve automatic verdicts."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json

ROOT=Path.cwd();OUT=ROOT/'artifacts/asd-extension-contract-review'
metadata=json.loads((OUT/'metadata.json').read_bytes())
assert metadata['plan_sha256']==hashlib.sha256((ROOT/'configs/asd-extension-contracts.json').read_bytes()).hexdigest()
assert metadata['check_module_sha256']==hashlib.sha256((ROOT/'memorydraft/evaluation/asd_extension_contracts.py').read_bytes()).hexdigest()
cells=json.loads((OUT/'cells-private.json').read_bytes());mapping=json.loads((OUT/'mapping-private.json').read_bytes())
review=json.loads((OUT/'reviews.json').read_bytes());indexed={}
for row in review:
    key=row['property_id'],mapping[row['property_id']][row['label']]
    assert key not in indexed and row['verdict'] in ('pass','fail','uncertain') and row['reason'].strip()
    indexed[key]=row
expected={(row['property_id'],row['arm']) for row in cells}
assert set(indexed)<=expected
for row in cells:
    key=row['property_id'],row['arm'];selected=indexed.get(key)
    if row['check']=='manual':assert selected is not None,key
    if selected:
        assert selected['answer_sha256']==row['answer_sha256'],key
        row.update(raw_automatic_verdict=row['verdict'],verdict=selected['verdict'],agent_review=selected)
    elif row['verdict']=='fail':
        raise AssertionError('Automatic failures must also receive an explicit checker-validity review: '+str(key))
groups=defaultdict(Counter)
for row in cells:groups[row['task_id'].split(':')[0],row['arm']][row['verdict']]+=1
result=dict(metadata=metadata,reviewer='Agent-assisted anonymous explicit-property review; not human expert, official MT-Bench or AlpacaEval.',
    rows=cells,summary=[dict(benchmark=b,arm=a,**counts) for (b,a),counts in sorted(groups.items())],
    primary_scores_unchanged=True,scope='Counts refer to selected properties, not full-answer accuracy or independent samples.')
target=ROOT/'results/asd-paper-extension/open-contract-audit.json'
assert not target.exists();target.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(property_cells=len(cells),manual_review_cells=len(indexed),summary=result['summary'])))
