"""Persist anonymous decisions without opening the method mapping."""
import json
from pathlib import Path


def record(decisions):
    directory=Path.cwd()/'artifacts/asd-extension-contract-review';path=directory/'reviews.json'
    rows=json.loads(path.read_bytes()) if path.exists() else []
    keys={(r['property_id'],r['label']) for r in rows}
    for number,labels in decisions.items():
        packet=json.loads((directory/'packets'/f'{int(number):02d}.json').read_bytes())
        for label,(verdict,reason) in labels.items():
            key=packet['rule']['id'],label
            assert key not in keys and label in packet['answers'] and verdict in ('pass','fail','uncertain') and reason.strip()
            rows.append(dict(property_id=key[0],label=label,verdict=verdict,reason=reason,
                answer_sha256=packet['answers'][label]['answer_sha256']))
            keys.add(key)
    path.write_bytes((json.dumps(rows,ensure_ascii=False,indent=2)+'\n').encode())
    print(json.dumps(dict(recorded=len(rows))))
