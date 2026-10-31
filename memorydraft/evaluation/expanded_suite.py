"""Frozen new-task requests and offline quality evaluation contracts."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from memorydraft.runtime.contracts import GenerationRequest
from memorydraft.evaluation.asd_suite import extract_code, score as suite_score

ROOT=Path(__file__).resolve().parents[2]
ARMS=('DFlash','ASD','BufferedKV-ASD')


def load_records(root=ROOT):
    directory=Path(root)/'data/research/expanded-suite'
    manifest=json.loads((directory/'manifest.json').read_bytes())
    raw=(directory/'tasks.json.gz').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['tasks_sha256']:
        raise ValueError('Changed expanded suite')
    rows=json.loads(gzip.decompress(raw))
    if len(rows)!=manifest['count'] or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Invalid expanded task IDs')
    return rows


def request_for(record,limit):
    messages=tuple(record['messages']) if 'messages' in record else ({'role':'user','content':record['prompt']},)
    return GenerationRequest(record['id'],'expanded-'+record['id'],1,messages,limit)


def validate_generated_code(code):
    allowed={'math','cmath','collections','itertools','functools','re','heapq','bisect','string','typing',
        'statistics','operator','decimal','fractions','copy','random','sys','numpy','dataclasses','json',
        'enum','datetime','time','abc','unittest','pytest','numbers','array','queue','sortedcontainers'}
    forbidden={'open','eval','exec','compile','globals','locals','vars','getattr','setattr','delattr','breakpoint','__import__'}
    for n in ast.walk(ast.parse(code)):
        if isinstance(n,ast.Import) and any(a.name.split('.')[0] not in allowed for a in n.names):
            raise ValueError('unsupported import')
        if isinstance(n,ast.ImportFrom) and (n.level or not n.module or n.module.split('.')[0] not in allowed):
            raise ValueError('unsupported import')
        if isinstance(n,ast.Name) and (n.id in forbidden or (n.id.startswith('__') and n.id!='__name__')):
            raise ValueError('unsupported builtin')
        if isinstance(n,ast.Attribute) and (n.attr.startswith('__') or n.attr in ('modules','system','popen','load','save','fromfile','tofile')):
            raise ValueError('unsupported attribute')


def score(record,text):
    if record['benchmark']=='aime25':
        return suite_score(dict(record,benchmark='math500'),text)
    payload=dict(record=record,text=text)
    try:
        p=subprocess.run([sys.executable,'-m','scripts.score_expanded_task'],input=json.dumps(payload),
            text=True,capture_output=True,cwd=ROOT,timeout=70)
        lines=p.stdout.splitlines()
        if p.returncode or not lines:
            return dict(correct=None,evaluation_error=p.stderr[-600:] or 'worker failed')
        return json.loads(lines[-1])
    except subprocess.TimeoutExpired:
        return dict(correct=False,error='global evaluation timeout',metric='bounded test execution')
