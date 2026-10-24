"""Question-defined partial properties; never replace task or reward scores."""
import hashlib
import json
from pathlib import Path
import re


def question_sha(record):
    payload=dict(id=record['id'],benchmark=record['benchmark'])
    payload['questions']=record['turns'] if record['benchmark']=='mtbench' else [record['instruction'],record['input']]
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def json_payloads(text):
    decoder=json.JSONDecoder();values=[];offset=0
    while match:=re.search(r'[\[{]',text[offset:]):
        start=offset+match.start()
        try:value,end=decoder.raw_decode(text[start:])
        except (json.JSONDecodeError,RecursionError):
            # A damaged outer object must not pass through one of its nested arrays.
            break
        if isinstance(value,(list,dict)):values.append(value)
        offset=start+end
    return values


def evaluate(rule,text):
    """Only declared structural properties receive an automatic verdict."""
    kind=rule['check'];details={}
    if kind=='manual':return dict(verdict='needs_review',criterion=rule['criterion'])
    if kind=='json_groups':
        ok=any(isinstance(v,dict) and len(v)==3 and all(isinstance(xs,list) and
            all(isinstance(x,str) for x in xs) for xs in v.values()) for v in json_payloads(text))
    elif kind=='json_integer_ratings':
        ok=any(isinstance(v,list) and len(v)==3 and all(type(x) is int and 1<=x<=5 for x in v)
               for v in json_payloads(text))
    elif kind=='json_years':
        ok=any(all(str(year) in json.dumps(v) for year in rule['years']) for v in json_payloads(text))
    elif kind=='single_line_json':
        details['nontrimmed_lines']=len(text.strip().splitlines())
        ok=bool(json_payloads(text)) and details['nontrimmed_lines']==1
    elif kind=='gendered_pronouns':
        details['matches']=re.findall(r'\b(?:he|him|his|she|her|hers|himself|herself)\b',text,re.I)
        ok=not details['matches']
    elif kind=='word_limit':
        # Fixed lexical convention includes numeric words, contractions and hyphenated words.
        details['lexical_words']=len(re.findall(r"\b\w+(?:['’-]\w+)*\b",text))
        details['whitespace_tokens']=len(text.split())
        ok=details['lexical_words']<=rule['maximum']
    elif kind=='redaction_literals':
        details['leaked_labels']=[label for label,pattern in rule['patterns'].items() if re.search(pattern,text,re.I)]
        ok=not details['leaked_labels']
    elif kind=='section_order':
        matches=list(re.finditer(r'(?im)^[\s#*>-]*(?:\*\*)?(header|body|footer)\b',text))
        matches+=list(re.finditer(r'(?i)\b(header|body|footer)(?:\*\*)?\s*:',text))
        labels=[m[1].lower() for m in sorted(matches,key=lambda m:m.start())]
        if not labels:
            for payload in json_payloads(text):
                if isinstance(payload,dict):labels.extend(str(key).lower() for key in payload)
        wanted=iter(('header','body','footer'));next_label=next(wanted);ok=False
        for label in labels:
            if label==next_label:
                next_label=next(wanted,None)
                if next_label is None:ok=True;break
        details['section_labels']=labels
    else:raise ValueError('Unknown property check: '+kind)
    return dict(verdict='pass' if ok else 'fail',details=details)


def load_plan(root):
    from memorydraft.evaluation.asd_paper_extension import load_records
    plan=json.loads((Path(root)/'configs/asd-extension-contracts.json').read_bytes())
    if plan['schema']!=1 or plan['study']!='asd-paper-extension':raise ValueError('Wrong property protocol')
    records={r['id']:r for r in load_records(root)}
    if len({r['id'] for r in plan['rules']})!=len(plan['rules']):raise ValueError('Duplicate property ID')
    for rule in plan['rules']:
        if rule['task_id'] not in records or question_sha(records[rule['task_id']])!=rule['question_sha256']:
            raise ValueError('Property question changed')
        if rule['turn'] not in ((1,2) if records[rule['task_id']]['benchmark']=='mtbench' else (1,)):
            raise ValueError('Wrong property turn')
    return plan
