"""Frozen seven-task study on one shared DFlash/Target execution stack."""
from dataclasses import replace
from collections import defaultdict
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from memorydraft.evaluation import asd_suite
from memorydraft.evaluation.asd_comparison import bootstrap
from memorydraft.evaluation.asd_joint_tree import audit_allocation
from memorydraft.runtime.contracts import GenerationRequest

STUDY='asd-paper'
SEED='MemoryDraft-asd-paper-v1'
BENCHMARKS=('gsm8k','math500','humaneval','mbpp','mmlu','mtbench','alpaca')
COUNTS=dict(gsm8k=64,math500=32,humaneval=32,mbpp=32,mmlu=57,mtbench=8,alpaca=16)
ARMS=('DFlash','ASD','BudgetTree','BudgetTree-ASD','JointTree-ASD')
REPEATS=dict(natural=2,prefix=1,fixed=1)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def load_records(root):
    directory=Path(root)/'data/research/asd-paper'
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    rows=[]
    if manifest['schema']!=1 or manifest['seed']!=SEED or manifest['counts']!=COUNTS:
        raise ValueError('Changed frozen paper suite')
    for benchmark in BENCHMARKS:
        name=benchmark+'.json';raw=(directory/name).read_bytes()
        selected=json.loads(raw)
        if (sha(raw)!=manifest['files'][name] or len(selected)!=COUNTS[benchmark]
                or any(r['benchmark']!=benchmark for r in selected)):
            raise ValueError('Changed frozen benchmark: '+benchmark)
        rows.extend(selected)
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate paper task')
    return rows


def expanded_ids(rows):
    return [r['id']+f':turn{turn}' if r['benchmark']=='mtbench' else r['id']
            for r in rows for turn in ((1,2) if r['benchmark']=='mtbench' else (1,))]


def mbpp_interface(record):
    definitions={node.name:node for node in ast.parse(record['code']).body if isinstance(node,ast.FunctionDef)}
    calls={node.func.id for test in record['test_list'] for node in ast.walk(ast.parse(test))
           if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
    required=[node for name,node in definitions.items() if name in calls]
    if not required:raise ValueError('MBPP function interface is ambiguous')
    return '\n'.join('def '+node.name+'('+ast.unparse(node.args)+'):' for node in required)


def request_for(record,limit):
    benchmark=record['benchmark']
    if benchmark in ('gsm8k','math500','humaneval'):
        old=asd_suite.request_for(record,limit)
        return replace(old,user_id='paper-'+record['id'])
    if benchmark=='mbpp':
        prompt=('Write the Python function for the following task. Return the complete implementation '
                'with its function signature, without tests or explanation.\n\n'+record['text']+
                '\n\nRequired function interface:\n'+record['signature'])
    elif benchmark=='mmlu':
        prompt=('Answer the multiple-choice question. Return only one letter: A, B, C, or D.\n\n'
            +record['question']+'\n'+'\n'.join(f'{letter}. {choice}' for letter,choice in zip('ABCD',record['choices'])))
    elif benchmark=='alpaca':
        prompt=record['instruction']+('\n\nInput:\n'+record['input'] if record['input'] else '')
    elif benchmark=='mtbench':
        if 'messages' not in record:raise ValueError('MT-Bench requires explicit realized history')
        return GenerationRequest(record['id'],'paper-'+record['conversation_id'],record['turn'],
                                 tuple(record['messages']),limit)
    else:raise ValueError('Unknown paper task')
    return GenerationRequest(record['id'],'paper-'+record['id'],1,({'role':'user','content':prompt},),limit)


def conversation_record(record,turn,first_answer=None):
    if record['benchmark']!='mtbench' or turn not in (1,2):raise ValueError('Invalid conversation turn')
    if turn==2 and first_answer is None:raise ValueError('Second turn needs actual first answer')
    messages=[dict(role='user',content=record['turns'][0])]
    if turn==2:messages += [dict(role='assistant',content=first_answer),dict(role='user',content=record['turns'][1])]
    return dict(record,id=record['id']+f':turn{turn}',conversation_id=record['id'],turn=turn,messages=messages)


def score(record,text):
    benchmark=record['benchmark']
    if benchmark in ('gsm8k','math500','humaneval'):
        result=asd_suite.score(record,text)
        if benchmark=='humaneval':result['metric']='HumanEval original tests withheld from prompt, single greedy completion'
        return result
    if benchmark=='mbpp':
        result=asd_suite.code_score(record,text)
        result['metric']='MBPP original tests withheld from prompt, single greedy completion'
        checks=record.get('challenge_test_list',[])
        if checks:
            extended=asd_suite.code_score(dict(record,test_list=record['test_list']+checks),text)
            result.update(challenge_correct=extended['correct'],challenge_error=extended['error'])
        return result
    if benchmark=='mmlu':
        match=re.fullmatch(r'\s*\(?([ABCD])\)?[.)]?\s*',text)
        expected='ABCD'[record['answer']]
        return dict(correct=bool(match and match[1]==expected),parsed_answer=match[1] if match else None,
                    expected_answer=expected,metric='zero-shot generated choice exact match')
    if benchmark in ('mtbench','alpaca'):
        return dict(correct=None,metric='open response; separate blind pairwise local-judge audit')
    raise ValueError('Unknown paper scorer')


def canonical(record):
    if record['benchmark']=='mmlu':return 'ABCD'[record['answer']]
    if record['benchmark'] in ('mtbench','alpaca'):return None
    return asd_suite.canonical(record)


def stages(record):
    if record['benchmark'] not in BENCHMARKS:raise ValueError('Unknown paper task')
    return ('natural','prefix','fixed')


def quality_ok(quality):
    return quality.get('secondary_correct',quality['correct']) if quality['correct'] is not None else None


def aggregate(rows,references,records,*,arms=ARMS,repeats=REPEATS):
    refs={r['request_id']:r for r in references}
    expected=set(expanded_ids(records))
    if len(refs)!=len(references) or set(refs)!=expected:raise ValueError('Incomplete AR coverage')
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows:groups[row['request_id']][row['stage'],row['arm']].append(row)
    if set(groups)!=expected:raise ValueError('Incomplete task coverage')
    audit=audit_allocation(rows)
    samples=[]
    for identity,cells in sorted(groups.items()):
        ref=refs[identity];benchmark=ref['benchmark']
        selected_stages=stages(dict(benchmark=benchmark))
        if set(cells)!={(s,a) for s in selected_stages for a in arms}:raise ValueError('Incomplete method matrix')
        item=dict(request_id=identity,benchmark=benchmark,arms={})
        for (stage,arm),entries in sorted(cells.items()):
            entries.sort(key=lambda r:r['repeat'])
            if [r['repeat'] for r in entries]!=list(range(repeats[stage])):raise ValueError('Incomplete repeats')
            first=entries[0]
            signature=lambda r:(r['token_ids'],r['rounds'],r['asd'],r['finish_reason'],r['text'])
            if any(signature(r)!=signature(first) for r in entries):raise ValueError('Unstable trajectory')
            if any((r['prompt_sha256'],r['snapshot_id'])!=(ref['prompt_sha256'],ref['snapshot_id']) for r in entries):
                raise ValueError('Target input mismatch')
            if any(not math.isfinite(r['latency_seconds']) or r['latency_seconds']<=0 for r in entries):
                raise ValueError('Invalid latency')
            if first['asd'] and not 0<=first['asd']['spent']<=8:raise ValueError('Budget exceeded')
            if stage=='fixed':
                if len(first['token_ids'])!=256:raise ValueError('Wrong fixed workload')
                if arm=='BudgetTree' and first['token_ids']!=cells[stage,'DFlash'][0]['token_ids']:
                    raise ValueError('Strict fixed tree differs from DFlash')
            elif arm in ('DFlash','BudgetTree'):
                limit=4096 if stage=='natural' else 128
                if first['token_ids']!=ref['token_ids'][:limit]:raise ValueError('Strict method differs from AR')
                if stage=='natural' and (first['text'],first['finish_reason'])!=(ref['text'],ref['finish_reason']):
                    raise ValueError('Strict natural result differs from AR')
            seconds=statistics.mean(r['latency_seconds'] for r in entries)
            blocks=first['rounds']
            item['arms'][stage+'/'+arm]=dict(seconds=seconds,tokens=len(first['token_ids']),calls=len(blocks),
                nodes=sum(b.get('tree_nodes',b['proposed']+1) for b in blocks),
                spent=first['asd']['spent'] if first['asd'] else 0.,
                relaxed=first['asd']['relaxed_tokens'] if first['asd'] else 0,
                correct=quality_ok(first['quality']) if stage=='natural' else None,
                quality=first.get('quality'),finish_reason=first['finish_reason'],
                repeat_seconds=[r['latency_seconds'] for r in entries],
                mean_ttft_seconds=statistics.mean(r['ttft_seconds'] for r in entries if r['ttft_seconds'] is not None)
                    if first['ttft_seconds'] is not None else None)
        samples.append(item)
    families={}
    for benchmark in BENCHMARKS:
        selected=[s for s in samples if s['benchmark']==benchmark]
        if not selected:continue
        families[benchmark]={}
        for arm in arms:
            family={}
            for stage in stages(dict(benchmark=benchmark)):
                items=[s['arms'][stage+'/'+arm] for s in selected]
                tokens=sum(v['tokens'] for v in items);seconds=sum(v['seconds'] for v in items)
                calls=sum(v['calls'] for v in items);nodes=sum(v['nodes'] for v in items)
                stats=dict(requests=len(items),tokens=tokens,seconds=seconds,tps=tokens/seconds,
                    calls=calls,nodes=nodes,tokens_per_call=tokens/calls if calls else None,
                    nodes_per_token=nodes/tokens if tokens else None,spent=sum(v['spent'] for v in items),
                    relaxed=sum(v['relaxed'] for v in items))
                if stage=='natural':
                    known=[v for v in items if v['correct'] is not None]
                    stats.update(scored=len(known),correct=sum(v['correct'] for v in known),
                        primary_correct=sum(v['quality']['correct'] for v in known),
                        needs_review=sum(v['quality'].get('needs_review',False) for v in known),
                        truncated=sum(v['finish_reason']=='length' for v in items),
                        baseline_repeat_ratio=statistics.mean(v['repeat_seconds'][1]/v['repeat_seconds'][0] for v in items))
                    challenge=[v['quality']['challenge_correct'] for v in known if 'challenge_correct' in v['quality']]
                    stats.update(challenge_scored=len(challenge),challenge_correct=sum(challenge))
                for baseline in ('DFlash','ASD','BudgetTree','BudgetTree-ASD'):
                    all_pairs=[(s,s['arms'][stage+'/'+arm],s['arms'][stage+'/'+baseline]) for s in selected]
                    pairs=[(s,a,b) for s,a,b in all_pairs if stage!='prefix' or a['tokens']==b['tokens']==128]
                    ratios=[a['seconds']/b['seconds'] for _,a,b in pairs]
                    stats[baseline]=dict(n=len(pairs),ratio=statistics.mean(ratios) if ratios else None,
                        ci95=bootstrap(ratios) if ratios else None,
                        length_ratio=statistics.mean(a['tokens']/b['tokens'] for _,a,b in pairs) if pairs else None)
                    if stage=='natural':stats[baseline].update(
                        lost=[s['request_id'] for s,a,b in all_pairs if b['correct'] is True and a['correct'] is False],
                        gained=[s['request_id'] for s,a,b in all_pairs if a['correct'] is True and b['correct'] is False])
                family[stage]=stats
            families[benchmark][arm]=family
    return dict(timed_generations=len(rows),ar_generations=len(references),samples=samples,
                families=families,allocation_audit=audit)
