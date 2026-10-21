"""Postrun diagnostics on frozen public data; no decoding or score changes."""
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import ast,hashlib,json
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.evaluation.asd_budget import normalize_answer
from memorydraft.evaluation.asd_suite import load_suite
from scripts.diagnose_transport_online import write
base=Path('results/asd-block-suite');out=base/'diagnostics'
out.mkdir(parents=True,exist_ok=True)
rows=sum((read_rows(base/f'part-{i}/rows.jsonl') for i in (0,1)),[])
mechanism=sum((read_rows(base/f'part-{i}/mechanism.jsonl') for i in (0,1)),[])
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for i in (0,1) for p in (base/f'part-{i}/rows.jsonl.gz',base/f'part-{i}/mechanism.jsonl.gz')}
strict={(r['request_id'],r['experimental_block_size']):r for r in rows if r['method']=='DFlash' and r['stage']=='natural' and r['repeat']==0}
states=[];excluded=[]
for q in dict.fromkeys(q for q,s in strict):
 a=strict[q,16];native={r['generated_before']:r for r in a['rounds']}
 for size in (24,32):
  z=strict[q,size]
  if a['token_ids']!=z['token_ids']:
   excluded.append(dict(request_id=q,size=size,reason='strict full token trajectory differs'));continue
  for rd in z['rounds']:
   old=native.get(rd['generated_before'])
   if old is None:continue
   h=min(old['proposed'],rd['proposed'],len(a['token_ids'])-rd['generated_before'])
   x,y=min(old['accepted'],h),min(rd['accepted'],h)
   states.append(dict(request_id=q,benchmark=a['benchmark'],size=size,generated_before=rd['generated_before'],useful_common_horizon=h,
                      native_accepted_within_horizon=x,long_accepted_within_horizon=y,delta=y-x))
observed={(r['request_id'],r['experimental_block_size']):r for r in mechanism if r['method']=='DFlash'}
candidates=[]
for q in dict.fromkeys(q for q,s in observed):
 a=observed[q,16];native={rd['generated_before']:(rd,e) for rd,e in zip(a['rounds'],a['events'])}
 for size in (24,32):
  z=observed[q,size]
  if a['token_ids']!=z['token_ids']:continue
  for rd,e in zip(z['rounds'],z['events']):
   if rd['generated_before'] not in native:continue
   old,oe=native[rd['generated_before']]
   h=min(old['proposed'],rd['proposed'],len(a['token_ids'])-rd['generated_before'])
   p,t=oe['proposal_token_ids'][:h],e['proposal_token_ids'][:h]
   candidates.append(dict(request_id=q,size=size,generated_before=rd['generated_before'],useful_common_horizon=h,
                          changed_positions=sum(x!=y for x,y in zip(p,t)),
                          first_difference=next((i for i,(x,y) in enumerate(zip(p,t)) if x!=y),None)))
summary={str(size):dict(tasks=len({r['request_id'] for r in states if r['size']==size}),matched_states=sum(r['size']==size for r in states),
                       worse=sum(r['size']==size and r['delta']<0 for r in states),better=sum(r['size']==size and r['delta']>0 for r in states),equal=sum(r['size']==size and r['delta']==0 for r in states),
                       observed_candidate_states=sum(r['size']==size for r in candidates),candidate_changed_states=sum(r['size']==size and r['changed_positions']>0 for r in candidates)) for size in (24,32)}
write(out/'same-prefix.json',dict(scope='Postrun exploratory matched-prefix diagnosis, separate from frozen paired comparisons',sources=sources,
 limitations='Only matched committed token prefixes. Cached hidden features are not checked bitwise. Within-question states are dependent; counts are not independent samples or causal training-length attribution. Candidate IDs available only on the 12 preselected mechanism questions. EOS tail clipped to useful common horizon.',
 summary=summary,excluded=excluded,timed_states=states,candidate_states=candidates))
initial=defaultdict(list)
for r in mechanism:initial[r['request_id'],r['experimental_block_size']].append(r)
initial_checks=[]
for (q,size),items in initial.items():
 es=[r['events'][0] for r in items]
 matched=all((e['proposal_token_ids'],e['target_top_token_ids'],e['regrets'])==(es[0]['proposal_token_ids'],es[0]['target_top_token_ids'],es[0]['regrets']) for e in es)
 initial_checks.append(dict(request_id=q,size=size,matched_proposals_top_and_regrets=matched,
                           choices={r['method']:{k:r['events'][0][k] for k in ('accepted','strict_accepted','feasible_accepted','relaxed','regret')} for r in items}))
write(out/'initial-policy-states.json',dict(scope='Postrun initial-block policy comparison only; no causal attribution of later latency',sources=sources,states=initial_checks))
records={r['id']:r for r in load_suite('data/research/asd-suite')}
case444=[]
values={'frac{(k^2+1)^2}{8k}':Fraction(25,6),'frac{k^2+1}{2k}':Fraction(5,3),'frac{1}{k(1+k^2)}':Fraction(1,30),
        'frac{k}{4}':Fraction(3,4),'frac{k^2+1}{8k}':Fraction(5,12),'frac{k^2+1}{4k}':Fraction(5,6)}
for r in rows:
 if r['request_id']!='math500:444' or r['stage']!='natural' or r['repeat']!=0:continue
 actual=r['quality']['parsed_answer'];value=None
 if actual is not None:
  key=normalize_answer(actual).replace(chr(92),'');assert key in values,key
  value=values[key];assert value!=Fraction(1,2)
 assert not r['quality']['correct']
 case444.append(dict(method=r['method'],size=r['experimental_block_size'],parsed_answer=actual,finish_reason=r['finish_reason'],primary_correct=False,
                     reviewed_correct=False,counterexample=None if value is None else dict(k=3,expression_value=str(value),expected='1/2')))
code=next(r for r in rows if r['request_id']=='mbpp:504' and r['method']=='ASD' and r['experimental_block_size']==32 and r['stage']=='natural' and r['repeat']==0)
assert 'sum_Of_Series' in records['mbpp:504']['test_list'][0]
names=[n.name for n in ast.parse(code['text']).body if isinstance(n,ast.FunctionDef)]
assert names==['sum_of_series'] and not code['quality']['correct']
write(base/'quality-review.json',dict(scope='Postrun model-assisted manual review, primary scores unchanged',sources=sources,
 reference_data_sha256={b:hashlib.sha256(Path(f'data/research/asd-suite/{b}.json').read_bytes()).hexdigest() for b in ('math500','mbpp')},
 primary_scores_unchanged=True,math444=dict(reference_argument='Four congruent corner rectangles; half-turn in each swaps O and its nearest original vertex, so probability is 1/2 for every positive k.',
 counterexample_rule='Explicit algebraic substitution at k=3 for the six observed expression forms; no arbitrary symbolic evaluator.',checks=case444),
 extra_asd32_failures=[dict(request_id='math500:448',expected=44,actual=49,expected_modulo=17*44%83,actual_modulo=17*49%83,
                          reason='Actual residue gives remainder3, not required1; not a formatting equivalence.'),
                      dict(request_id='mbpp:504',expected_function='sum_Of_Series',actual_functions=names,
                           reason='Provided tests call a case-sensitive function name that the output does not define. Contract failure; no renaming or relaxed grading applied.')]))
print(json.dumps(dict(matched_prefix=summary,initial_states=len(initial_checks),initial_matched=sum(s['matched_proposals_top_and_regrets'] for s in initial_checks),math444_reviews=len(case444))))
