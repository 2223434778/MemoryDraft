"""Run only after anonymous property review is finalized; preference is not accuracy."""
from collections import Counter
import gzip,hashlib,json,statistics
from pathlib import Path

from memorydraft.evaluation.asd_paper_extension import ARMS,open_units
from memorydraft.evaluation.asd_paper_metrics import message_sha

ROOT=Path.cwd();OUT=ROOT/'results/asd-paper-extension'
review=OUT/'open-contract-audit.json';assert review.exists(),'Finish anonymous property review before inspecting reward cases'
rows=[json.loads(line) for line in gzip.decompress((OUT/'reward/rows.jsonl.gz').read_bytes()).splitlines()]
indexed={(r['request_id'],r['arm']):r for r in rows};units=open_units(ROOT,OUT)
assert len(rows)==len(indexed)==320
cases=[];summaries=[]
for benchmark in ('mtbench','alpaca'):
    selected=[u for u in units if u['benchmark']==benchmark]
    for arm in ARMS:
        counter=Counter();pairs=[];differences=[];a_lengths=[];b_lengths=[]
        for unit in selected:
            identity=unit['request_id'];a=indexed[identity,arm];b=indexed[identity,'ASD']
            assert a['messages_sha256']==message_sha(unit['answers'][arm])
            assert b['messages_sha256']==message_sha(unit['answers']['ASD'])
            if a['score'] is None or b['score'] is None:
                counter['unscored']+=1;continue
            identical=a['messages_sha256']==b['messages_sha256'];delta=a['score']-b['score']
            verdict='tie' if identical or abs(delta)<=1e-6 else 'win' if delta>0 else 'loss'
            counter[verdict]+=1;counter['identical']+=int(identical);pairs.append(a['score']);differences.append(delta)
            text=lambda messages:'\n'.join(m['content'] for m in messages if m['role']=='assistant')
            left,right=text(unit['answers'][arm]),text(unit['answers']['ASD'])
            a_lengths.append(len(left));b_lengths.append(len(right))
            cases.append(dict(request_id=identity,benchmark=benchmark,arm=arm,baseline='ASD',
                verdict=verdict,score=a['score'],baseline_score=b['score'],difference=delta,identical_messages=identical,
                messages_sha256=a['messages_sha256'],baseline_messages_sha256=b['messages_sha256'],
                assistant_characters=len(left),baseline_assistant_characters=len(right),
                prompt_tokens=a['prompt_tokens'],baseline_prompt_tokens=b['prompt_tokens']))
        count=len(pairs)
        summaries.append(dict(benchmark=benchmark,arm=arm,requests=len(selected),scored_pairs=count,
            wins=counter['win'],ties=counter['tie'],losses=counter['loss'],unscored=counter['unscored'],identical=counter['identical'],
            preference_rate=(counter['win']+.5*counter['tie'])/count if count else None,
            mean_score=statistics.mean(pairs) if pairs else None,mean_difference=statistics.mean(differences) if differences else None,
            total_assistant_characters=sum(a_lengths),baseline_assistant_characters=sum(b_lengths)))
result=dict(scope='Independent trained reward preference proxy; not correctness, human rating, official MT-Bench or AlpacaEval.',
    property_review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(),tie_tolerance=1e-6,
    comparison='Actual own-history two-turn MT conversations; Alpaca full single-turn answers',summary=summaries,cases=cases)
path=OUT/'reward-analysis.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summaries))
