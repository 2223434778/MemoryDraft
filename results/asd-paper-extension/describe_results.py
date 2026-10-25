"""Post-hoc length, stopping, dialogue and quality diagnostics; no generation."""
from collections import defaultdict
import hashlib,json,re,statistics
from pathlib import Path

from memorydraft.evaluation.asd_comparison import bootstrap
from memorydraft.evaluation.asd_paper import BENCHMARKS
from memorydraft.evaluation.asd_paper_extension import ARMS,load_records,normalize_fixed,read_rows

ROOT=Path.cwd();OUT=ROOT/'results/asd-paper-extension'


def main():
    analysis=json.loads((OUT/'analysis.json').read_bytes())
    raw=read_rows(OUT,'rows');rows,_=normalize_fixed(raw);refs=read_rows(OUT,'references')
    reference={r['request_id']:r for r in refs}
    selected={(r['request_id'],r['arm']):r for r in rows if r['stage']=='natural' and r['repeat']==0}
    bins=defaultdict(list);input_bins=defaultdict(list);alpaca_input=defaultdict(list)
    tasks={r['id']:r for r in load_records(ROOT)};changed=[];mmlu=[];stopping=[]
    for sample in analysis['samples']:
        identity=sample['request_id'];length=len(reference[identity]['token_ids'])
        bucket='0-32' if length<=32 else '33-128' if length<=128 else '129-512' if length<=512 else '513+'
        baseline=sample['arms']['natural/ASD']
        for arm in ARMS:
            value=sample['arms']['natural/'+arm]
            bins[sample['benchmark'],bucket,arm].append(dict(ratio=value['seconds']/baseline['seconds'],
                tokens=value['tokens'],baseline_tokens=baseline['tokens'],calls=value['calls'],baseline_calls=baseline['calls']))
            prompt=reference[identity]['prompt_tokens'];input_bucket='0-128' if prompt<=128 else '129-512' if prompt<=512 else '513+'
            input_bins[sample['benchmark'],input_bucket,arm].append(value['seconds']/baseline['seconds'])
            if sample['benchmark']=='alpaca':alpaca_input[bool(tasks[identity]['input']),arm].append(value['seconds']/baseline['seconds'])
            row=selected[identity,arm]
            if value['finish_reason']=='length':stopping.append(dict(request_id=identity,arm=arm,
                seconds=value['seconds'],tokens=value['tokens'],quality=row['quality'],text_sha256=hashlib.sha256(row['text'].encode()).hexdigest()))
        if sample['benchmark'] in ('gsm8k','math500','humaneval','mbpp'):
            correct={a:sample['arms']['natural/'+a]['correct'] for a in ARMS}
            if len(set(correct.values()))>1:
                changed.append(dict(request_id=identity,benchmark=sample['benchmark'],answers={a:dict(
                    quality=selected[identity,a]['quality'],tokens=len(selected[identity,a]['token_ids']),
                    finish_reason=selected[identity,a]['finish_reason'],
                    text_sha256=hashlib.sha256(selected[identity,a]['text'].encode()).hexdigest()) for a in ARMS}))
        if sample['benchmark']=='mmlu':
            options={}
            for arm in ARMS:
                text=selected[identity,arm]['text'];match=re.match(r'^\s*\(?([ABCD])\b',text)
                options[arm]=match[1] if match else None
            expected=selected[identity,'ASD']['quality']['expected_answer']
            mmlu.append(dict(request_id=identity,expected=expected,first_choices=options,
                primary_correct={a:selected[identity,a]['quality']['correct'] for a in ARMS}))
    length_stats=[dict(benchmark=b,ar_output_length_bin=bucket,arm=a,requests=len(values),
        natural_ratio=statistics.mean(v['ratio'] for v in values),ci95=bootstrap([v['ratio'] for v in values]),
        output_tokens=sum(v['tokens'] for v in values),baseline_output_tokens=sum(v['baseline_tokens'] for v in values),
        calls=sum(v['calls'] for v in values),baseline_calls=sum(v['baseline_calls'] for v in values))
        for (b,bucket,a),values in sorted(bins.items())]
    conversations=defaultdict(list)
    for sample in analysis['samples']:
        if sample['benchmark']=='mtbench':conversations[sample['request_id'].rsplit(':turn',1)[0]].append(sample)
    rollout=read_rows(OUT,'rollout');own={(r['conversation_id'],r['arm']):r for r in rollout};dialogues=[]
    for arm in ARMS:
        matched=[];realized=[]
        for identity,turns in sorted(conversations.items()):
            assert len(turns)==2
            matched.append(sum(s['arms']['natural/'+arm]['seconds'] for s in turns)/sum(s['arms']['natural/ASD']['seconds'] for s in turns))
            a=own[identity,arm];b=own[identity,'ASD']
            realized.append((a['first_latency_seconds']+a['latency_seconds'])/(b['first_latency_seconds']+b['latency_seconds']))
        dialogues.append(dict(arm=arm,conversations=len(conversations),matched_history_mean_ratio=statistics.mean(matched),
            matched_history_ci95=bootstrap(matched),own_history_mean_ratio=statistics.mean(realized),
            own_history_ci95=bootstrap(realized),note='Both turns clustered by conversation; own history inputs differ.'))
    eos={r['token_ids'][-1] for r in rows if r['stage']=='natural' and r['finish_reason']=='eos'};forced=[]
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            chosen=[r for r in rows if r['benchmark']==benchmark and r['arm']==arm and r['stage']=='fixed']
            ends=[next((i+1 for i,t in enumerate(r['token_ids']) if t in eos),256) for r in chosen]
            forced.append(dict(benchmark=benchmark,arm=arm,requests=len(chosen),after_first_eos_tokens=sum(256-x for x in ends),
                fraction=sum(256-x for x in ends)/(256*len(ends))))
    old=json.loads((ROOT/'results/asd-paper/analysis.json').read_bytes());cohorts=[]
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            phase=[];pooled=[]
            for label,data in (('previous',old),('extension',analysis)):
                samples=[s for s in data['samples'] if s['benchmark']==benchmark]
                ratios=[s['arms']['natural/'+arm]['seconds']/s['arms']['natural/ASD']['seconds'] for s in samples]
                pooled+=ratios;n=data['families'][benchmark][arm]['natural']
                phase.append(dict(cohort=label,requests=len(samples),natural_ratio=statistics.mean(ratios),
                    scored=n['scored'],correct=n['correct'],truncated=n['truncated']))
            cohorts.append(dict(benchmark=benchmark,arm=arm,cohorts=phase,pooled_request_mean_ratio=statistics.mean(pooled),
                scope='Descriptive aggregation of within-request ratios across two frozen cohorts, not a new timing run.'))
    mmlu_summary={a:dict(requests=len(mmlu),leading_choice_correct=sum(v['first_choices'][a]==v['expected'] for v in mmlu),
        leading_choice_parsed=sum(v['first_choices'][a] is not None for v in mmlu),
        primary_correct=sum(v['primary_correct'][a] for v in mmlu)) for a in ARMS}
    factorial=[];held=[]
    for benchmark in BENCHMARKS:
        samples=[s for s in analysis['samples'] if s['benchmark']==benchmark]
        fixed={a:[] for a in ('strict_tree_over_linear','approx_tree_over_linear','interaction')}
        for sample in samples:
            v={a:sample['arms']['fixed/'+a]['seconds'] for a in ARMS}
            fixed['strict_tree_over_linear'].append(v['BudgetTree']/v['DFlash'])
            fixed['approx_tree_over_linear'].append(v['BudgetTree-ASD']/v['ASD'])
            fixed['interaction'].append((v['BudgetTree-ASD']/v['BudgetTree'])/(v['ASD']/v['DFlash']))
        factorial.append(dict(benchmark=benchmark,requests=len(samples),
            ratios={key:dict(mean=statistics.mean(values),ci95=bootstrap(values)) for key,values in fixed.items()},
            scope='Within-request fixed-work factorial ratios. Trajectories can differ; this does not hold intermediate states fixed.'))
        known=[s for s in samples if all(selected[s['request_id'],a]['quality']['correct'] is True
            and s['arms']['natural/'+a]['finish_reason']=='eos' for a in ('ASD','BudgetTree-ASD'))]
        ratios=[s['arms']['natural/BudgetTree-ASD']['seconds']/s['arms']['natural/ASD']['seconds'] for s in known]
        held.append(dict(benchmark=benchmark,requests=len(known),mean_natural_ratio=statistics.mean(ratios) if ratios else None,
            ci95=bootstrap(ratios) if ratios else None,
            scope='Post-hoc intersection of primary-correct EOS outputs; selected with future outcomes, not a deployable policy. Open tasks have no primary correctness labels.'))
    result=dict(note='All diagnostics are post-hoc; original primary results and grades are preserved. AR-length bins use future outcomes and are not a deployable router.',
        length_bins=length_stats,mtbench_dialogues=dialogues,observed_eos_ids=sorted(eos),fixed_post_eos=forced,
        mmlu_leading_choice=mmlu_summary,mmlu_cases=mmlu,quality_changed_cases=changed,truncated_cases=stopping,
        cohorts=cohorts,fixed_factorial=factorial,both_primary_correct_eos=held,
        prompt_length_bins=[dict(benchmark=b,input_tokens=bucket,arm=a,requests=len(values),
            natural_ratio=statistics.mean(values),ci95=bootstrap(values),scope='Post-hoc stratum using input length known at request time; not a validated routing rule')
            for (b,bucket,a),values in sorted(input_bins.items())],
        alpaca_input_presence=[dict(has_input=present,arm=a,requests=len(values),natural_ratio=statistics.mean(values),
            ci95=bootstrap(values),scope='Post-hoc descriptive split of the frozen balanced source selection')
            for (present,a),values in sorted(alpaca_input.items())])
    (OUT/'descriptive.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(dict(length_cells=len(length_stats),quality_changes=len(changed),truncations=len(stopping),
        mmlu=mmlu_summary,mtbench_dialogues=dialogues)))


if __name__=='__main__':main()
