"""Descriptive follow-up of the frozen study; never selects tasks or tunes a policy."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_comparison import bootstrap
from memorydraft.evaluation.asd_paper import ARMS,BENCHMARKS
from scripts.write_asd_paper_report import read_rows

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/asd-paper'


def summarize_pairs(pairs):
    ratios=[a['seconds']/b['seconds'] for _,a,b in pairs]
    return dict(n=len(pairs),latency_ratio=statistics.mean(ratios) if ratios else None,
        ci95=bootstrap(ratios) if ratios else None,
        output_ratio=statistics.mean(a['tokens']/b['tokens'] for _,a,b in pairs) if pairs else None,
        call_density_ratio=statistics.mean((a['calls']/a['tokens'])/(b['calls']/b['tokens'])
            for _,a,b in pairs if a['calls'] and b['calls'])
            if any(a['calls'] and b['calls'] for _,a,b in pairs) else None)


def main():
    rows=read_rows(OUT,'rows');saved=json.loads((OUT/'analysis.json').read_bytes())
    primary={(r['request_id'],r['arm']):r for r in rows if r['stage']=='natural' and r['repeat']==0}
    contrasts=[];mechanisms=[];drift=[]
    for benchmark in BENCHMARKS:
        samples=[s for s in saved['samples'] if s['benchmark']==benchmark]
        for arm in ARMS:
            for baseline in ('ASD','BudgetTree-ASD'):
                pairs=[(s,s['arms']['natural/'+arm],s['arms']['natural/'+baseline]) for s in samples]
                eos=[p for p in pairs if p[1]['finish_reason']==p[2]['finish_reason']=='eos']
                correct=[p for p in eos if p[1]['correct'] is p[2]['correct'] is True]
                identical=[p for p in pairs if primary[p[0]['request_id'],arm]['token_ids']==primary[p[0]['request_id'],baseline]['token_ids']]
                contrasts.append(dict(benchmark=benchmark,arm=arm,baseline=baseline,
                    common_natural_eos=summarize_pairs(eos),common_correct_eos=summarize_pairs(correct),
                    identical_output=summarize_pairs(identical)))
            values=[s['arms']['natural/'+arm]['repeat_seconds'][1]/s['arms']['natural/'+arm]['repeat_seconds'][0] for s in samples]
            drift.append(dict(benchmark=benchmark,arm=arm,n=len(values),mean_ratio=statistics.mean(values),
                min_ratio=min(values),max_ratio=max(values),outside_three_percent=sum(not .97<=x<=1.03 for x in values)))
            for stage in ('natural','prefix','fixed'):
                selected=[r for r in rows if r['benchmark']==benchmark and r['arm']==arm and r['stage']==stage and r['repeat']==0]
                blocks=[b for r in selected for b in r['rounds']]
                trees=[b for b in blocks if 'tree_budget' in b]
                joint=[b for b in trees if 'spine_reserved' in b['tree_budget']]
                mechanisms.append(dict(benchmark=benchmark,arm=arm,stage=stage,requests=len(selected),
                    tokens=sum(len(r['token_ids']) for r in selected),calls=len(blocks),
                    nodes=sum(b.get('tree_nodes',b['proposed']+1) for b in blocks),
                    approximate_blocks=sum(b.get('relaxed_tokens',0)>0 for b in blocks),
                    selected_path_extra=sum(b['accepted']-b['strict_accepted'] for b in blocks if 'strict_accepted' in b),
                    mean_tree_depth=statistics.mean(b['tree_budget']['max_depth'] for b in trees) if trees else None,
                    node_cap_blocks=sum(b['tree_nodes']==32 for b in trees),
                    mean_reserved_spine=statistics.mean(b['tree_budget']['spine_reserved'] for b in joint) if joint else None,
                    shortened_blocks=sum(b['tree_budget']['spine_requested']<b['proposed'] for b in joint),
                    zero_regret_requests=sum(not r['asd'] or r['asd']['spent']==0 for r in selected)))
    result=dict(note='Post-hoc descriptive strata. Quality/length selection here never replaces the full frozen primary results. '
        'Repeat0 mechanisms only; repeat timing ratios are not independent prompts or original-paper B-C-B drift.',
        paired_strata=contrasts,mechanisms=mechanisms,repeat_timing=drift)
    (OUT/'mechanism.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(dict(paired_strata=len(contrasts),method_stage_cells=len(mechanisms),timing_cells=len(drift))))


if __name__=='__main__':main()
