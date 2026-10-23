"""Describe allocation behavior; no post-hoc parameter fitting or GPU runs."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_joint_tree import ARMS
from memorydraft.evaluation.asd_comparison import quality_ok
from scripts.write_asd_tree_report import read_rows

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/asd-joint-tree'


def main():
    rows=[r for r in read_rows(OUT,'rows') if r['repeat']==0]
    grouped=defaultdict(list)
    budget_bins=defaultdict(list)
    for row in rows:
        grouped[row['benchmark'],row['arm'],row['stage']].append(row)
        if row['arm']=='JointTree-ASD':
            for block in row['rounds']:
                meta=block['tree_budget']
                left=meta['regret_before']
                state='zero' if left==0 else 'full' if meta['spine_requested']==block['proposed'] else 'partial'
                budget_bins[row['benchmark'],row['stage'],state].append(block)
    summaries=[]
    for (benchmark,arm,stage),selected in sorted(grouped.items()):
        blocks=[b for row in selected for b in row['rounds']]
        tree=[b for b in blocks if 'tree_budget' in b]
        summaries.append(dict(benchmark=benchmark,arm=arm,stage=stage,requests=len(selected),
            tokens=sum(len(r['token_ids']) for r in selected),calls=len(blocks),
            nodes=sum(b.get('tree_nodes',b['proposed']+1) for b in blocks),
            spent=sum(r['asd']['spent'] for r in selected if r['asd']),
            extra=sum(b['accepted']-b['strict_accepted'] for b in blocks if 'strict_accepted' in b),
            full_depth_blocks=sum(b['tree_budget']['max_depth']==b['proposed'] for b in tree),
            node_cap_blocks=sum(b['tree_nodes']==32 for b in tree),
            mean_depth=statistics.mean(b['tree_budget']['max_depth'] for b in tree) if tree else None,
            mean_spine=statistics.mean(b['tree_budget'].get('spine_reserved',0) for b in tree) if tree else None))
    bins=[]
    for (benchmark,stage,state),blocks in sorted(budget_bins.items()):
        bins.append(dict(benchmark=benchmark,stage=stage,state=state,blocks=len(blocks),
            mean_requested=statistics.mean(b['tree_budget']['spine_requested'] for b in blocks),
            mean_reserved=statistics.mean(b['tree_budget']['spine_reserved'] for b in blocks),
            mean_nodes=statistics.mean(b['tree_nodes'] for b in blocks),
            mean_depth=statistics.mean(b['tree_budget']['max_depth'] for b in blocks),
            mean_accepted=statistics.mean(b['accepted'] for b in blocks),
            relaxed_blocks=sum(b['relaxed_tokens']>0 for b in blocks),regret=sum(b['regret'] for b in blocks)))
    cells={(r['request_id'],r['stage'],r['arm']):r for r in rows}
    samples=[]
    for (identity,stage,arm),row in sorted(cells.items()):
        if arm!='JointTree-ASD':continue
        full=cells[identity,stage,'SpineTree-ASD']
        dynamic=any(b['tree_budget']['spine_requested']<b['proposed'] for b in row['rounds'])
        if not dynamic:
            normalized=deepcopy(full['rounds'])
            for b in normalized:b['tree_budget']['spine_mode']='remaining'
            assert row['rounds']==normalized and row['token_ids']==full['token_ids']
        first_change=next((i for i,(a,b) in enumerate(zip(row['token_ids'],full['token_ids'])) if a!=b),None)
        samples.append(dict(request_id=identity,benchmark=row['benchmark'],stage=stage,
            dynamic_used=dynamic,joint_tokens=len(row['token_ids']),spine_tokens=len(full['token_ids']),
            joint_calls=len(row['rounds']),spine_calls=len(full['rounds']),first_token_difference=first_change,
            exact_same_output=row['token_ids']==full['token_ids'],
            joint_correct=quality_ok(row) if stage=='natural' else None,
            spine_correct=quality_ok(full) if stage=='natural' else None))
    result=dict(note='Repeat0 mechanisms only. Budget strata are observational, not causal controls.',
                summaries=summaries,joint_budget_bins=bins,paired_spine_samples=samples)
    (OUT/'mechanism.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(dict(method_cells=len(summaries),budget_bins=len(bins),paired_samples=len(samples),
        dynamic_natural_requests=sum(s['dynamic_used'] for s in samples if s['stage']=='natural'),
        same_natural_outputs=sum(s['exact_same_output'] for s in samples if s['stage']=='natural'))))


if __name__=='__main__':main()
