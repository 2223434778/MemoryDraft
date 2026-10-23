"""Descriptive four-arm interaction and per-request attribution; no selection."""
from pathlib import Path
import json,statistics
from collections import Counter
from memorydraft.evaluation.asd_comparison import bootstrap
from scripts.write_asd_tree_report import read_rows

root=Path(__file__).resolve().parents[2]
out=root/'results/asd-budget-tree'
samples=json.loads((out/'analysis.json').read_text())['samples']
result={}
raw=read_rows(out,'rows')
for benchmark in sorted({s['benchmark'] for s in samples}):
    chosen=[s for s in samples if s['benchmark']==benchmark]
    stages={}
    for stage in ('natural','prefix'):
        common=[s for s in chosen if stage=='natural' or all(s['arms'][f'{stage}/{a}']['tokens']==128 for a in ('DFlash','ASD','BudgetTree','BudgetTree-ASD'))]
        values=[]
        for s in common:
            v=lambda a:s['arms'][f'{stage}/{a}']['seconds']
            values.append((v('BudgetTree-ASD')/v('BudgetTree'))/(v('ASD')/v('DFlash')))
        stages[stage]=dict(n=len(common),mean_interaction_ratio=statistics.mean(values) if values else None,ci95=bootstrap(values) if values else None)
    result[benchmark]=stages
    mechanisms={}
    for arm in ('BudgetTree','BudgetTree-ASD'):
        entries=[r for row in raw if row['benchmark']==benchmark and row['arm']==arm and row['stage']=='natural'
                 and row['repeat']==0 for r in row['rounds']]
        full=[r for r in entries if r['tree_budget']['horizon']==15]
        mechanisms[arm]=dict(blocks=len(entries),node_histogram=dict(Counter(r['tree_nodes'] for r in entries)),
            full_horizon_blocks=len(full),full_horizon_mean_max_depth=statistics.mean(r['tree_budget']['max_depth'] for r in full) if full else None,
            full_horizon_shallower_than_native=sum(r['tree_budget']['max_depth']<15 for r in full),
            mean_advance=statistics.mean(r['advance'] for r in entries) if entries else None,
            sum_predicted_advance=sum(r['tree_budget']['expected_advance'] for r in entries),
            sum_recorded_advance=sum(r['advance'] for r in entries),
            relaxed_blocks=sum(r.get('relaxed_tokens',0)>0 for r in entries),
            extra_vs_same_tree_strict=sum(r['accepted']-r['strict_accepted'] for r in entries) if arm=='BudgetTree-ASD' else None)
    result[benchmark]['mechanisms']=mechanisms
(out/'interaction.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result))
