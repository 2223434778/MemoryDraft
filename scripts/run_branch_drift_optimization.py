"""Fixed seven-task comparison of lazy JS, allocation and raw token regret."""
import math
import time
from unittest.mock import patch

from memorydraft.backends.branch_drift import BranchDriftVerifier, DriftConfig, GateTreeConfig, PairedTreeConfig
from memorydraft.evaluation.branch_drift import records, summarize
from memorydraft.evaluation.asd_paper import request_for
from scripts import run_asd_tree as common
from scripts.run_branch_drift import execute as baseline, main

ARMS = ('ASD', 'BudgetTree', 'DriftTree-ASD', 'LazyDrift-ASD',
        'GateTree-ASD', 'TokenCap-ASD', 'GateCap-ASD')
SETTINGS = dict(max_js=.1, token_probability_ratio_min=.25, max_token_regret=math.log(4),
                gate_penalty=1., budget=8., gate=.25, cap=2,
                allocation_proxy='Draft q(rank)/q(top1), not calibrated Target safety')


def select_records(root):
    # Fixed low-cost development subset; retain both known failure diagnostics.
    pool = records(root)
    counts = dict(gsm8k=8, math500=12, humaneval=8, mbpp=8, mmlu=14, mtbench=4, alpaca=8)
    selected = []
    for benchmark, count in counts.items():
        candidates = [r for r in pool if r['benchmark'] == benchmark]
        if benchmark == 'humaneval':
            candidates.sort(key=lambda r: r['id'] != 'humaneval:76')
        selected.extend(candidates[:count])
    return selected


def execute(engine, record, arm, limit, tree_config):
    if arm in ('ASD', 'BudgetTree', 'DriftTree-ASD'):
        return baseline(engine, record, arm, limit, tree_config)
    if arm not in ARMS:
        raise ValueError('Unknown optimization arm')
    allocation = arm in ('GateTree-ASD', 'GateCap-ASD')
    cap = math.log(4) if arm in ('TokenCap-ASD', 'GateCap-ASD') else None
    cls = GateTreeConfig if allocation else PairedTreeConfig
    tree = cls(tree_config.costs, tree_config.min_nodes, tree_config.max_nodes,
               tree_config.top_k, tree_config.spine_mode)
    factory = lambda config: BranchDriftVerifier(config,
        drift=DriftConfig(evaluation='lazy', max_token_regret=cap))
    engine.synchronize()
    started = time.perf_counter()
    with patch.object(common, 'TreeASDVerifier', factory):
        row = common.execute(engine, record, 'BudgetTree-ASD', limit, tree, request_factory=request_for)
    row.update(arm=arm, method=arm, latency_seconds=time.perf_counter()-started)
    return row


def summarize_results(rows, refs):
    indexed = {(r['request_id'], r['stage'], r['arm']): r for r in rows}
    for row in rows:
        if row['arm'] != 'LazyDrift-ASD':
            continue
        eager = indexed[row['request_id'], row['stage'], 'DriftTree-ASD']
        fields = ('tree_path', 'relaxed_tokens', 'regret', 'accepted', 'advance')
        if (row['token_ids'] != eager['token_ids'] or row['finish_reason'] != eager['finish_reason']
                or row['asd']['spent'] != eager['asd']['spent']
                or [[r[k] for k in fields] for r in row['rounds']]
                != [[r[k] for k in fields] for r in eager['rounds']]):
            raise ValueError('Lazy/eager decision mismatch: '+row['request_id'])
    return summarize(rows, refs, arms=ARMS, prefix_arms=ARMS)


if __name__ == '__main__':
    main(study='branch-drift-optimization', arms=ARMS, prefix_arms=ARMS,
         execution=execute, summarize_results=summarize_results,
         select_records=select_records, settings=SETTINGS)
