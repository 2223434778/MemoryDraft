"""Direct natural-answer comparison of score reuse and buffered KV storage."""
import math
import time
from unittest.mock import patch

from memorydraft.backends.branch_drift import DriftConfig, FeedbackTreeConfig
from memorydraft.backends.buffered_cache import BufferedDynamicCache
from memorydraft.backends.gate_width import MinGainFeedbackVerifier
from memorydraft.evaluation.asd_paper import request_for
from memorydraft.evaluation.branch_drift import summarize
from scripts import run_asd_tree as common
from scripts.run_branch_drift import main
from scripts.run_branch_drift_optimization import select_records
from scripts.run_gate_min_gain import execute as previous

ARMS = ('MinGainFeedback-ASD', 'ReuseScores-ASD', 'BufferedKV-ASD')
SETTINGS = dict(min_extra_tokens=2, max_js=.1, max_token_regret=math.log(4),
                budget=8., gate=.25, cap=2, max_nodes=32, feedback_window=4, cache_block_tokens=256)


def execute(engine, record, arm, limit, tree_config):
    if arm == ARMS[0]:
        return previous(engine, record, arm, limit, tree_config)
    if arm not in ARMS:
        raise ValueError('Unknown tree execution arm')
    from memorydraft.runtime import engine as module
    native, detail = module.generate_reuse, None
    def optimized(*args, **kwargs):
        nonlocal detail
        if arm == 'BufferedKV-ASD':
            kwargs['cache_factory'] = BufferedDynamicCache
        detail = native(*args, **kwargs)
        return detail
    tree = FeedbackTreeConfig(tree_config.costs, tree_config.min_nodes, tree_config.max_nodes,
                              tree_config.top_k, tree_config.spine_mode)
    factory = lambda config: MinGainFeedbackVerifier(config, reuse_scores=True,
        drift=DriftConfig(evaluation='lazy', max_token_regret=math.log(4)))
    engine.synchronize()
    started = time.perf_counter()
    with patch.object(common, 'TreeASDVerifier', factory), patch.object(module, 'generate_reuse', optimized):
        row = common.execute(engine, record, 'BudgetTree-ASD', limit, tree, request_factory=request_for)
    row.update(arm=arm, method=arm, latency_seconds=time.perf_counter()-started,
               cache_statistics=detail.cache_statistics)
    return row


def summarize_results(rows, refs):
    analysis = summarize(rows, refs, arms=ARMS, prefix_arms=())
    indexed = {(r['request_id'], r['arm']):r for r in rows}
    differences = []
    for row in rows:
        old = indexed[row['request_id'], ARMS[0]]
        if any(row[k] != old[k] for k in ('token_ids', 'text', 'finish_reason')):
            differences.append(dict(id=row['request_id'], arm=row['arm']))
    analysis['output_differences'] = differences
    return analysis


if __name__ == '__main__':
    main(study='tree-execution', arms=ARMS, prefix_arms=(), execution=execute,
         summarize_results=summarize_results, select_records=select_records, settings=SETTINGS)
