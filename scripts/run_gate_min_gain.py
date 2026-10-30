"""Natural-answer comparison of a minimum verified gain before approximation."""
import math
import time
from unittest.mock import patch

from memorydraft.backends.branch_drift import DriftConfig, FeedbackTreeConfig
from memorydraft.backends.gate_width import MinGainFeedbackVerifier
from memorydraft.evaluation.asd_paper import request_for
from memorydraft.evaluation.branch_drift import summarize
from scripts import run_asd_tree as common
from scripts.run_branch_drift import main
from scripts.run_branch_drift_optimization import select_records
from scripts.run_gate_width import execute as previous

ARMS = ('ASD', 'WidthFeedback-ASD', 'MinGainFeedback-ASD')
SETTINGS = dict(min_extra_tokens=2, max_js=.1, max_token_regret=math.log(4),
                budget=8., gate=.25, cap=2, max_nodes=32, feedback_window=4)


def execute(engine, record, arm, limit, tree_config):
    if arm in ARMS[:2]:
        return previous(engine, record, arm, limit, tree_config)
    if arm != ARMS[2]:
        raise ValueError('Unknown minimum gain arm')
    tree = FeedbackTreeConfig(tree_config.costs, tree_config.min_nodes, tree_config.max_nodes,
                              tree_config.top_k, tree_config.spine_mode)
    factory = lambda config: MinGainFeedbackVerifier(config,
        drift=DriftConfig(evaluation='lazy', max_token_regret=math.log(4)))
    engine.synchronize()
    started = time.perf_counter()
    with patch.object(common, 'TreeASDVerifier', factory):
        row = common.execute(engine, record, 'BudgetTree-ASD', limit, tree, request_factory=request_for)
    row.update(arm=arm, method=arm, latency_seconds=time.perf_counter()-started)
    return row


def summarize_results(rows, refs):
    for row in rows:
        if row['arm'] == ARMS[2] and any(b['relaxed_tokens']
                and b['accepted']-b['strict_accepted'] < 2 for b in row['rounds']):
            raise ValueError('Committed approximation below minimum gain')
    return summarize(rows, refs, arms=ARMS, prefix_arms=())


if __name__ == '__main__':
    main(study='gate-min-gain', arms=ARMS, prefix_arms=(), execution=execute,
         summarize_results=summarize_results, select_records=select_records, settings=SETTINGS)
