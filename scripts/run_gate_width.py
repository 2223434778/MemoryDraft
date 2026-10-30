"""Fixed-width and observed-coverage feedback under unchanged fidelity gates."""
import math
import time
from unittest.mock import patch

from memorydraft.backends.branch_drift import BranchDriftVerifier, DriftConfig, FeedbackTreeConfig, GateTreeConfig
from memorydraft.backends.gate_width import FeedbackWidthVerifier
from memorydraft.evaluation.asd_paper import request_for
from memorydraft.evaluation.branch_drift import summarize
from scripts import run_asd_tree as common
from scripts.run_branch_drift import main
from scripts.run_branch_drift_optimization import execute as previous, select_records

ARMS = ('ASD', 'BudgetTree', 'GateCap-ASD', 'WidthHalf-ASD', 'WidthZero-ASD', 'WidthFeedback-ASD')
SETTINGS = dict(fixed_penalties=[1., .5, 0.], feedback_window=4,
    feedback_rule='zero/one/two-or-more recoverable misses select 1/.5/0',
    max_js=.1, max_token_regret=math.log(4), budget=8., gate=.25, cap=2, max_nodes=32)


def execute(engine, record, arm, limit, tree_config):
    if arm in ('ASD', 'BudgetTree', 'GateCap-ASD'):
        return previous(engine, record, arm, limit, tree_config)
    if arm not in ARMS:
        raise ValueError('Unknown width arm')
    feedback = arm == 'WidthFeedback-ASD'
    penalty = .5 if arm == 'WidthHalf-ASD' else 0. if arm == 'WidthZero-ASD' else 1.
    cls = FeedbackTreeConfig if feedback else GateTreeConfig
    tree = cls(tree_config.costs, tree_config.min_nodes, tree_config.max_nodes,
               tree_config.top_k, tree_config.spine_mode, penalty)
    verifier_cls = FeedbackWidthVerifier if feedback else BranchDriftVerifier
    factory = lambda config: verifier_cls(config,
        drift=DriftConfig(evaluation='lazy', max_token_regret=math.log(4)))
    engine.synchronize()
    started = time.perf_counter()
    with patch.object(common, 'TreeASDVerifier', factory):
        row = common.execute(engine, record, 'BudgetTree-ASD', limit, tree, request_factory=request_for)
    row.update(arm=arm, method=arm, latency_seconds=time.perf_counter()-started)
    return row


def summarize_results(rows, refs):
    for row in rows:
        if row['arm'] not in ARMS[2:]:
            continue
        config = row['asd']['branch_drift']
        if (config['max_js'] != .1 or config['max_token_regret'] != math.log(4)
                or config['protect_stop'] is not True or config['evaluation'] != 'lazy'):
            raise ValueError('Changed fidelity conditions')
        history = []
        for block in row['rounds']:
            meta = block['tree_budget']
            if block['tree_nodes'] > 32:
                raise ValueError('Node cap exceeded')
            if row['arm'] == 'WidthFeedback-ASD':
                misses = sum(history[-4:])
                expected = 1. if not misses else .5 if misses == 1 else 0.
                if meta['gate_penalty'] != expected:
                    raise ValueError('Feedback used future or incorrect history')
                history.append(meta['width_recoverable_miss'])
    return summarize(rows, refs, arms=ARMS, prefix_arms=ARMS)


if __name__ == '__main__':
    main(study='gate-width', arms=ARMS, prefix_arms=ARMS, execution=execute,
         summarize_results=summarize_results, select_records=select_records, settings=SETTINGS)
