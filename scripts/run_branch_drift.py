"""Direct seven-task evaluation of Target sibling distribution gates."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random
import time
from unittest.mock import patch

from memorydraft.backends.branch_drift import BranchDriftVerifier, DriftConfig, PairedTreeConfig
from memorydraft.evaluation.asd_joint_tree import frozen_config
from memorydraft.evaluation.asd_paper import conversation_record, request_for, score
from memorydraft.evaluation.branch_drift import ARMS, PREFIX_ARMS, records, references, summarize
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts import run_asd_tree as common
from scripts.diagnose_transport_online import write
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]


def execute(engine, record, arm, limit, tree_config):
    if arm not in ('CoverageTree-ASD', 'DriftTree-ASD'):
        return common.execute(engine, record, arm, limit, tree_config, request_factory=request_for)
    paired = PairedTreeConfig(tree_config.costs,tree_config.min_nodes,tree_config.max_nodes,
                              tree_config.top_k,tree_config.spine_mode)
    threshold = 1. if arm == 'CoverageTree-ASD' else .10
    factory = lambda config: BranchDriftVerifier(config, drift=DriftConfig(max_js=threshold))
    engine.synchronize()
    started = time.perf_counter()
    with patch.object(common, 'TreeASDVerifier', factory):
        row = common.execute(engine, record, 'BudgetTree-ASD', limit, paired, request_factory=request_for)
    row.update(arm=arm, method=arm, latency_seconds=time.perf_counter()-started)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/raw/branch-drift')
    args = parser.parse_args()
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=False)
    pool = records(ROOT)
    refs = references(ROOT, pool)
    indexed = {r['request_id']: r for r in refs}
    write(out/'references.json', refs)
    write(out/'tasks.json', pool)
    info = dict(status='started', arms=ARMS, prefix_arms=PREFIX_ARMS,
        selected_tasks=len(pool), matched_inputs=len(refs), max_js=.10,
        selection='Deterministic subset of previously evaluated extension tasks; no outcome selection',
        reference_policy='Reuse prior strict AR tokens, never reuse latency as same-session timing',
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), max_new_tokens=4096)
    write(out/'provenance.json', info)
    engine = None
    rows = []
    try:
        started = time.perf_counter()
        config = replace(RuntimeConfig.load(ROOT/'configs/runtime-dflash-rmsnorm.json'), db=None, max_new_tokens=4096)
        engine = load_engine(config)
        info['load_seconds'] = time.perf_counter()-started
        tree_config, cost_source = frozen_config(ROOT, int(os.environ['CUDA_VISIBLE_DEVICES']))
        write(out/'cost-table.json', cost_source)
        warm = dict(id='warmup', benchmark='gsm8k', question='What is 3 plus 4?', answer='#### 7')
        for arm in ARMS:
            append(out/'warmup.jsonl', execute(engine, warm, arm,64,tree_config))
        info['runtime'] = provenance(engine)
        info['started_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write(out/'provenance.json', info)
        for task_index, task in enumerate(pool):
            first = indexed.get(task['id']+':turn1')
            for turn in ((1, 2) if task['benchmark'] == 'mtbench' else (1,)):
                record = (conversation_record(task, turn, first['text'] if turn == 2 else None)
                          if task['benchmark'] == 'mtbench' else task)
                ref = indexed[record['id']]
                cache = {}
                for stage, limit, arms in (('natural',4096,ARMS), ('prefix',128,PREFIX_ARMS)):
                    order = list(arms)
                    random.Random('branch-drift:'+record['id']+':'+stage).shuffle(order)
                    for arm in order:
                        row = execute(engine,record,arm,limit,tree_config)
                        row.update(stage=stage,repeat=0,stop_policy='natural-eos',
                            context_policy='shared frozen strict-AR first-turn history' if turn == 2 else 'original request')
                        if (row['prompt_sha256'],row['snapshot_id']) != (ref['prompt_sha256'],ref['snapshot_id']):
                            raise ValueError('Changed Target prompt or snapshot: '+record['id'])
                        if stage == 'natural':
                            if row['text'] not in cache:
                                cache[row['text']] = score(record,row['text'])
                            row['quality'] = cache[row['text']]
                        else:
                            row['quality'] = None
                        rows.append(row)
                        append(out/'rows.jsonl',row)
                        info.update(completed_tasks=task_index, timed_generations=len(rows),
                                    last_task=record['id'], last_arm=arm, last_stage=stage)
                        write(out/'provenance.json',info)
            print(json.dumps(dict(completed=task_index+1,total=len(pool),benchmark=task['benchmark'],rows=len(rows))),flush=True)
        write(out/'analysis.json',summarize(rows,refs))
        info.update(status='completed',completed_tasks=len(pool),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    except BaseException as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error))
        raise
    finally:
        if engine is not None:
            info['compiler'] = engine._regional.summary()
            info['peak_gpu_bytes'] = max((r['peak_gpu_bytes'] for r in rows),default=0)
            engine.close()
        write(out/'provenance.json',info)


if __name__ == '__main__':
    main()
