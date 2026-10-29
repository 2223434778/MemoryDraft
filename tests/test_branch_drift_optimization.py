from dataclasses import replace
import math

import pytest
import torch

from memorydraft.backends.asd import ASDConfig
from memorydraft.backends.branch_drift import (BranchDriftVerifier, DriftConfig,
    GateTreeConfig, PairedTreeConfig, build_paired_tree)
from tests.test_branch_drift import example
from tests.test_budget_tree import costs
from tests.test_runtime import engine


def test_lazy_matches_eager_across_trees_stops_budgets_and_ties():
    generator = torch.Generator().manual_seed(20261029)
    for sample in range(120):
        scores = torch.randn(1, 15, 8, generator=generator)
        count = 8+sample % 25
        tree = build_paired_tree(torch.zeros(1, 16, dtype=torch.long), scores,
                                PairedTreeConfig(costs(), count, count, 4), 128)
        logits = torch.randn(len(tree.parents), 8, generator=generator)*(.1 if sample % 2 else 2.)
        if sample % 7 == 0:
            logits.zero_()
        config = ASDConfig(budget=(0., .5, 8.)[sample % 3], cap=sample % 3,
                           gate=(0., .25, 2.)[sample % 3])
        drift = DriftConfig(max_js=(0., .1, .5, 1.)[sample % 4],
                            max_token_regret=None if sample % 2 else math.log(4))
        eager = BranchDriftVerifier(config, drift=drift)
        lazy = BranchDriftVerifier(config, drift=replace(drift, evaluation='lazy'))
        eager.spent = lazy.spent = config.budget*(sample % 4)/4
        for _ in range(2):
            kwargs = dict(horizon=15, stop_token_ids=(0,) if sample % 5 == 0 else ())
            assert eager.decide_tree(tree, logits, **kwargs) == lazy.decide_tree(tree, logits, **kwargs)
            assert eager.spent == lazy.spent and eager.relaxed == lazy.relaxed
        assert lazy.counters['compared_edges'] <= eager.counters['compared_edges']


def test_lazy_skips_irrelevant_edges_and_checks_rejected_winner():
    tree, logits = example(far=True)
    eager = BranchDriftVerifier()
    lazy = BranchDriftVerifier(drift=DriftConfig(evaluation='lazy'))
    assert eager.decide_tree(tree, logits, horizon=15) == lazy.decide_tree(tree, logits, horizon=15)
    assert lazy.counters['drift_blocked'] == 1
    # Stop protection eliminates the approximate path before expensive JS.
    lazy = BranchDriftVerifier(drift=DriftConfig(evaluation='lazy'))
    lazy.decide_tree(tree, logits, horizon=15, stop_token_ids=(2,))
    assert lazy.counters['compared_edges'] == 0


def test_token_regret_cap_is_independent_of_remaining_horizon():
    tree, logits = example()
    logits[0, 2] = 0.
    original = BranchDriftVerifier(drift=DriftConfig(evaluation='lazy'))
    capped = BranchDriftVerifier(drift=DriftConfig(evaluation='lazy', max_token_regret=math.log(4)))
    assert original.decide_tree(tree, logits, horizon=15)[0] == [0, 2, 3]
    assert capped.decide_tree(tree, logits, horizon=15)[0] == [0, 1]
    assert capped.spent == 0 and capped.counters['token_regret_blocked'] == 1


def test_gate_allocation_retains_depth_and_reference_before_alternatives():
    scores = torch.tensor([[[6., 0., -1., -2.]]*15])
    block = torch.zeros(1, 16, dtype=torch.long)
    paired = build_paired_tree(block, scores, PairedTreeConfig(costs(), 16, 16, 4), 128)
    guarded = build_paired_tree(block, scores, GateTreeConfig(costs(), 16, 16, 4), 128)
    assert max(guarded.depths) == 15 > max(paired.depths)
    assert len(guarded.parents) <= 16
    scores = torch.zeros(1, 15, 4)
    tree = build_paired_tree(block, scores, GateTreeConfig(costs(), 8, 32, 4), 128)
    first = int(scores[0, 0].topk(4).indices[0])
    children = {}
    for i, parent in enumerate(tree.parents[1:], 1):
        assert parent < i
        children.setdefault(parent, set()).add(int(tree.tokens[0, i]))
    assert all(first in group for group in children.values())


@pytest.mark.parametrize('kwargs', [dict(evaluation='other'), dict(max_token_regret=-1),
    dict(max_token_regret=float('inf')), dict(max_token_regret=True)])
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        DriftConfig(**kwargs)


@pytest.mark.parametrize('arm', ['LazyDrift-ASD', 'GateTree-ASD', 'TokenCap-ASD', 'GateCap-ASD'])
def test_runtime_dispatch_keeps_typed_costs_and_explicit_options(engine, arm):
    from memorydraft.backends.budget_tree import BudgetTreeConfig
    from scripts.run_branch_drift_optimization import execute
    row = execute(engine, dict(id='opt-fixture', benchmark='gsm8k', question='w1 w2', answer='#### 2'),
                  arm, 19, BudgetTreeConfig(costs(), 2, 12, 4))
    assert row['arm'] == arm and len(row['token_ids']) == 19
    config = row['asd']['branch_drift']
    assert config['evaluation'] == 'lazy'
    assert (config['max_token_regret'] is not None) == (arm in ('TokenCap-ASD', 'GateCap-ASD'))
    assert all(r['tree_budget']['sibling_pairs'] == (arm not in ('GateTree-ASD', 'GateCap-ASD'))
               for r in row['rounds'])


def study_matrix():
    from memorydraft.evaluation.asd_paper import BENCHMARKS
    from scripts.run_branch_drift_optimization import ARMS
    refs, rows = [], []
    for benchmark in BENCHMARKS:
        ref = dict(request_id=benchmark+':audit', benchmark=benchmark, token_ids=[1],
                   prompt_sha256='same', snapshot_id='same')
        refs.append(ref)
        for stage in ('natural', 'prefix'):
            for arm in ARMS:
                rows.append(dict(ref, stage=stage, arm=arm, latency_seconds=1., max_new_tokens=4096,
                    finish_reason='eos', rounds=[], asd=dict(spent=0., relaxed_tokens=0,
                        branch_drift=dict(max_token_regret=math.log(4))),
                    quality=dict(correct=None) if stage == 'natural' else None))
    return rows, refs


def test_study_audit_rejects_lazy_output_or_budget_changes():
    from copy import deepcopy
    from scripts.run_branch_drift_optimization import summarize_results
    rows, refs = study_matrix()
    assert summarize_results(rows, refs)['timed_generations'] == len(refs)*14
    bad = deepcopy(rows)
    next(r for r in bad if r['arm'] == 'LazyDrift-ASD')['token_ids'] = [2]
    with pytest.raises(ValueError, match='decision mismatch'):
        summarize_results(bad, refs)
    bad = deepcopy(rows)
    next(r for r in bad if r['arm'] == 'LazyDrift-ASD')['asd']['spent'] = .1
    with pytest.raises(ValueError, match='decision mismatch'):
        summarize_results(bad, refs)


def test_report_rebuilds_compressed_publication_without_changing_payload(tmp_path, monkeypatch):
    import json
    from scripts import report_branch_drift_optimization as report
    from scripts.run_branch_drift_optimization import summarize_results
    rows, refs = study_matrix()
    source = tmp_path/'source'
    source.mkdir()
    (source/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    for name, value in dict(references=refs, analysis=summarize_results(rows, refs),
                            provenance=dict(status='completed'), tasks=[], **{'cost-table': {}}).items():
        (source/(name+'.json')).write_text(json.dumps(value), encoding='utf-8')
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    args = ['report', '--input', 'source', '--output', 'public', '--report', 'report.md']
    monkeypatch.setattr('sys.argv', args)
    report.main()
    before = {p.name: p.read_bytes() for p in (tmp_path/'public').iterdir()}
    monkeypatch.setattr('sys.argv', ['report', '--input', 'public', '--output', 'public', '--report', 'report.md'])
    report.main()
    assert before == {p.name: p.read_bytes() for p in (tmp_path/'public').iterdir()}
