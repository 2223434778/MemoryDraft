import math

import pytest
import torch

from memorydraft.backends.asd import ASDConfig
from memorydraft.backends.branch_drift import DriftConfig, FeedbackTreeConfig
from memorydraft.backends.buffered_cache import BufferedDynamicCache
from memorydraft.backends.dflash_reuse import generate_reuse
from memorydraft.backends.gate_width import MinGainFeedbackVerifier
from tests.test_budget_tree import costs
from tests.test_dflash_reuse import CheckedDraft, tiny
from tests.test_gate_min_gain import fixture


@pytest.mark.parametrize('gain', [1, 2])
def test_reused_scores_preserve_paths_regret_and_feedback(gain):
    outputs, verifiers = [], []
    for reuse in (False, True):
        tree, logits = fixture(gain=gain)
        verifier = MinGainFeedbackVerifier(record=True, reuse_scores=reuse)
        outputs.append(verifier.decide_tree(tree, logits, horizon=3))
        verifiers.append(verifier)
    assert outputs[0] == outputs[1]
    assert verifiers[0].events == verifiers[1].events
    assert verifiers[0].width_history == verifiers[1].width_history


@pytest.mark.parametrize('budget', [0, 8])
@pytest.mark.parametrize('stop', [False, True])
def test_real_generation_tokens_stream_tree_and_kv_match(tiny, budget, stop):
    results, streams, events = [], [], []
    tree = FeedbackTreeConfig(costs(), 2, 12, 4)
    for optimized in (False, True):
        stream, seen = [], []
        verifier = MinGainFeedbackVerifier(ASDConfig(budget=budget), reuse_scores=optimized,
            drift=DriftConfig(evaluation='lazy', max_token_regret=math.log(4)))
        result = generate_reuse(tiny.target, CheckedDraft(tiny.features), tiny.ids, max_new_tokens=25,
            tree_config=tree, verifier=verifier, incremental_stop=True, on_tokens=stream.extend,
            on_event=seen.append, cache_factory=BufferedDynamicCache if optimized else None,
            stop_token_ids=[int(tiny.reference[0, 8])] if stop else ())
        results.append(result); streams.append(stream); events.append(seen)
    assert results[0].sequences.tolist() == results[1].sequences.tolist()
    assert streams[0] == streams[1] and events[0] == events[1]
    for a, b in zip(results[0].rounds, results[1].rounds):
        assert {k:v for k,v in a.items() if k != 'tree_budget'} == {k:v for k,v in b.items() if k != 'tree_budget'}
    assert results[1].cache_statistics['target']['allocations'] > 0


def test_custom_storage_rejects_incompatible_prefix_prefill(tiny):
    with pytest.raises(ValueError, match='prefill'):
        generate_reuse(tiny.target, CheckedDraft(tiny.features), tiny.ids, max_new_tokens=2,
                       cache_factory=BufferedDynamicCache, prefill=lambda: None)
