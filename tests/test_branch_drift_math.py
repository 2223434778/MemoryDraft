import math

import pytest

from scripts.diagnose_branch_drift_math import exceptions,factors,periodic_tail


def test_factorization_separates_length_from_amortized_round_cost():
    baseline=dict(seconds=1.,tokens=100,rounds=10)
    assert factors(dict(seconds=3.,tokens=300,rounds=30),baseline)==dict(
        latency_ratio=3.,length_ratio=3.,rounds_per_token_ratio=1.,amortized_wall_per_round_ratio=1.,product=3.)
    v=factors(dict(seconds=1.2,tokens=100,rounds=10),baseline)
    assert v['length_ratio']==v['rounds_per_token_ratio']==1
    assert v['amortized_wall_per_round_ratio']==pytest.approx(1.2)


def test_periodic_suffix_includes_partial_final_cycle_and_excludes_unrelated_prefix():
    assert periodic_tail([5,9]+[1,2,3]*4+[1,2])==dict(
        period=3,tokens=14,start_1based=3,complete_cycles=4,trailing_partial=2)
    assert periodic_tail(list(range(40))) is None
    assert periodic_tail([]) is None


def test_exception_positions_are_bounds_and_multi_exception_cost_is_not_a_single_probability():
    row=dict(rounds=[dict(generated_before=10,accepted=3,relaxed_tokens=1,regret=2.),
                     dict(generated_before=14,accepted=4,relaxed_tokens=2,regret=3.)])
    events=exceptions(row)
    assert events[0]['possible_position_1based']==[11,13]
    assert events[0]['candidate_to_argmax_ratio']==pytest.approx(math.exp(-2))
    assert events[1]['candidate_to_argmax_ratio'] is None


def test_invalid_counts_fail_instead_of_producing_a_decomposition():
    with pytest.raises(ValueError):
        factors(dict(seconds=1.,tokens=10,rounds=0),dict(seconds=1.,tokens=10,rounds=1))
