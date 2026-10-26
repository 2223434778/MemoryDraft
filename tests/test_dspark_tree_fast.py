from unittest.mock import patch

import pytest
import torch

from memorydraft.backends.asd import ASDConfig
from memorydraft.backends.budget_tree import BudgetTreeConfig, TreeCosts, TreeASDVerifier
from memorydraft.backends.dspark_reuse import generate_dspark
from memorydraft.backends.dspark_tree import build_dspark_tree
from memorydraft.backends.dspark_tree_fast import MarkovRowCache, build_dspark_tree_fast, future_can_improve
from tests.test_dspark import small_draft, tiny_dspark
from tests.test_dflash_reuse import tiny


def semantic(tree):
    meta=tree.budget_metadata
    return (tree.tokens.tolist(),tree.parents,tree.depths,
            {k:meta[k] for k in ('nodes','max_depth','expected_advance','estimated_cycle_seconds','horizon','top_k','proposal')})


@pytest.mark.parametrize('cache_enabled',[False,True])
@pytest.mark.parametrize('early_stop',[False,True])
def test_fast_tree_matches_reference_across_probabilities_costs_and_eos(cache_enabled,early_stop):
    torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(428)
    draft=small_draft();cache=MarkovRowCache(draft,max_entries=3) if cache_enabled else None
    for i in range(80):
        horizon=i%4+1
        base=torch.randn(1,horizon,4,generator=generator)*(20 if i%3==0 else 1)
        values=torch.rand(2,2,generator=generator).add(.01).tolist()
        costs=TreeCosts((1,100),(2,32),tuple(tuple(x) for x in values),.02)
        config=BudgetTreeConfig(costs,min_nodes=2+i%7,max_nodes=12+i%21,top_k=1+i%4)
        anchor=torch.tensor([i%4]);stops=(1,) if i%5==0 else ()
        reference=build_dspark_tree(anchor,base,draft,config,10+i,stop_token_ids=stops)
        actual=build_dspark_tree_fast(anchor,base,draft,config,10+i,stop_token_ids=stops,
                                     cache=cache,early_stop=early_stop)
        assert semantic(actual)==semantic(reference)


def test_bound_stops_early_without_changing_selected_tree():
    draft=small_draft();base=torch.full((1,4,4),-40.);base[:,:,0]=40.
    costs=TreeCosts((1,100),(2,32),((.01,.5),(.01,.5)),.02)
    config=BudgetTreeConfig(costs,8,32,4)
    reference=build_dspark_tree(torch.tensor([0]),base,draft,config,10)
    actual=build_dspark_tree_fast(torch.tensor([0]),base,draft,config,10)
    assert semantic(actual)==semantic(reference)
    assert actual.budget_metadata['search_bound_stopped']
    assert actual.budget_metadata['built_nodes']<reference.budget_metadata['built_nodes']


def test_bound_keeps_search_when_later_cost_valley_can_win():
    draft=small_draft();base=torch.zeros(1,4,4)
    costs=TreeCosts((1,100),(2,8,16,32),((.01,.08,.06,.001),(.01,.08,.06,.001)),.001)
    config=BudgetTreeConfig(costs,8,32,4)
    reference=build_dspark_tree(torch.tensor([0]),base,draft,config,10)
    actual=build_dspark_tree_fast(torch.tensor([0]),base,draft,config,10)
    assert semantic(actual)==semantic(reference)
    assert actual.budget_metadata['nodes']==32
    assert future_can_improve(1.,1,1.,2**-54,{2:1.},2)


def test_markov_rows_reuse_identical_projection_and_bound_memory():
    draft=small_draft();cache=MarkovRowCache(draft,max_entries=2);anchor=torch.tensor([0])
    with patch.object(draft.markov_w2,'forward',wraps=draft.markov_w2.forward) as projected:
        a=cache.bias(0,anchor);cache.bias(1,anchor);b=cache.bias(0,anchor)
        assert torch.equal(a,b) and projected.call_count==2
        cache.bias(2,anchor);cache.bias(1,anchor)
        assert projected.call_count==4
    assert len(cache.rows)==2 and cache.summary()['peak_bytes']==2*4*4
    with torch.no_grad():draft.markov_w2.weight.add_(.1)
    with pytest.raises(ValueError,match='weights changed'):cache.bias(0,anchor)
    with pytest.raises(ValueError):MarkovRowCache(draft,max_entries=0)


@pytest.mark.parametrize('budget',[0,8])
def test_fast_real_generation_and_budget_match_reference(tiny,budget):
    from memorydraft.backends import dspark_reuse as module
    config=BudgetTreeConfig(TreeCosts((1,100),(2,32),((.01,.03),(.01,.03)),.02),2,12,4)
    draft=tiny_dspark(tiny)
    # Disable the fixture's fixed-history assertion for approximate paths;
    # the existing real Target/cache tests separately check cache correctness.
    if budget:draft.backbone.expected=None
    original_forward=draft.backbone.forward
    if budget:
        def forward(**kwargs):
            noise=kwargs['noise_embedding'];cache=kwargs['past_key_values']
            length=noise.shape[1]+kwargs['target_hidden'].shape[1]
            values=noise.new_zeros((1,1,length,8));cache.update(values,values,0)
            return torch.zeros_like(noise)
        draft.backbone.forward=forward
    reference_verifier=TreeASDVerifier(ASDConfig(budget=budget,gate=100,cap=3))
    reference=generate_dspark(tiny.target,draft,tiny.ids,max_new_tokens=29,tree_config=config,verifier=reference_verifier)
    cache=MarkovRowCache(draft)
    def builder(*args,**kwargs):return build_dspark_tree_fast(*args,cache=cache,**kwargs)
    actual_verifier=TreeASDVerifier(ASDConfig(budget=budget,gate=100,cap=3))
    with patch.object(module,'build_dspark_tree',builder):
        actual=generate_dspark(tiny.target,draft,tiny.ids,max_new_tokens=29,tree_config=config,verifier=actual_verifier)
    assert actual.sequences.tolist()==reference.sequences.tolist()
    assert actual_verifier.summary()==reference_verifier.summary()
    for x,y in zip(actual.rounds,reference.rounds):
        for name in ('accepted','advance','regret','tree_path'):assert x[name]==y[name]
    draft.backbone.forward=original_forward
