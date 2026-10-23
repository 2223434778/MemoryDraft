from dataclasses import replace

import pytest
import torch

from memorydraft.backends.asd import ASDConfig
from memorydraft.backends.asd_cpu import CPUASDVerifier
from memorydraft.backends.budget_tree import BudgetTreeConfig, RegretBudget, TreeASDVerifier, build_budget_tree
from memorydraft.backends.dflash_reuse import generate_reuse
from tests.test_budget_tree import costs
from tests.test_dflash_reuse import CheckedDraft, tiny


def test_spine_shortens_monotonically_with_actual_remaining_regret():
    states = [RegretBudget(8, left, .25, 2) for left in (8, 7.25, 3.625, .01, 0)]
    assert [s.spine_length(15) for s in states] == [15, 15, 8, 1, 0]
    assert RegretBudget(8, 8, 0, 2).spine_length(15) == 0
    assert RegretBudget(8, 8, .25, 0).spine_length(15) == 0


def test_snapshot_is_read_only_and_construction_never_charges_budget():
    verifier = TreeASDVerifier()
    verifier.spent = 3.
    state = verifier.construction_budget()
    assert state.remaining == 5
    with pytest.raises(AttributeError):state.remaining = 1
    block = torch.zeros(1,16,dtype=torch.long)
    logits = torch.zeros(1,15,8)
    build_budget_tree(block,logits,BudgetTreeConfig(costs(),spine_mode="remaining"),10,regret_budget=state)
    assert verifier.spent == 3


def test_full_spine_survives_topk_ties_and_respects_eos_node_limit_and_parents():
    block = torch.tensor([[7,0,1,2,3]])
    logits = torch.zeros(1,4,16)
    config = BudgetTreeConfig(costs(),2,12,4,"full")
    tree = build_budget_tree(block,logits,config,10,stop_token_ids=(2,))
    assert tree.tokens[0,:4].tolist() == [7,0,1,2]
    assert tree.budget_metadata['spine_reserved'] == 3
    assert len(tree.parents) <= 12
    assert all(p < i and tree.depths[i] == tree.depths[p]+1 for i,p in enumerate(tree.parents) if i)
    assert not any(int(tree.tokens[0,p]) == 2 for p in tree.parents[1:])
    children = [(p,int(tree.tokens[0,i])) for i,p in enumerate(tree.parents) if i]
    assert len(children) == len(set(children))


def test_exhausted_budget_reverts_to_original_best_first_tree():
    torch.manual_seed(19)
    logits = torch.randn(1,15,32)
    block = torch.cat([torch.tensor([[5]]),logits.argmax(-1)],1)
    config = BudgetTreeConfig(costs())
    original = build_budget_tree(block,logits,config,10)
    joint = build_budget_tree(block,logits,replace(config,spine_mode="remaining"),10,
                              regret_budget=RegretBudget(8,0,.25,2))
    assert original.tokens.tolist() == joint.tokens.tolist()
    assert original.parents == joint.parents and original.depths == joint.depths
    assert joint.budget_metadata['spine_reserved'] == 0


@pytest.mark.parametrize('mode',['full','remaining'])
def test_full_budget_retains_linear_asd_option_under_real_target_logits(tiny,mode):
    from transformers import DynamicCache
    block = tiny.reference[:,4:8].clone()
    block[0,1] = (block[0,1]+1)%32
    logits = torch.zeros(1,3,32)
    config = BudgetTreeConfig(costs(),2,12,4,mode)
    policy = ASDConfig(budget=8,gate=.25,cap=2)
    tree = build_budget_tree(block,logits,config,4,regret_budget=RegretBudget(8,8,.25,2))
    with torch.inference_mode():
        cache=DynamicCache()
        tiny.target(tiny.ids,past_key_values=cache,use_cache=True)
        output=tiny.target(tree.tokens,past_key_values=cache,use_cache=True,
            position_ids=torch.tensor([tree.depths])+4,attention_mask=tree.mask(4,torch.float32))
        reference=tiny.target(torch.cat([tiny.ids,block],1)).logits[0,4:-1]
        original=CPUASDVerifier(policy).decide(block[0,1:],reference)
        path,_,decision=TreeASDVerifier(policy).decide_tree(tree,output.logits[0],horizon=3)
        assert decision.accepted >= original.accepted
        assert path[0]==0


@pytest.mark.parametrize('mode',['full','remaining'])
@pytest.mark.parametrize('limit',[0,1,2,9,29])
def test_zero_budget_generation_stream_kv_and_features_are_strict(tiny,mode,limit):
    published=[]
    verifier=TreeASDVerifier(ASDConfig(budget=0))
    result=generate_reuse(tiny.target,CheckedDraft(tiny.features),tiny.ids,max_new_tokens=limit,
        tree_config=BudgetTreeConfig(costs(),2,12,4,mode),verifier=verifier,on_tokens=published.extend)
    assert result.sequences.tolist()==tiny.reference[:,:4+limit].tolist()
    assert published==tiny.reference[0,4:4+limit].tolist() and verifier.spent==0


def test_root_only_standalone_selector_has_empty_integer_edge_indices():
    from memorydraft.backends.draft_tree import DraftTree
    tree=DraftTree(torch.tensor([[1]]),(-1,),(0,),1)
    path,recovery,decision=TreeASDVerifier().decide_tree(tree,torch.tensor([[0.,1.]]),horizon=0)
    assert path==[0] and recovery==1 and decision.regret==0


@pytest.mark.parametrize('state',[(-1,0,.25,2),(8,9,.25,2),(8,1,float('nan'),2),(8,1,.25,-1)])
def test_invalid_budget_snapshot(state):
    with pytest.raises(ValueError):RegretBudget(*state)


def test_missing_snapshot_is_explicit_error():
    with pytest.raises(ValueError):
        build_budget_tree(torch.zeros(1,4,dtype=torch.long),torch.zeros(1,3,8),
            BudgetTreeConfig(costs(),spine_mode="remaining"),10)
