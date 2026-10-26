import itertools
from unittest.mock import patch

import pytest
import torch
from torch import nn

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.backends.budget_tree import BudgetTreeConfig, TreeASDVerifier
from memorydraft.backends.dspark import DSparkDraft
from memorydraft.backends.dspark_tree import build_dspark_tree
from memorydraft.backends.dspark_reuse import generate_dspark
from tests.test_budget_tree import costs
from tests.test_dflash_reuse import CheckedDraft, tiny


def small_draft(backbone=None):
    w1=nn.Embedding(4,4);w2=nn.Linear(4,4,bias=False)
    with torch.no_grad():
        w1.weight.copy_(torch.eye(4))
        w2.weight.copy_(torch.tensor([[0.,0.,0.,0.],[2.,3.,0.,0.],[1.,0.,0.,1.],[0.,0.,4.,2.]]))
    return DSparkDraft(backbone or nn.Identity(),w1,w2,block_size=4,mask_token_id=0,target_layer_ids=[0,1])


def test_markov_tree_matches_exhaustive_conditional_path_mass():
    draft=small_draft();base=torch.zeros(1,3,4);config=BudgetTreeConfig(costs(),12,12,2)
    tree=build_dspark_tree(torch.tensor([0]),base,draft,config,10)
    candidates={():1.}
    for depth in range(3):
        for path,mass in list(candidates.items()):
            if len(path)!=depth:continue
            probs=draft.conditional_logits(base[:,depth],torch.tensor([path[-1] if path else 0]))[0].softmax(-1)
            for token in probs.topk(2).indices.tolist():candidates[(*path,token)]=mass*float(probs[token])
    paths=[()]
    for i,parent in enumerate(tree.parents[1:],1):paths.append((*paths[parent],int(tree.tokens[0,i])))
    actual=sorted(candidates[p] for p in paths)
    assert actual==pytest.approx(sorted(sorted(candidates.values(),reverse=True)[:12]),rel=2e-6)
    assert (2,3) in paths
    assert tree.budget_metadata['proposal']=='DSpark-Markov'


def test_dspark_greedy_matches_sequential_conditional_argmax():
    draft=small_draft();base=torch.zeros(1,3,4)
    assert draft.greedy(base,torch.tensor([0])).tolist()==[[1,1,1]]
    assert draft.greedy(base,torch.tensor([2])).tolist()==[[3,3,3]]


def test_dspark_tree_eos_is_leaf_and_budget_ties_use_existing_rule():
    draft=small_draft();base=torch.zeros(1,3,4)
    tree=build_dspark_tree(torch.tensor([0]),base,draft,BudgetTreeConfig(costs(),2,12,2),10,stop_token_ids=(1,))
    assert all(int(tree.tokens[0,p])!=1 for p in tree.parents[1:])
    logits=torch.zeros(len(tree.parents),4)
    verifier=TreeASDVerifier(ASDConfig(budget=.25,gate=100,cap=1))
    path,_,decision=verifier.decide_tree(tree,logits,horizon=3,stop_token_ids=(1,))
    assert decision.relaxed<=1 and verifier.spent<=.25
    assert all(tree.parents[child]==parent for parent,child in zip(path,path[1:]))


def tiny_dspark(tiny):
    w1=nn.Embedding(32,2);w2=nn.Linear(2,32,bias=False)
    nn.init.zeros_(w1.weight);nn.init.zeros_(w2.weight)
    return DSparkDraft(CheckedDraft(tiny.features),w1,w2,block_size=4,mask_token_id=31,target_layer_ids=[0,1])


@pytest.mark.parametrize('limit',[0,1,2,9,29])
@pytest.mark.parametrize('tree',[False,True])
def test_dspark_zero_budget_real_target_and_feature_frontiers(tiny,limit,tree):
    draft=tiny_dspark(tiny)
    config=BudgetTreeConfig(costs(),2,12,4) if tree else None
    verifier=TreeASDVerifier(ASDConfig(budget=0)) if tree else ASDVerifier(ASDConfig(budget=0))
    result=generate_dspark(tiny.target,draft,tiny.ids,max_new_tokens=limit,verifier=verifier,tree_config=config)
    assert result.sequences.tolist()==tiny.reference[:,:4+limit].tolist()
    if limit:
        eos=int(tiny.reference[0,4])
        stopped=generate_dspark(tiny.target,tiny_dspark(tiny),tiny.ids,max_new_tokens=limit,
                               stop_token_ids=(eos,),verifier=verifier,tree_config=config)
        assert stopped.sequences.tolist()==tiny.reference[:,:5].tolist()


def test_dspark_nonzero_tree_commits_only_realized_path_kv(tiny):
    class Draft(nn.Module):
        def forward(self,*,noise_embedding,target_hidden,past_key_values,**kwargs):
            length=noise_embedding.shape[1]+target_hidden.shape[1]
            values=noise_embedding.new_zeros((1,1,length,8))
            past_key_values.update(values,values,0)
            return torch.zeros_like(noise_embedding)
    draft=tiny_dspark(tiny);draft.backbone=Draft()
    caches=[];forward=tiny.target.forward
    def observe(*args,**kwargs):
        if kwargs.get('past_key_values') is not None:caches.append(kwargs['past_key_values'])
        return forward(*args,**kwargs)
    verifier=TreeASDVerifier(ASDConfig(budget=8,gate=100,cap=3))
    with patch.object(tiny.target,'forward',observe):
        result=generate_dspark(tiny.target,draft,tiny.ids,max_new_tokens=29,verifier=verifier,
                               tree_config=BudgetTreeConfig(costs(),2,12,4))
    assert verifier.relaxed>0 and 0<=verifier.spent<=8
    cache=caches[-1]
    assert cache.get_seq_length()==result.sequences.shape[1]-1
    with torch.inference_mode():
        actual=tiny.target(result.sequences[:,-1:],past_key_values=cache,use_cache=True)
        expected=tiny.target(result.sequences)
    torch.testing.assert_close(actual.logits[:,-1],expected.logits[:,-1],atol=2e-6,rtol=2e-5)


def test_dspark_tree_rejects_unsupported_joint_and_malformed_inputs():
    draft=small_draft()
    with pytest.raises(ValueError):
        build_dspark_tree(torch.tensor([0]),torch.zeros(1,3,4),draft,BudgetTreeConfig(costs(),spine_mode='full'),10)
    with pytest.raises(ValueError):
        build_dspark_tree(torch.tensor([0]),torch.zeros(1,5,4),draft,BudgetTreeConfig(costs()),10)
