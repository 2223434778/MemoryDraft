"""Best-first cost tree using each DSpark branch's own previous token."""
from __future__ import annotations

import heapq
import math
import torch

from memorydraft.backends.budget_tree import BudgetTreeConfig
from memorydraft.backends.draft_tree import DraftTree


def build_dspark_tree(anchor, base, draft, config: BudgetTreeConfig, context, *, stop_token_ids=()):
    if (anchor.numel()!=1 or base.ndim!=3 or base.shape[0]!=1 or not 0<base.shape[1]<=draft.block_size
            or base.shape[2]<config.top_k or config.spine_mode!='none'):
        raise ValueError('Invalid DSpark tree input/configuration')
    horizon=base.shape[1]
    tokens,parents,depths,masses=[int(anchor.item())],[-1],[0],[1.]
    queue=[]
    conditionals={}
    def expand(node):
        depth,previous=depths[node],tokens[node]
        if depth>=horizon or previous in stop_token_ids:return
        key=depth,previous
        if key not in conditionals:
            logits=draft.conditional_logits(base[:,depth],anchor.new_tensor([previous]))[0].float()
            values,ids=logits.topk(config.top_k)
            probabilities=(values-logits.logsumexp(-1)).exp().tolist()
            if any(not math.isfinite(p) or not 0<=p<=1 for p in probabilities):
                raise ValueError('Invalid conditional DSpark probabilities')
            conditionals[key]=list(zip(ids.tolist(),probabilities))
        for token,probability in conditionals[key]:
            heapq.heappush(queue,(-masses[node]*probability,depth+1,node,token))
    expand(0)
    total=1.
    best,best_score,best_mass=None,-math.inf,None
    while queue and len(tokens)<config.max_nodes:
        negative,depth,parent,token=heapq.heappop(queue)
        tokens.append(token);parents.append(parent);depths.append(depth);masses.append(-negative)
        total-=negative
        count=len(tokens)
        if count>=config.min_nodes:
            score=total/config.costs.cycle(context,count)
            if score>best_score:best,best_score,best_mass=count,score,total
        if count<config.max_nodes:expand(count-1)
    if best is None:best,best_mass=len(tokens),total
    tree=DraftTree(anchor.new_tensor([tokens[:best]]),tuple(parents[:best]),tuple(depths[:best]),1)
    tree.budget_metadata=dict(nodes=best,built_nodes=len(tokens),max_depth=max(depths[:best]),
        expected_advance=best_mass,estimated_cycle_seconds=config.costs.cycle(context,best),
        horizon=horizon,top_k=config.top_k,conditional_distributions=len(conditionals),proposal='DSpark-Markov')
    return tree
