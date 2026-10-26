"""Equivalent DSpark trees with request-local Markov rows and safe search bounds."""
from __future__ import annotations

from collections import OrderedDict
import heapq
import math

import torch

from memorydraft.backends.budget_tree import BudgetTreeConfig
from memorydraft.backends.draft_tree import DraftTree


class MarkovRowCache:
    """Bounded, cold-per-request cache; every miss uses the original GEMV shape."""
    def __init__(self,draft,*,max_entries=64):
        if type(max_entries) is not int or max_entries<1:
            raise ValueError('Markov cache must have a positive entry bound')
        self.draft=draft
        self.max_entries=max_entries
        self.rows=OrderedDict()
        self.signature=self._signature()
        self.hits=self.misses=self.evictions=self.peak_entries=0

    def _signature(self):
        return tuple((id(w),w._version,w.device,w.dtype) for w in (self.draft.markov_w1.weight,self.draft.markov_w2.weight))

    @torch.inference_mode()
    def bias(self,previous,anchor):
        if self._signature()!=self.signature:
            raise ValueError('Markov weights changed while a request cache was active')
        if previous in self.rows:
            self.hits+=1
            self.rows.move_to_end(previous)
            return self.rows[previous]
        self.misses+=1
        bias=self.draft.markov_w2(self.draft.markov_w1(anchor.new_tensor([previous])))
        if len(self.rows)==self.max_entries:
            self.rows.popitem(last=False);self.evictions+=1
        self.rows[previous]=bias
        self.peak_entries=max(self.peak_entries,len(self.rows))
        return bias

    def summary(self):
        width=self.draft.markov_w2.out_features
        size=self.draft.markov_w2.weight.element_size()
        return dict(hits=self.hits,misses=self.misses,evictions=self.evictions,
                    max_entries=self.max_entries,peak_entries=self.peak_entries,peak_bytes=self.peak_entries*width*size)


def future_can_improve(total,count,best_score,next_mass,cycles,max_nodes):
    """Future masses cannot exceed the frontier maximum; round the bound upward."""
    for future in range(count+1,max_nodes+1):
        cost=cycles[future]
        if not math.isfinite(cost) or cost<=0:return True
        mass=total+(future-count)*next_mass
        # Covers sequential FP64 sum error plus product/division rounding.
        mass+=(max_nodes+2)*math.ulp(mass)
        if math.nextafter(mass/cost,math.inf)>best_score:return True
    return False


@torch.inference_mode()
def build_dspark_tree_fast(anchor,base,draft,config: BudgetTreeConfig,context,*,stop_token_ids=(),
                           cache=None,early_stop=True):
    if (anchor.numel()!=1 or base.ndim!=3 or base.shape[0]!=1 or not 0<base.shape[1]<=draft.block_size
            or base.shape[2]<config.top_k or config.spine_mode!='none'):
        raise ValueError('Invalid DSpark tree input/configuration')
    if cache is not None and cache.draft is not draft:raise ValueError('Cache belongs to another Draft')
    horizon=base.shape[1]
    tokens,parents,depths,masses=[int(anchor.item())],[-1],[0],[1.]
    queue=[];conditionals={}
    cycles={n:config.costs.cycle(context,n) for n in range(config.min_nodes,config.max_nodes+1)}
    def expandable(node):return depths[node]<horizon and tokens[node] not in stop_token_ids
    def expand(node):
        if not expandable(node):return
        depth,previous=depths[node],tokens[node]
        key=depth,previous
        if key not in conditionals:
            logits=(draft.conditional_logits(base[:,depth],anchor.new_tensor([previous])) if cache is None else
                    base[:,depth]+cache.bias(previous,anchor))[0].float()
            values,ids=logits.topk(config.top_k)
            probabilities=(values-logits.logsumexp(-1)).exp().tolist()
            if any(not math.isfinite(p) or not 0<=p<=1 for p in probabilities):
                raise ValueError('Invalid conditional DSpark probabilities')
            conditionals[key]=list(zip(ids.tolist(),probabilities))
        for token,probability in conditionals[key]:
            heapq.heappush(queue,(-masses[node]*probability,depth+1,node,token))
    expand(0)
    total=1.;best,best_score,best_mass=None,-math.inf,None
    bounded=False
    while queue and len(tokens)<config.max_nodes:
        negative,depth,parent,token=heapq.heappop(queue)
        tokens.append(token);parents.append(parent);depths.append(depth);masses.append(-negative)
        total-=negative;count=len(tokens)
        if count>=config.min_nodes:
            score=total/cycles[count]
            if score>best_score:best,best_score,best_mass=count,score,total
        if count<config.max_nodes:
            # Before querying this node, its mass bounds every possible child.
            upper=max(-queue[0][0] if queue else 0.,masses[-1] if expandable(count-1) else 0.)
            if early_stop and best is not None and not future_can_improve(total,count,best_score,upper,cycles,config.max_nodes):
                bounded=True;break
            expand(count-1)
            upper=-queue[0][0] if queue else 0.
            if early_stop and best is not None and not future_can_improve(total,count,best_score,upper,cycles,config.max_nodes):
                bounded=True;break
    if best is None:best,best_mass=len(tokens),total
    tree=DraftTree(anchor.new_tensor([tokens[:best]]),tuple(parents[:best]),tuple(depths[:best]),1)
    tree.budget_metadata=dict(nodes=best,built_nodes=len(tokens),max_depth=max(depths[:best]),
        expected_advance=best_mass,estimated_cycle_seconds=config.costs.cycle(context,best),horizon=horizon,
        top_k=config.top_k,conditional_distributions=len(conditionals),proposal='DSpark-Markov',search_bound_stopped=bounded)
    return tree
