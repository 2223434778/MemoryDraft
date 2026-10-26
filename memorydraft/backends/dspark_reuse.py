"""DSpark-only thin adapter of the audited R/DFlash decode/cache loop.

The existing DFlash implementation is unchanged. Tree attention, path selection,
ASD ledger and selected-path KV commit reuse the same tested components.
"""
from __future__ import annotations

import torch
from transformers import DynamicCache

from memorydraft.backends.dflash_reuse import ReuseGeneration, _features
from memorydraft.backends.draft_tree import commit_tree_cache
from memorydraft.backends.dspark_tree import build_dspark_tree


@torch.inference_mode()
def generate_dspark(target,draft,input_ids,*,max_new_tokens,stop_token_ids=(),verifier=None,tree_config=None,
                    on_draft=None,on_tokens=None,should_stop=None):
    if input_ids.ndim!=2 or input_ids.shape[0]!=1 or not input_ids.shape[1] or max_new_tokens<0:
        raise ValueError('Expected one nonempty prompt and a nonnegative generation limit')
    if target.config._attn_implementation!='eager':raise ValueError('Use audited eager Target attention')
    if max_new_tokens==0:return ReuseGeneration(input_ids.clone())
    target.eval();draft.eval()
    prompt=input_ids.shape[1];limit=prompt+max_new_tokens
    ids=input_ids.new_full((1,limit+draft.block_size+2),draft.mask_token_id)
    ids[:,:prompt]=input_ids
    positions=torch.arange(ids.shape[1],device=ids.device).unsqueeze(0)
    target_cache,draft_cache=DynamicCache(),DynamicCache()
    output=target(input_ids,past_key_values=target_cache,position_ids=positions[:,:prompt],use_cache=True,
                  output_hidden_states=True,logits_to_keep=1)
    ids[:,prompt]=output.logits[:,-1].argmax(-1)
    features=_features(output,draft.target_layer_ids)
    result=ReuseGeneration(ids,feature_peak_tokens=prompt)
    start=prompt
    stops=set(stop_token_ids)
    published=prompt
    def publish(end):
        nonlocal published
        if on_tokens and end>published:
            tokens=ids[0,published:end].tolist()
            stop=next((i+1 for i,token in enumerate(tokens) if token in stops),len(tokens))
            on_tokens(tokens[:stop]);published+=stop
    publish(prompt+1)
    while start+1<limit and int(ids[0,start]) not in stops:
        if should_stop and should_stop():
            from memorydraft.runtime.contracts import RequestCancelled
            raise RequestCancelled('DSpark request cancelled')
        if target_cache.get_seq_length()!=start or draft_cache.get_seq_length()+features.shape[1]!=start:
            raise RuntimeError('DSpark cache and realized feature frontiers diverged')
        noise=ids.new_full((1,draft.block_size),draft.mask_token_id)
        noise[:,0]=ids[:,start]
        hidden=draft(target_hidden=features,noise_embedding=target.model.embed_tokens(noise),
                     position_ids=positions[:,draft_cache.get_seq_length():start+draft.block_size],
                     past_key_values=draft_cache,use_cache=True,is_causal=False)
        draft_cache.crop(start)
        base=target.lm_head(hidden[:,:min(draft.block_size,limit-start-1)])
        anchor=ids[:,start]
        if on_draft is not None:on_draft(anchor,base)
        tree=None
        if tree_config is None:
            proposal=draft.greedy(base,anchor)
            block=torch.cat((anchor.reshape(1,1),proposal),dim=1)
            verify_positions=positions[:,start:start+block.shape[1]]
            kwargs={}
        else:
            tree=build_dspark_tree(anchor,base,draft,tree_config,start,stop_token_ids=stops)
            block=tree.tokens
            verify_positions=ids.new_tensor([tree.depths])+start
            kwargs={'attention_mask':tree.mask(start,target.dtype)}
        output=target(block,position_ids=verify_positions,past_key_values=target_cache,
                      use_cache=True,output_hidden_states=True,**kwargs)
        posterior=output.logits.argmax(-1)
        decision=None
        if tree is not None:
            if verifier is None:path,recovery,_=tree.select(posterior)
            else:path,recovery,decision=verifier.decide_tree(tree,output.logits[0],horizon=base.shape[1],stop_token_ids=stops)
            accepted=len(path)-1
            committed=tree.tokens[:,path]
            valid_features=_features(output,draft.target_layer_ids)[:,path]
            commit_tree_cache(target_cache,start,path)
        else:
            if verifier is None:accepted=int((block[:,1:]==posterior[:,:-1]).cumprod(1).sum())
            else:
                decision=verifier.decide(block[0,1:],output.logits[0,:-1],stop_token_ids=stops)
                accepted=decision.accepted
            recovery=int(posterior[0,accepted])
            committed=block[:,:accepted+1]
            valid_features=_features(output,draft.target_layer_ids)[:,:accepted+1].clone()
        advance=accepted+1
        ids[:,start:start+advance]=committed
        ids[:,start+advance]=recovery
        old_start=start;start+=advance
        target_cache.crop(start)
        features=valid_features
        result.feature_peak_tokens=max(result.feature_peak_tokens,features.shape[1])
        row=dict(route='D',generated_before=old_start-prompt+1,proposed=base.shape[1],accepted=accepted,advance=advance,
                 draft_forward=True,target_cache_after=start)
        if decision is not None:row.update(strict_accepted=decision.strict_accepted,relaxed_tokens=decision.relaxed,regret=decision.regret)
        if tree is not None:row.update(tree_nodes=len(tree.parents),tree_path=path,tree_budget=tree.budget_metadata)
        result.rounds.append(row)
        available=min(start+1,limit)
        publish(available)
        if any(int(v) in stops for v in ids[0,old_start:available]):break
    end=min(start+1,limit)
    stop=next((prompt+i+1 for i,v in enumerate(ids[0,prompt:end].tolist()) if v in stops),None)
    result.sequences=ids[:,:stop or end].clone()
    return result
