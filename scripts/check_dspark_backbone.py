"""Compare the adapter against pinned, separately supplied upstream inference code."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import typing

import torch
from torch import nn
from transformers import DynamicCache
from transformers.models.qwen3 import modeling_qwen3
from memorydraft.backends.dflash_reuse import _features


def check(engine,draft,directory):
    directory=Path(directory)
    expected={'modeling.py':'24d44416f88e6215d7c7a9811f60ca65d883b62f',
              'markov_head.py':'9a43beba31be00f6c8e350acfc5adb180f25438a'}
    texts={};hashes={}
    for name,oid in expected.items():
        data=(directory/name).read_bytes()
        if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=oid:
            raise ValueError('Upstream reference file differs from pinned Git blob: '+name)
        texts[name]=data.decode();hashes[name]=hashlib.sha256(data).hexdigest()
    scope=dict(vars(modeling_qwen3));scope.update(__name__='memorydraft_dspark_upstream_check',
        torch=torch,nn=nn,Optional=typing.Optional,Callable=typing.Callable,Tuple=typing.Tuple,
        DSparkForwardOutput=SimpleNamespace,sample_tokens=lambda logits,temperature:logits.argmax(-1))
    class Predictor(nn.Module):
        def __init__(self,input_dim):
            super().__init__();self.proj=nn.Linear(input_dim,1)
    scope['AcceptRatePredictor']=Predictor
    for name in ('markov_head.py','modeling.py'):
        source=ast.parse(texts[name]);body=[n for n in source.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        exec(compile(ast.Module(body=body,type_ignores=[]),name,'exec'),scope)
    config=draft.backbone.config
    for key,value in dict(target_layer_ids=draft.target_layer_ids,mask_token_id=draft.mask_token_id,num_anchors=512,
        enable_confidence_head=True,confidence_head_with_markov=True,markov_rank=draft.markov_w1.embedding_dim,
        markov_head_type='vanilla').items():setattr(config,key,value)
    with torch.device('meta'):reference=scope['Qwen3DSparkModel'](config)
    reference.rotary_emb=type(draft.backbone.rotary_emb)(config).to(engine.target.device)
    missing=reference.load_state_dict(draft.backbone.state_dict(),strict=False,assign=True)
    if missing.unexpected_keys:raise ValueError('Unexpected reference backbone parameters')
    reference.markov_head.load_state_dict({'markov_w1.weight':draft.markov_w1.weight,
                                           'markov_w2.weight':draft.markov_w2.weight},assign=True)
    reference.eval()
    cases=[]
    with torch.inference_mode():
        for length in (4,17):
            ids=torch.full((1,length),27,dtype=torch.long,device=engine.target.device)
            output=engine.target(ids,output_hidden_states=True,use_cache=False)
            features=_features(output,draft.target_layer_ids)
            noise=ids.new_full((1,draft.block_size),draft.mask_token_id);noise[:,0]=output.logits[:,-1].argmax(-1)
            positions=torch.arange(length+draft.block_size,device=ids.device).unsqueeze(0)
            actual=draft(target_hidden=features,noise_embedding=engine.target.model.embed_tokens(noise),
                position_ids=positions,past_key_values=DynamicCache(),use_cache=True,is_causal=False)
            target=reference._forward_backbone(target_hidden_states=features,noise_embedding=engine.target.model.embed_tokens(noise),
                position_ids=positions,past_key_values=DynamicCache(),use_cache=True,is_causal=False)
            torch.testing.assert_close(actual,target,atol=2e-6,rtol=2e-5)
            base=engine.target.lm_head(actual)
            ref_tokens,_=reference.sample_draft_tokens(base,first_prev_token_ids=noise[:,0],temperature=0.,hidden_states=actual)
            tokens=draft.greedy(base,noise[:,0])
            if not torch.equal(ref_tokens,tokens):raise ValueError('Markov greedy token mismatch')
            cases.append(dict(context=length,max_hidden_difference=float((actual-target).abs().max()),
                              greedy_token_sha256=hashlib.sha256(bytes(str(tokens.tolist()),'utf-8')).hexdigest()))
    return dict(upstream_commit='005e03b81cec38b7da6399833d609ee89a2587f2',reference_source_sha256=hashes,
                status='passed',cases=cases)
