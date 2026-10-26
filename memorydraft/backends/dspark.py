"""Pinned DSpark Markov drafter over the existing audited DFlash backbone.

Inference equations follow DeepSpec@005e03b81cec38b7da6399833d609ee89a2587f2
(MIT, Copyright 2026 The DeepSpec Authors; docs/third_party/DEEPSPEC_LICENSE.txt).
The two backbones have identical inference operations/state names. Anchor output
alignment and branch-conditioned Markov logits are specific to DSpark.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from importlib import resources

import torch
from torch import nn
from safetensors import safe_open

ASSET_MANIFEST=json.loads(resources.files('memorydraft').joinpath('assets/dspark-manifest.json').read_text())


class DSparkDraft(nn.Module):
    def __init__(self, backbone, markov_w1, markov_w2, *, block_size, mask_token_id, target_layer_ids):
        super().__init__()
        self.backbone=backbone
        self.markov_w1=markov_w1
        self.markov_w2=markov_w2
        self.block_size=int(block_size)
        self.mask_token_id=int(mask_token_id)
        self.target_layer_ids=list(target_layer_ids)

    def conditional_logits(self, base, previous):
        return base+self.markov_w2(self.markov_w1(previous))

    def greedy(self, base, anchor):
        previous=anchor.reshape(1)
        tokens=[]
        for depth in range(base.shape[1]):
            previous=self.conditional_logits(base[:,depth],previous).argmax(-1)
            tokens.append(previous)
        return torch.stack(tokens,1) if tokens else anchor.new_empty((1,0))

    def forward(self, **kwargs):
        return self.backbone(**kwargs)


def load_dspark(directory, target, audited_dflash):
    """Keep both existing model directories immutable; share only equal weights."""
    directory=Path(directory)
    manifest=json.loads((directory/'VERIFIED.json').read_bytes())
    if manifest['revision']!=ASSET_MANIFEST['revision']:
        raise ValueError('Unexpected DSpark checkpoint revision')
    weight=directory/'model.safetensors'
    h=hashlib.sha256()
    with weight.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    expected=ASSET_MANIFEST['files']['model.safetensors']
    if weight.stat().st_size!=expected['bytes'] or h.hexdigest()!=expected['sha256']:
        raise ValueError('DSpark weight checksum mismatch')
    config_path=directory/'config.json'
    if hashlib.sha256(config_path.read_bytes()).hexdigest()!=ASSET_MANIFEST['files']['config.json']['sha256']:
        raise ValueError('DSpark config checksum mismatch')
    raw=json.loads(config_path.read_bytes())
    if (raw['markov_head_type']!='vanilla' or raw['block_size']!=7 or raw['target_layer_ids']!=audited_dflash.target_layer_ids
            or raw['hidden_size']!=target.config.hidden_size or raw['vocab_size']!=target.config.vocab_size):
        raise ValueError('Unsupported DSpark/Target architecture')
    config=copy.deepcopy(audited_dflash.config)
    config.block_size=raw['block_size']
    config.dflash_config={'mask_token_id':raw['mask_token_id'],'target_layer_ids':raw['target_layer_ids']}
    for name in ('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads',
                 'head_dim','rms_norm_eps','attention_bias','layer_types','max_position_embeddings'):
        if getattr(config,name)!=raw[name]:raise ValueError('Backbone mismatch: '+name)
    if config.rope_theta!=raw['rope_parameters']['rope_theta']:
        raise ValueError('RoPE mismatch')
    with torch.device('meta'):
        backbone=type(audited_dflash)(config)
        w1=nn.Embedding(raw['vocab_size'],raw['markov_rank'])
        w2=nn.Linear(raw['markov_rank'],raw['vocab_size'],bias=False)
    # Rotary frequencies are non-persistent buffers, absent from state_dict.
    backbone.rotary_emb=type(audited_dflash.rotary_emb)(config)
    with safe_open(weight,framework='pt',device='cpu') as reader:
        # Official training freezes these Target weights. Verify rather than
        # silently replacing checkpoint embeddings/head with unrelated weights.
        for key,actual in [('embed_tokens.weight',target.model.embed_tokens.weight),('lm_head.weight',target.lm_head.weight)]:
            source=reader.get_slice(key)
            if tuple(source.get_shape())!=tuple(actual.shape):raise ValueError('Shared weight shape mismatch')
            for start in range(0,actual.shape[0],1024):
                if not torch.equal(source[start:start+1024].float(),actual[start:start+1024].detach().float().cpu()):
                    raise ValueError('DSpark checkpoint differs from Target: '+key)
        keys=set(backbone.state_dict())
        state={key:reader.get_tensor(key).float() for key in keys}
        extras=set(reader.keys())-keys-{'embed_tokens.weight','lm_head.weight','markov_head.markov_w1.weight',
                                    'markov_head.markov_w2.weight','confidence_head.proj.weight','confidence_head.proj.bias'}
        if extras:raise ValueError('Unexpected DSpark tensors: '+str(sorted(extras)))
        backbone.load_state_dict(state,strict=True,assign=True)
        w1.load_state_dict({'weight':reader.get_tensor('markov_head.markov_w1.weight').float()},assign=True)
        w2.load_state_dict({'weight':reader.get_tensor('markov_head.markov_w2.weight').float()},assign=True)
    return DSparkDraft(backbone,w1,w2,block_size=raw['block_size'],mask_token_id=raw['mask_token_id'],
                       target_layer_ids=raw['target_layer_ids']).to(target.device).eval()
