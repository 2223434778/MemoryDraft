"""Bounded, offline DFlash adaptation: frozen teacher, anchored masked blocks."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math

import torch
import torch.nn.functional as F
from transformers import DynamicCache

SEED = 20261022
SOURCE_ROUNDS = (4, 12)
RANK, ALPHA, UPDATES, ACCUMULATE = 8, 16, 128, 2
ARMS = ("ordinary", "asd-prefix")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prefix_digest(ids) -> str:
    import json
    return digest(json.dumps(ids, separators=(",", ":")).encode())


def parameter_guard(model):
    return tuple((n, p.data_ptr(), p._version, tuple(p.shape), p.requires_grad) for n, p in model.named_parameters())


@torch.no_grad()
def teacher_example(target, prefix: torch.Tensor, layers, block_size: int, stops=()):
    """The prefix ends in the confirmed anchor; features strictly precede it."""
    if prefix.ndim != 2 or prefix.shape[0] != 1 or prefix.shape[1] < 2 or block_size < 2:
        raise ValueError("Expected a nonempty context plus anchor and proposal budget")
    cache = DynamicCache()
    positions = torch.arange(prefix.shape[1], device=prefix.device)[None]
    output = target(prefix, position_ids=positions, past_key_values=cache,
                    use_cache=True, output_hidden_states=True, logits_to_keep=1)
    context = torch.cat([output.hidden_states[i+1][:, :-1] for i in layers], -1).detach().cpu().clone()
    token = output.logits[:, -1].argmax(-1)
    labels = []
    for index in range(block_size-1):
        value = int(token.item())
        labels.append(value)
        if value in stops or index == block_size-2:
            break
        output = target(token[:, None], past_key_values=cache, use_cache=True, logits_to_keep=1)
        token = output.logits[:, -1].argmax(-1)
    return dict(context=context, anchor=prefix[:, -1:].detach().cpu().clone(),
                labels=torch.tensor([labels], dtype=torch.long), prefix_ids=prefix.detach().cpu().clone())


def position_loss(logits, labels, *, decay=7.):
    if (logits.ndim != 3 or labels.shape != logits.shape[:2] or not labels.numel()
            or not math.isfinite(decay) or decay <= 0 or not (labels != -100).any()):
        raise ValueError("Invalid masked-block supervision")
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                            reduction="none", ignore_index=-100).reshape_as(labels)
    weight = torch.exp(-torch.arange(labels.shape[1], device=logits.device, dtype=logits.dtype)/decay)
    weight = weight[None]*(labels != -100)
    return (losses*weight).sum()/weight.sum()


def block_logits(target, draft, sample):
    device = target.device
    size = draft.block_size
    context = sample["context"].to(device)
    if context.shape[1] != sample["prefix_ids"].shape[1]-1 or not 1 <= sample["labels"].shape[1] < size:
        raise ValueError("Features or labels do not align with the anchor")
    tokens = torch.full((1, size), draft.mask_token_id, dtype=torch.long, device=device)
    tokens[:, :1] = sample["anchor"].to(device)
    # Only the draft adapter receives gradients; teacher embedding/head are frozen.
    with torch.no_grad():
        noise = target.model.embed_tokens(tokens)
    positions = torch.arange(context.shape[1]+size, device=device)[None]
    hidden = draft(target_hidden=context, noise_embedding=noise, position_ids=positions,
                   past_key_values=None, use_cache=False, is_causal=False)
    return target.lm_head(hidden[:, -size+1:])


def sample_loss(target, draft, sample):
    logits = block_logits(target, draft, sample)
    labels = torch.full(logits.shape[:2], -100, device=logits.device, dtype=torch.long)
    labels[:, :sample["labels"].shape[1]] = sample["labels"].to(logits.device)
    return position_loss(logits, labels)


@torch.no_grad()
def evaluate_blocks(target, draft, examples):
    if not examples:
        raise ValueError("Empty adaptation calibration set")
    draft.eval()
    grouped = defaultdict(list)
    for info, sample in examples:
        logits = block_logits(target, draft, sample)
        n = sample["labels"].shape[1]
        labels = torch.full(logits.shape[:2], -100, device=logits.device, dtype=torch.long)
        labels[:, :n] = sample["labels"].to(logits.device)
        accepted = int((logits[:, :n].argmax(-1) == labels[:, :n]).cumprod(1).sum().item())
        loss = float(position_loss(logits, labels))
        if not math.isfinite(loss):
            raise ValueError("Nonfinite calibration loss")
        grouped[info["request_id"]].append((loss, accepted, n))
    means = [tuple(sum(x[i] for x in values)/len(values) for i in range(3)) for values in grouped.values()]
    return dict(requests=len(grouped), states=len(examples), loss=sum(x[0] for x in means)/len(means),
                strict_accepted=sum(x[1] for x in means)/len(means), labels=sum(x[2] for x in means)/len(means))


def attach_lora(target, draft):
    linears = [m for n, m in draft.named_modules() if n.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}]
    if not linears or any(type(m) is not torch.nn.Linear or m.weight.dtype != torch.float32 for m in linears):
        raise ValueError("This adapter experiment requires native FP32 Linear projections")
    import peft
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import model as lora_module
    from unittest.mock import patch
    if peft.__version__ != "0.11.1":
        raise RuntimeError("Bounded training pins PEFT 0.11.1")
    target.requires_grad_(False)
    draft.requires_grad_(False)
    torch.manual_seed(SEED)
    # Generic PEFT forwarding preserves DFlash's custom target_hidden interface.
    # PEFT 0.11 probes an unrelated system bitsandbytes installation even for
    # native FP32 Linear. Scope the dispatch choice to these validated layers;
    # no package or CUDA setting changes, and both optional probes are restored.
    with patch.object(lora_module, "is_bnb_available", return_value=False), \
            patch.object(lora_module, "is_bnb_4bit_available", return_value=False):
        wrapped = get_peft_model(draft, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.,
                                 target_modules=["q_proj", "v_proj"], bias="none"))
    return wrapped


class AdapterBank:
    """Offline dense merges; no extra LoRA branches in timed inference."""
    def __init__(self, draft):
        self.draft = draft
        self.original = {n: m.weight.detach().cpu().clone() for n, m in draft.named_modules()
                         if n.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}}
        if not self.original:
            raise ValueError("No supported draft projections")

    @torch.no_grad()
    def select(self, state=None):
        updates = {}
        if state is not None:
            for name, weight in self.original.items():
                prefix = "base_model.model."+name
                akey, bkey = prefix+".lora_A.weight", prefix+".lora_B.weight"
                if akey not in state or bkey not in state:
                    raise ValueError("Missing adapter projection: "+name)
                a, b = state[akey], state[bkey]
                if (a.shape != (RANK, weight.shape[1]) or b.shape != (weight.shape[0], RANK)
                        or a.dtype != weight.dtype or b.dtype != weight.dtype
                        or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
                    raise ValueError("Invalid adapter projection shape, dtype or values")
                updates[name] = (a, b)
            if len(state) != 2*len(self.original):
                raise ValueError("Unexpected adapter tensors")
        for name, weight in self.original.items():
            parameter = self.draft.get_submodule(name).weight
            value = weight.to(parameter.device)
            if name in updates:
                a, b = updates[name]
                value = value+(b.to(parameter.device) @ a.to(parameter.device))*(ALPHA/RANK)
            parameter.copy_(value)
