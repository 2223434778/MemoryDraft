"""Greedy R/DFlash adapter, derived from the audited Z Lab MIT decode loop.

Upstream: z-lab/dflash@07ebd93db9f472af339b644bb70221ad8428328a.
Model snapshot: 9b41424b7109f9c5413454f481b09a82b85333f4/dflash.py.
Copyright (c) 2026 Z Lab. Full notice: docs/third_party/DFLASH_LICENSE.txt.
Changes: candidate routing, bounded output, valid-feature accumulation and traces.
The model forward, attention, strict longest-prefix verification and KV crop are reused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from transformers import DynamicCache

from memorydraft.retrieval.reuse import ReuseMatcher


@dataclass(frozen=True)
class ReuseConfig:
    max_tokens: int = 32
    min_match: int = 6
    cooldown_rounds: int = 1
    max_pending_tokens: int = 1024

    def __post_init__(self):
        if self.max_tokens < 1 or self.min_match < 1 or self.cooldown_rounds < 0 or self.max_pending_tokens < 1:
            raise ValueError("Invalid reuse configuration")


@dataclass
class ReuseGeneration:
    sequences: torch.Tensor
    rounds: list[dict] = field(default_factory=list)
    feature_peak_tokens: int = 0
    buffer_flushes: int = 0


def _features(output, layers):
    return torch.cat([output.hidden_states[layer + 1] for layer in layers], dim=-1)


@torch.inference_mode()
def generate_reuse(target, draft, input_ids: torch.Tensor, *, max_new_tokens: int,
                   stop_token_ids: Sequence[int] = (), matcher: ReuseMatcher | None = None,
                   config: ReuseConfig = ReuseConfig(), trace: bool = True,
                   route_sequence: Sequence[str] = ()) -> ReuseGeneration:
    """Single-request strict greedy decoding. route_sequence is a diagnostic override.

At each loop entrance target KV covers [0,start), and output[start] is the
already confirmed but uncached anchor. Draft KV plus pending valid features
also cover [0,start). R skips draft forward and accumulates those features.
"""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.shape[1]:
        raise ValueError("Expected one nonempty token sequence")
    if max_new_tokens < 0 or any(route not in {"R", "D"} for route in route_sequence):
        raise ValueError("Invalid generation budget or diagnostic route")
    if max_new_tokens == 0:
        return ReuseGeneration(input_ids.clone())
    if draft.block_size < 2:
        raise ValueError("DFlash requires an anchor and at least one proposal")
    target.eval()
    draft.eval()
    prompt = input_ids.shape[1]
    limit = prompt + max_new_tokens
    extra = max(config.max_tokens + 1, draft.block_size) + 1
    ids = torch.full((1, limit + extra), draft.mask_token_id, device=input_ids.device, dtype=torch.long)
    ids[:, :prompt] = input_ids
    positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    target_cache, draft_cache = DynamicCache(), DynamicCache()
    output = target(input_ids, position_ids=positions[:, :prompt], past_key_values=target_cache,
                    use_cache=True, logits_to_keep=1, output_hidden_states=True)
    ids[:, prompt:prompt + 1] = output.logits.argmax(dim=-1)
    pending = [_features(output, draft.target_layer_ids)]
    pending_tokens = prompt
    result = ReuseGeneration(ids, feature_peak_tokens=prompt)
    start, blocked_until, step = prompt, 0, 0
    stops = set(stop_token_ids)
    if int(ids[0, prompt]) in stops:
        result.sequences = ids[:, :prompt + 1].clone()
        return result
    while start + 1 < limit:
        if target_cache.get_seq_length() != start or draft_cache.get_seq_length() + pending_tokens != start:
            raise RuntimeError("R/DFlash feature and KV frontiers diverged")
        forced = route_sequence[step % len(route_sequence)] if route_sequence else None
        can_reuse = matcher is not None and step >= blocked_until and forced != "D"
        flush = can_reuse and pending_tokens >= config.max_pending_tokens
        if flush:
            can_reuse = False
            result.buffer_flushes += 1
        confirmed = None
        proposal = None
        if can_reuse:
            confirmed = ids[0, prompt:start + 1].tolist()
            proposal = matcher.propose(confirmed, min(config.max_tokens, limit - start - 1))
            if proposal is not None and proposal.matched < config.min_match and forced != "R":
                proposal = None
        before_draft = draft_cache.get_seq_length()
        before_pending = pending_tokens
        if proposal is not None:
            route = "R"
            block = torch.cat([ids[:, start:start + 1], ids.new_tensor([proposal.tokens])], dim=1)
        else:
            route = "D"
            block = ids[:, start:start + draft.block_size].clone()
            # Clear stale proposals left in the output workspace by previous rounds.
            block[:, 1:] = draft.mask_token_id
            context = pending[0] if len(pending) == 1 else torch.cat(pending, dim=1)
            hidden = draft(target_hidden=context, noise_embedding=target.model.embed_tokens(block),
                           position_ids=positions[:, before_draft:start + draft.block_size],
                           past_key_values=draft_cache, use_cache=True, is_causal=False)
            logits = target.lm_head(hidden[:, -draft.block_size + 1:, :])
            block[:, 1:] = logits.argmax(dim=-1)
            draft_cache.crop(start)
            pending.clear()
            pending_tokens = 0
            block = block[:, :min(block.shape[1], limit - start)]
        output = target(block, position_ids=positions[:, start:start + block.shape[1]],
                        past_key_values=target_cache, use_cache=True, output_hidden_states=True)
        posterior = output.logits.argmax(dim=-1)
        accepted = int((block[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum().item())
        advance = accepted + 1
        ids[:, start:start + advance] = block[:, :advance]
        ids[:, start + advance] = posterior[:, accepted]
        old_start = start
        start += advance
        target_cache.crop(start)
        pending.append(_features(output, draft.target_layer_ids)[:, :advance, :].clone())
        pending_tokens += advance
        result.feature_peak_tokens = max(result.feature_peak_tokens, pending_tokens)
        if proposal is not None:
            matcher.observe(proposal, accepted, [*confirmed, *proposal.tokens[:accepted]])
            if accepted < len(proposal.tokens):
                blocked_until = step + config.cooldown_rounds + 1
        if trace:
            result.rounds.append(dict(route=route, generated_before=old_start - prompt + 1,
                                      proposed=block.shape[1] - 1, accepted=accepted, advance=advance,
                                      draft_cache_before=before_draft, pending_before=before_pending,
                                      pending_after=pending_tokens, target_cache_after=start,
                                      source=proposal.source if proposal else None,
                                      lookup=proposal.lookup if proposal else None,
                                      matched=proposal.matched if proposal else 0, buffer_flush=bool(flush)))
        step += 1
        available = min(start + 1, limit)
        if stops:
            generated = ids[0, prompt:available].tolist()
            ended = next((i for i, token in enumerate(generated) if token in stops), None)
            if ended is not None:
                result.sequences = ids[:, :prompt + ended + 1].clone()
                return result
    result.sequences = ids[:, :min(start + 1, limit)].clone()
    return result
