"""Greedy R/DFlash adapter, derived from the audited Z Lab MIT decode loop.

Upstream: z-lab/dflash@07ebd93db9f472af339b644bb70221ad8428328a.
Model snapshot: 9b41424b7109f9c5413454f481b09a82b85333f4/dflash.py.
Copyright (c) 2026 Z Lab. Full notice: docs/third_party/DFLASH_LICENSE.txt.
Changes: candidate routing, bounded output, valid-feature accumulation and traces.
The model forward, attention, strict longest-prefix verification and KV crop are reused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import Callable, Sequence

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
                   route_sequence: Sequence[str] = (),
                   on_tokens: Callable[[list[int]], None] | None = None,
                   should_stop: Callable[[], bool] | None = None,
                   on_draft: Callable[[list[int], list[int], torch.Tensor, int], None] | None = None,
                   verifier=None,
                   profile: bool = False,
                   draft_block_size: int | None = None,
                   verify_length: Callable[[int], int] | None = None,
                   incremental_stop: bool = False,
                   round_action: Callable[[int, int, tuple], str] | None = None,
                   tree_config=None, prefill=None,
                   on_event: Callable[[dict], None] | None = None) -> ReuseGeneration:
    """Single-request decoding; strict unless an experimental verifier is supplied.

At each loop entrance target KV covers [0,start), and output[start] is the
already confirmed but uncached anchor. Draft KV plus pending valid features
also cover [0,start). Exact R skips draft forward and accumulates those features;
guided R may replace a block after its draft forward has already run.
"""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.shape[1]:
        raise ValueError("Expected one nonempty token sequence")
    if max_new_tokens < 0 or any(route not in {"R", "D"} for route in route_sequence):
        raise ValueError("Invalid generation budget or diagnostic route")
    if round_action is not None and (matcher is not None or route_sequence or verify_length is not None
                                     or draft_block_size is not None or not trace):
        raise ValueError("Round allocation requires traced, exclusive model drafting")
    if tree_config is not None:
        from memorydraft.backends.draft_tree import TreeConfig
        from memorydraft.backends.budget_tree import BudgetTreeConfig, TreeASDVerifier
        if (not isinstance(tree_config, (TreeConfig, BudgetTreeConfig)) or matcher is not None
                or (verifier is not None and not isinstance(verifier, TreeASDVerifier))
                or round_action is not None or route_sequence or verify_length is not None or draft_block_size is not None):
            raise ValueError("Draft tree requires exclusive native drafting and a tree-aware verifier")
        if target.config._attn_implementation != "eager":
            raise ValueError("Draft tree requires the audited eager attention backend")
    if max_new_tokens == 0:
        return ReuseGeneration(input_ids.clone())
    if should_stop and should_stop():
        from memorydraft.runtime.contracts import RequestCancelled
        raise RequestCancelled("Request cancelled before prefill")
    block_size = draft.block_size if draft_block_size is None else draft_block_size
    if type(block_size) is not int or not 2 <= block_size <= draft.block_size:
        raise ValueError("DFlash requires an anchor and at least one proposal")
    target.eval()
    draft.eval()
    prompt = input_ids.shape[1]
    limit = prompt + max_new_tokens
    extra = max(config.max_tokens + 1, block_size) + 1
    ids = torch.full((1, limit + extra), draft.mask_token_id, device=input_ids.device, dtype=torch.long)
    ids[:, :prompt] = input_ids
    positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    target_cache, draft_cache = DynamicCache(), DynamicCache()
    def mark(name):
        return torch.profiler.record_function("memorydraft::" + name) if profile else nullcontext()

    with mark("target_prefill"):
        if prefill is None:
            output = target(input_ids, position_ids=positions[:, :prompt], past_key_values=target_cache,
                            use_cache=True, logits_to_keep=1, output_hidden_states=True)
            first_logits, features = output.logits, _features(output, draft.target_layer_ids)
        else:
            first_logits, features, target_cache = prefill(
                target, input_ids, positions[:, :prompt], draft.target_layer_ids, should_stop)
    ids[:, prompt:prompt + 1] = first_logits.argmax(dim=-1)
    pending = [features]
    del features, first_logits
    pending_tokens = prompt
    result = ReuseGeneration(ids, feature_peak_tokens=prompt)
    start, blocked_until, step = prompt, 0, 0
    stops = set(stop_token_ids)
    published = prompt

    def publish(end):
        nonlocal published
        if on_tokens and end > published:
            tokens = ids[0, published:end].tolist()
            ended = next((i for i, token in enumerate(tokens) if token in stops), None)
            if ended is not None:
                tokens = tokens[:ended + 1]
            on_tokens(tokens)
            published += len(tokens)

    publish(prompt + 1)
    if int(ids[0, prompt]) in stops:
        result.sequences = ids[:, :prompt + 1].clone()
        return result
    while start + 1 < limit:
        if should_stop and should_stop():
            from memorydraft.runtime.contracts import RequestCancelled
            raise RequestCancelled("Request cancelled between verification rounds")
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
            with mark("retrieval_lookup"):
                confirmed = ids[0, prompt:start + 1].tolist()
                proposal = matcher.propose(confirmed, min(config.max_tokens, limit - start - 1))
            if (proposal is not None and proposal.matched < config.min_match
                    and not proposal.context_qualified and forced != "R"):
                proposal = None
        before_draft = draft_cache.get_seq_length()
        before_pending = pending_tokens
        current_block, allocated_verify, allocation = block_size, None, None
        if round_action is not None:
            from memorydraft.backends.allocation import action_spec
            allocation = round_action(step, start - prompt + 1, tuple(result.rounds[-4:]))
            current_block, allocated_verify = action_spec(allocation, block_size)
            # Bound deferred feature storage even for a policy that never drafts.
            if allocation == "target" and pending_tokens >= config.max_pending_tokens:
                allocation = "native"
                current_block, allocated_verify = action_spec(allocation, block_size)
        draft_forward = proposal is None and allocation != "target"
        if allocation == "target":
            route = "A"
            block = ids[:, start:start + 1].clone()
        elif proposal is not None:
            route = "R"
            block = torch.cat([ids[:, start:start + 1], ids.new_tensor([proposal.tokens])], dim=1)
        else:
            route = "D"
            block = ids[:, start:start + current_block].clone()
            # Clear stale proposals left in the output workspace by previous rounds.
            block[:, 1:] = draft.mask_token_id
            context = pending[0] if len(pending) == 1 else torch.cat(pending, dim=1)
            with mark("draft_forward"):
                hidden = draft(target_hidden=context, noise_embedding=target.model.embed_tokens(block),
                               position_ids=positions[:, before_draft:start + current_block],
                               past_key_values=draft_cache, use_cache=True, is_causal=False)
                logits = target.lm_head(hidden[:, -current_block + 1:, :])
                block[:, 1:] = logits.argmax(dim=-1)
            draft_cache.crop(start)
            pending.clear()
            pending_tokens = 0
            block = block[:, :min(block.shape[1], limit - start)]
            if on_draft is not None:
                # Optional diagnostic observer; no extra Target forward. The
                # callback must not mutate the live logits or decoding state.
                on_draft(ids[0, prompt:start + 1].tolist(), block[0, 1:].tolist(),
                         logits[0, :block.shape[1] - 1].detach(), min(config.max_tokens, limit - start - 1))
            guided = getattr(matcher, "propose_with_draft", None) if can_reuse else None
            if guided is not None:
                with mark("guided_retrieval"):
                    receive_logits = getattr(matcher, "set_draft_logits", None)
                    if receive_logits is not None:
                        receive_logits(logits[0, :block.shape[1] - 1].detach())
                    proposal = guided(confirmed, block[0, 1:].tolist(),
                                      min(config.max_tokens, limit - start - 1))
                if proposal is not None:
                    route = "R"
                    block = torch.cat([ids[:, start:start + 1], ids.new_tensor([proposal.tokens])], dim=1)
        if allocated_verify is not None and allocation != "target":
            block = block[:, :min(block.shape[1], allocated_verify + 1)]
        if verify_length is not None:
            length = verify_length(block.shape[1] - 1)
            if type(length) is not int or not 1 <= length < block.shape[1]:
                raise ValueError("Verification length must select a nonempty proposal prefix")
            block = block[:, :length + 1]
        tree = None
        verify_kwargs = {}
        verify_block = block
        if tree_config is not None:
            from memorydraft.backends.draft_tree import build_tree, commit_tree_cache
            with mark("build_tree"):
                if isinstance(tree_config, BudgetTreeConfig):
                    from memorydraft.backends.budget_tree import build_budget_tree
                    from memorydraft.backends.branch_drift import PairedTreeConfig, build_paired_tree
                    snapshot = None
                    if tree_config.spine_mode != "none":
                        if not isinstance(verifier, TreeASDVerifier):
                            raise ValueError("Budget-dependent tree construction requires a tree ASD verifier")
                        snapshot = verifier.construction_budget()
                    builder = build_paired_tree if isinstance(tree_config, PairedTreeConfig) else build_budget_tree
                    tree = builder(block, logits, tree_config, start,
                        stop_token_ids=stop_token_ids, regret_budget=snapshot)
                else:
                    tree = build_tree(block, logits, tree_config)
                verify_block = tree.tokens
                verify_positions = block.new_tensor([tree.depths])+start
                verify_kwargs["attention_mask"] = tree.mask(start, target.dtype)
        else:
            verify_positions = positions[:, start:start + block.shape[1]]
        if on_event is not None:
            on_event(dict(phase="draft", round=step + 1, offset=start - prompt + 1,
                proposal_ids=block[0, 1:].tolist(), tree=None if tree is None else dict(
                    token_ids=tree.tokens[0].tolist(), parents=list(tree.parents), depths=list(tree.depths))))
        with mark("target_verify"):
            output = target(verify_block, position_ids=verify_positions,
                            past_key_values=target_cache, use_cache=True, output_hidden_states=True, **verify_kwargs)
        with mark("accept_decision"):
            posterior = output.logits.argmax(dim=-1)
            decision = None
            if tree is not None:
                if verifier is None:
                    path, recovery, native_accepted = tree.select(posterior)
                else:
                    path, recovery, decision = verifier.decide_tree(tree, output.logits[0],
                        horizon=block.shape[1]-1, stop_token_ids=stop_token_ids)
                    native_accepted = None
                if isinstance(tree_config, BudgetTreeConfig):
                    native_accepted = None
                accepted = len(path)-1
            elif allocation == "target":
                accepted = 0
            elif verifier is None:
                accepted = int((block[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum().item())
            else:
                decision = verifier.decide(block[0, 1:], output.logits[0, :-1], stop_token_ids=stop_token_ids)
                accepted = decision.accepted
        with mark("commit_kv"):
            advance = accepted + 1
            if tree is None:
                ids[:, start:start + advance] = block[:, :advance]
                ids[:, start + advance] = posterior[:, accepted]
            else:
                selected = block.new_tensor(path)
                ids[:, start:start + advance] = tree.tokens.index_select(1, selected)
                ids[:, start + advance] = recovery
                valid_features = _features(output, draft.target_layer_ids).index_select(1, selected)
                commit_tree_cache(target_cache, start, path)
            old_start = start
            start += advance
            target_cache.crop(start)
            if tree is None:
                valid_features = _features(output, draft.target_layer_ids)[:, :advance, :].clone()
            pending.append(valid_features)
            pending_tokens += advance
            result.feature_peak_tokens = max(result.feature_peak_tokens, pending_tokens)
        if proposal is not None:
            matcher.observe(proposal, accepted, [*confirmed, *proposal.tokens[:accepted]])
            if getattr(proposal, "draft_prefix", 0) <= accepted < len(proposal.tokens):
                blocked_until = step + config.cooldown_rounds + 1
        if trace:
            result.rounds.append(dict(route=route, generated_before=old_start - prompt + 1,
                                      proposed=block.shape[1] - 1, accepted=accepted, advance=advance,
                                      draft_cache_before=before_draft, pending_before=before_pending,
                                      pending_after=pending_tokens, target_cache_after=start,
                                      source=proposal.source if proposal else None,
                                      lookup=proposal.lookup if proposal else None,
                                      matched=proposal.matched if proposal else 0, buffer_flush=bool(flush),
                                      draft_forward=draft_forward))
            if decision is not None:
                result.rounds[-1].update(strict_accepted=decision.strict_accepted,
                                         relaxed_tokens=decision.relaxed, regret=decision.regret,
                                         draft_prefix=getattr(proposal, "draft_prefix", None))
            if allocation is not None:
                result.rounds[-1]["allocation"] = allocation
            if tree is not None:
                result.rounds[-1].update(tree_nodes=len(tree.parents), native_accepted=native_accepted,
                                         tree_path=path, tree_gain=accepted-native_accepted if native_accepted is not None else None)
                if hasattr(tree, "budget_metadata"):
                    result.rounds[-1]["tree_budget"] = tree.budget_metadata
        step += 1
        available = min(start + 1, limit)
        if on_event is not None:
            # Observe the actual published suffix, including EOS/output limits.
            # The anchor was already published and is never counted twice.
            committed = ids[0, old_start + 1:available].tolist()
            ended = next((i for i, token in enumerate(committed) if token in stops), None)
            if ended is not None:
                committed = committed[:ended + 1]
            visible = min(accepted, len(committed))
            selected_path = path[:visible + 1] if tree is not None else None
            predictions = posterior[0].tolist()
            if tree is None:
                relaxed = [i for i, token in enumerate(committed[:visible]) if token != predictions[i]]
            else:
                relaxed = [node for parent, node in zip(selected_path, selected_path[1:])
                           if int(tree.tokens[0, node]) != predictions[parent]]
            on_event(dict(phase="verified", round=step, accepted=visible,
                committed_ids=committed, correction_ids=committed[visible:],
                selected_path=selected_path, relaxed=relaxed, stopped=ended is not None,
                strict_accepted=decision.strict_accepted if decision is not None else accepted,
                asd=None if verifier is None else dict(relaxed=len(relaxed), regret=decision.regret,
                    spent=verifier.spent, budget=verifier.config.budget)))
        with mark("publish_stop"):
            publish(available)
            if stops:
                # The confirmed prefix was already checked. Scan only newly committed
                # proposals and the recovery/bonus; the old anchor is checked earlier.
                checked_from = old_start + 1 if incremental_stop else prompt
                generated = ids[0, checked_from:available].tolist()
                ended = next((i for i, token in enumerate(generated) if token in stops), None)
                if ended is not None:
                    result.sequences = ids[:, :checked_from + ended + 1].clone()
                    return result
    result.sequences = ids[:, :min(start + 1, limit)].clone()
    return result
