from __future__ import annotations

import copy
import time
from dataclasses import asdict

import torch
from transformers.generation.candidate_generator import AssistedCandidateGenerator, CandidateGenerator
from transformers.generation.logits_process import LogitsProcessorList

from memorydraft.retrieval.pool import ContinuationIndex, MemoryBundle
from memorydraft.routing.policy import Router


class IndependentDraft:
    """Shares S's weights, owns one upstream generator/KV per context.

    Only confirmed token IDs cross the context boundary. Cropping immediately
    after verification is essential: an inactive route must never retain rejected
    tokens when another route has advanced the output beyond its old cache end.
    """

    def __init__(self, model, prefix: torch.Tensor, target_prompt_length: int,
                 generation_config, block_size: int):
        config = copy.deepcopy(generation_config)
        config.max_length += prefix.shape[1] - target_prompt_length
        self.prefix = prefix.to(model.device)
        self.generator = AssistedCandidateGenerator(
            input_ids=self.prefix, assistant_model=model, generation_config=config,
            model_kwargs={"attention_mask": torch.ones_like(self.prefix), "use_cache": True},
            logits_processor=LogitsProcessorList(),
        )
        self.generator.num_assistant_tokens = block_size
        self.generator.assistant_confidence_threshold = 0.0
        self.generator.generation_config.assistant_confidence_threshold = 0.0
        self.generator.generation_config.num_assistant_tokens_schedule = "constant"
        self.committed_output_len = 0
        self.context_epoch = 0
        self.last_input_tokens = 0

    def propose(self, confirmed: torch.Tensor):
        local = torch.cat((self.prefix, confirmed.to(self.prefix.device)), dim=1)
        # Lazy first activation can happen after other routes already committed Y.
        # Upstream only resizes masks when a cache already exists.
        self.generator.assistant_kwargs["attention_mask"] = torch.ones_like(local)
        cache = self.generator.assistant_kwargs.get("past_key_values")
        covered = min(cache.get_seq_length(), max(0, local.shape[1] - 2)) if cache is not None else 0
        self.last_input_tokens = local.shape[1] - covered
        result, _ = self.generator.get_candidates(local)
        return result[:, local.shape[1]:]

    def reconcile(self, confirmed_length: int):
        self.committed_output_len = confirmed_length
        cache = self.generator.assistant_kwargs.get("past_key_values")
        if cache is not None:
            cache.crop(max(0, self.prefix.shape[1] + confirmed_length - 1))

    def state(self):
        cache = self.generator.assistant_kwargs.get("past_key_values")
        covered = cache.get_seq_length() if cache is not None else 0
        return {
            "context_epoch": self.context_epoch,
            "committed_output_len": self.committed_output_len,
            "kv_covered_output_len": max(0, covered - self.prefix.shape[1]),
            "cached_prefix_tokens": min(covered, self.prefix.shape[1]),
        }


class MemoryCandidates(CandidateGenerator):
    def __init__(self, *, target_ids, draft_model, prefixes, generation_config,
                 bundle: MemoryBundle, method: str, block_size: int, min_match: int = 3):
        self.target_length = target_ids.shape[1]
        self.max_length = generation_config.max_length
        self.draft_model = draft_model
        self.prefixes = prefixes
        self.config = generation_config
        self.block_size = block_size
        self.router = Router(method, bool(bundle.chunks))
        index_start = time.perf_counter()
        self.index = ContinuationIndex(bundle, min_match=min_match)
        self.index_seconds = time.perf_counter() - index_start
        self.providers: dict[str, IndependentDraft] = {}
        self.events = []
        self.pending = None

    def get_candidates(self, input_ids):
        start = time.perf_counter()
        confirmed = input_ids[:, self.target_length:]
        # Leave room for Target's correction/bonus token, as upstream does.
        budget = min(self.block_size, self.max_length - input_ids.shape[1] - 1)
        copied, source, matched = self.index.match(confirmed[0].tolist(), budget)
        match_done = time.perf_counter()
        route = self.router.choose(bool(copied)) if budget > 0 else "AR"
        setup_seconds = 0.0
        before = None
        if route in {"H", "D"}:
            if route not in self.providers:
                setup_start = time.perf_counter()
                self.providers[route] = IndependentDraft(
                    self.draft_model, self.prefixes[route], self.target_length, self.config, self.block_size,
                )
                setup_seconds = time.perf_counter() - setup_start
            provider = self.providers[route]
            before = provider.state()
            tokens = provider.propose(confirmed).to(input_ids.device)
        elif route == "R":
            tokens = torch.tensor([copied], dtype=input_ids.dtype, device=input_ids.device)
        else:
            tokens = input_ids[:, :0]
        self.pending = {
            "route": route, "proposed": tokens.shape[1], "source_id": source if route == "R" else None,
            "match_length": matched, "l2_host_seconds": match_done - start,
            "setup_host_seconds": setup_seconds, "candidate_host_seconds": time.perf_counter() - start,
            "before_state": before, "start": start, "old_length": input_ids.shape[1],
            "draft_prefill_or_catchup_tokens": self.providers[route].last_input_tokens if route in self.providers else 0,
        }
        return torch.cat((input_ids, tokens), dim=1), None

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        event = self.pending
        accepted = int(num_matches)
        route = event["route"]
        committed = input_ids.shape[1] - event.pop("old_length")
        reconcile_start = time.perf_counter()
        # All providers track the committed frontier; only the active cache changed.
        for provider in self.providers.values():
            provider.reconcile(input_ids.shape[1] - self.target_length)
        event["reconcile_host_seconds"] = time.perf_counter() - reconcile_start
        event["round_seconds"] = time.perf_counter() - event.pop("start")
        event.update(accepted=accepted, committed=committed)
        event["after_state"] = self.providers[route].state() if route in self.providers else None
        self.router.update(route, event["proposed"], accepted, committed, event["round_seconds"])
        self.events.append(event)
        self.pending = None

    def summary(self):
        return {route: asdict(stats) for route, stats in self.router.stats.items()}
