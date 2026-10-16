from __future__ import annotations

import copy
import time
from dataclasses import asdict

import torch
from transformers.generation.candidate_generator import AssistedCandidateGenerator, CandidateGenerator, PromptLookupCandidateGenerator
from transformers.generation.logits_process import LogitsProcessorList

from memorydraft.retrieval.pool import ContinuationIndex, MemoryBundle
from memorydraft.routing.policy import Router


class RecordedCandidates(CandidateGenerator):
    """Observe upstream proposals; optionally enforce the final output boundary."""

    def __init__(self, delegate, route, max_length=None):
        self.delegate, self.route = delegate, route
        self.max_length = max_length
        self.events = []
        self.index_seconds = 0.0

    def get_candidates(self, input_ids):
        self.start = time.perf_counter()
        self.old_length = input_ids.shape[1]
        result = self.delegate.get_candidates(input_ids)
        if self.max_length is not None:
            # HF 4.52.2 prompt lookup can propose past the final output budget.
            # Reserve one Target correction/bonus token, as the model drafter does.
            candidate_ids, logits = result
            end = max(self.old_length, self.max_length - 1)
            candidate_ids = candidate_ids[:, :end]
            if logits is not None:
                logits = logits[:, :candidate_ids.shape[1] - self.old_length]
            result = candidate_ids, logits
        self.proposed = result[0].shape[1] - self.old_length
        return result

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        self.delegate.update_candidate_strategy(input_ids, scores, num_matches)
        self.events.append(dict(route=self.route if self.proposed else "AR", proposed=self.proposed,
                                accepted=int(num_matches), committed=input_ids.shape[1] - self.old_length,
                                round_seconds=time.perf_counter() - self.start))

    def summary(self):
        result = {}
        for e in self.events:
            stats = result.setdefault(e["route"], dict(rounds=0, proposed=0, accepted=0, committed=0))
            stats["rounds"] += 1
            for key in ("proposed", "accepted", "committed"):
                stats[key] += e[key]
        return result


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
        self.generator.assistant_kwargs["use_model_defaults"] = False
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
                 bundle: MemoryBundle, method: str, block_size: int, min_match: int = 3,
                 retrieval_index=None, dynamic_lookup=False, special_ids=()):
        self.target_length = target_ids.shape[1]
        self.max_length = generation_config.max_length
        self.draft_model = draft_model
        self.prefixes = prefixes
        self.config = generation_config
        self.block_size = block_size
        self.router = Router(method, bool(bundle.chunks))
        index_start = time.perf_counter()
        self.index = retrieval_index if retrieval_index is not None else ContinuationIndex(bundle, min_match=min_match)
        self.index_seconds = time.perf_counter() - index_start
        self.dynamic_lookup, self.special_ids = dynamic_lookup, set(special_ids)
        self.lookup = PromptLookupCandidateGenerator(
            eos_token_id=torch.tensor(list(special_ids), dtype=torch.long),
            num_output_tokens=block_size, max_matching_ngram_size=3,
            max_length=self.max_length,
        ) if dynamic_lookup else None
        self.providers: dict[str, IndependentDraft] = {}
        self.events = []
        self.pending = None

    def get_candidates(self, input_ids):
        start = time.perf_counter()
        confirmed = input_ids[:, self.target_length:]
        # Leave room for Target's correction/bonus token, as upstream does.
        budget = min(self.block_size, self.max_length - input_ids.shape[1] - 1)
        r_enabled = self.router.method in {"R", "RD", "RHD"}
        prefix = input_ids[0].tolist() if self.dynamic_lookup else confirmed[0].tolist()
        copied, source, matched = self.index.match(prefix, budget) if r_enabled else ([], None, 0)
        if r_enabled and not copied and self.lookup is not None and budget > 0:
            cpu_ids = input_ids.cpu()
            proposed, _ = self.lookup.get_candidates(cpu_ids)
            copied = proposed[0, input_ids.shape[1]:input_ids.shape[1] + budget].tolist()
            if copied:
                source = "__confirmed_context__"
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
