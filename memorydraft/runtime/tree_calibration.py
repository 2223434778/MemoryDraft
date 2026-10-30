"""Offline, single-owner synthetic latency calibration; no benchmark answers."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch
from transformers import DynamicCache

from memorydraft.backends.budget_tree import BudgetTreeConfig, TreeCosts, build_budget_tree
from memorydraft.backends.dflash_reuse import generate_reuse
from memorydraft.backends.draft_tree import DraftTree
from memorydraft.runtime.cost_tree import cost_environment


@torch.inference_mode()
def calibrate_tree(engine, output, *, contexts=(128, 1024), nodes=(8, 12, 16, 20, 24, 32), repeats=4):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve existing calibration: " + str(output))
    if engine.config.backend != "dflash" or engine.draft is None or not engine.config.target_revision:
        raise ValueError("Calibration requires DFlash and a pinned Target revision")
    if (type(repeats) is not int or repeats < 2 or len(contexts) < 2 or len(nodes) < 2
            or tuple(sorted(set(contexts))) != tuple(contexts) or tuple(sorted(set(nodes))) != tuple(nodes)
            or any(type(n) is not int or n < 1 for n in contexts)
            or any(type(n) is not int or not 2 <= n <= 64 for n in nodes)):
        raise ValueError("Invalid calibration grid or repeats")
    if max(contexts) + max(nodes) > engine.target.config.max_position_embeddings:
        raise ValueError("Synthetic calibration exceeds Target context")
    if not engine.lock.acquire(blocking=False):
        raise RuntimeError("Calibration requires exclusive engine ownership")
    started = time.perf_counter()
    try:
        if engine._closed:
            raise RuntimeError("InferenceEngine is closed")
        captured, draft_times, replays = [], [], []
        original = engine.draft.forward
        def measured(*args, **kwargs):
            engine.synchronize();begin = time.perf_counter()
            result = original(*args, **kwargs)
            engine.synchronize();draft_times.append(time.perf_counter()-begin)
            return result
        def capture(confirmed, tokens, logits, budget):
            if not captured:
                captured.append((int(confirmed[-1]), list(tokens), logits.detach().clone()))
        cache_factory = engine._tree_policy.cache_factory() if engine._tree_policy is not None else None
        messages = [{"role": "user", "content": "List the integers from one to twenty."}]
        ids = engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
            return_tensors="pt", **engine.config.template_options).to(engine.target.device)
        with patch.object(engine.draft, "forward", measured):
            # Ignore EOS only for this explicitly synthetic calibration workload.
            generate_reuse(engine.target, engine.draft, ids, max_new_tokens=64, on_draft=capture,
                           cache_factory=cache_factory)
        if not captured or not draft_times:
            raise ValueError("Calibration did not execute Draft")
        text_ids = engine.tokenizer.encode("Calculate carefully using the supplied facts. " * max(contexts),
                                           add_special_tokens=False)
        seconds = []
        for context in contexts:
            ids = torch.tensor([text_ids[:context]], device=engine.target.device)
            if ids.shape[1] != context:
                raise ValueError("Synthetic prefix too short")
            cache = cache_factory() if cache_factory is not None else DynamicCache()
            initial = engine.target(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            anchor = int(initial.logits[0, -1].argmax())
            times = []
            for count in nodes:
                parents = (-1,) + tuple((i-1)//2 for i in range(1, count))
                depths = [0]
                for parent in parents[1:]:
                    depths.append(depths[parent]+1)
                tree = DraftTree(ids.new_full((1, count), anchor), parents, tuple(depths), 1)
                positions = ids.new_tensor([depths])+context
                observed = []
                for repeat in range(repeats):
                    cache.crop(context);engine.synchronize();begin = time.perf_counter()
                    result = engine.target(tree.tokens, position_ids=positions, past_key_values=cache,
                        attention_mask=tree.mask(context, engine.target.dtype), use_cache=True, output_hidden_states=True)
                    result.logits.argmax(-1).tolist();engine.synchronize()
                    elapsed = time.perf_counter()-begin
                    replays.append(dict(context=context, nodes=count, repeat=repeat, seconds=elapsed, warmup=repeat==0))
                    if repeat:
                        observed.append(elapsed)
                    del result
                times.append(statistics.median(observed))
            seconds.append(tuple(times))
            del initial, cache
        costs = TreeCosts(tuple(contexts), tuple(nodes), tuple(seconds), statistics.median(draft_times))
        anchor, tokens, logits = captured[0]
        block = torch.tensor([[anchor, *tokens]], device=logits.device)
        auxiliary = []
        config = BudgetTreeConfig(costs, nodes[0], nodes[-1], min(4, logits.shape[-1]))
        builder, build_kwargs = build_budget_tree, {}
        if engine._tree_policy is not None:
            from memorydraft.backends.branch_drift import build_paired_tree
            config = engine._tree_policy.bind(config)
            builder = build_paired_tree
            build_kwargs['regret_budget'] = engine._tree_policy.verifier(engine._asd_config).construction_budget()
        for _ in range(repeats):
            engine.synchronize();begin = time.perf_counter()
            builder(block, logits.unsqueeze(0), config, contexts[0], stop_token_ids=engine.stops, **build_kwargs)
            engine.synchronize();auxiliary.append(time.perf_counter()-begin)
        costs = replace(costs, auxiliary_seconds=statistics.median(auxiliary[1:]))
        data = dict(schema=1, environment=cost_environment(engine), costs=asdict(costs), measurement=dict(
            method="synthetic-prefix tree replay; first sample excluded per shape; median remaining",
            replays=replays, draft_seconds=draft_times, auxiliary_seconds=auxiliary,
            calibration_seconds=time.perf_counter()-started))
        output.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(data, indent=2)+"\n").encode()
        with output.open("xb") as stream:
            stream.write(raw)
        return dict(path=str(output.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                    calibration_seconds=data["measurement"]["calibration_seconds"], environment=data["environment"])
    finally:
        try:
            engine.synchronize()
        finally:
            engine.lock.release()
