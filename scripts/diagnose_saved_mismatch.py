"""Inspect the actual Target logits at one saved greedy mismatch; no relaxed acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from memorydraft.backends.transformers import TransformersBackend
from memorydraft.evaluation.replay import read_jsonl
from memorydraft.memory.store import MemoryStore


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", default="results/qwen3-pilot/smoke-bf16")
parser.add_argument("--output", default="results/qwen3-pilot/bf16-diagnostic.json")
args = parser.parse_args()
run = Path(args.run)
provenance = json.loads((run / "provenance.json").read_text())
failed = next(r for path in run.glob("replay*.jsonl") for r in read_jsonl(path) if not r["correct"])
reference = failed["reference_token_ids"]
position = failed["first_difference"]
request = next(r for path in provenance["input_sha256"] for r in read_jsonl(path) if r["id"] == failed["request_id"])
torch.set_num_threads(4)
torch.set_float32_matmul_precision("highest")
backend = TransformersBackend.load(provenance["config"])
prompt = backend.prompt(request["messages"])[0].tolist()
full_prefix = prompt + reference[:position]
target_position = len(prompt) - 1 + position
captures, pending, label = {}, {}, ""


def before(model, inputs, kwargs):
    ids = kwargs.get("input_ids", inputs[0] if inputs else None)
    cache = kwargs.get("past_key_values")
    start = cache.get_seq_length() if cache is not None else 0
    stop = min(ids.shape[1], target_position - start + 1)
    pending.update(start=start, query=ids.shape[1],
                   prefix_matches=stop > 0 and ids[0, :stop].tolist() == full_prefix[start:start + stop])


def after(model, inputs, kwargs, output):
    first = pending["start"] + pending["query"] - output.logits.shape[1]
    index = target_position - first
    if 0 <= index < output.logits.shape[1] and pending["prefix_matches"]:
        logits = output.logits[0, index].detach().float().clone()
        captures.setdefault(label, []).append((dict(pending), logits))


pre = backend.target.register_forward_pre_hook(before, with_kwargs=True)
post = backend.target.register_forward_hook(after, with_kwargs=True)
results = {}
try:
    with MemoryStore(run / "history.db") as store:
        snapshot = store.snapshot(request["user_id"], request["order"])
        for label in ("AR", failed["method"]):
            result = backend.generate(request["messages"], snapshot, method=label,
                                      max_new_tokens=provenance["args"]["max_new_tokens"],
                                      block_size=provenance["args"]["blocks"][0])
            results[label] = result.token_ids
finally:
    pre.remove()
    post.remove()
evidence = dict(request_id=request["id"], position=position, dtype=str(backend.target.dtype),
                attention=provenance["config"]["attention"], results=results,
                allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                note="Actual final Target forward covering the mismatch, with matching confirmed input prefix. Diagnostic timings are excluded from performance.")
for name, values in captures.items():
    state, logits = values[-1]
    top = logits.topk(5)
    evidence[name] = dict(state=state, top5=[(int(i), float(v)) for i, v in zip(top.indices, top.values)],
                          top2_gap=float(top.values[0] - top.values[1]), matching_forwards=len(values))
if len(captures) == 2:
    a, b = [values[-1][1] for values in captures.values()]
    evidence["max_abs_logit_difference"] = float((a - b).abs().max())
evidence["backend_source_sha256"] = hashlib.sha256(Path("memorydraft/backends/transformers.py").read_bytes()).hexdigest()
Path(args.output).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in evidence.items() if k != "results"}, ensure_ascii=False, indent=2))
