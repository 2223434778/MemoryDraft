"""Compare real AR/assisted outputs without relaxing token acceptance."""
import argparse
import copy
import json
from pathlib import Path

import torch

from memorydraft.backends.transformers import TransformersBackend
from memorydraft.evaluation.replay import read_jsonl
from memorydraft.memory.store import MemoryStore


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--request-id", default="docs-3")
parser.add_argument("--output", required=True)
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
request = next(r for r in read_jsonl("data/synthetic/requests.jsonl") if r["id"] == args.request_id)
results = []
for dtype, attention in [("bfloat16", "sdpa"), ("float32", "eager")]:
    cfg = copy.deepcopy(config)
    cfg.update(dtype=dtype, attention=attention)
    backend = TransformersBackend.load(cfg)
    with MemoryStore("pilot.db") as store:
        snapshot = store.snapshot(request["user_id"], request["order"])
        ar = backend.generate(request["messages"], snapshot, method="AR", max_new_tokens=64)
        d = backend.generate(request["messages"], snapshot, method="D", max_new_tokens=64)
    first_diff = next((i for i, (x, y) in enumerate(zip(ar.token_ids, d.token_ids)) if x != y), None)
    row = {"dtype": dtype, "attention": attention, "equal": ar.token_ids == d.token_ids,
           "first_diff": first_diff, "ar": ar.token_ids, "d": d.token_ids}
    if first_diff is not None:
        prefix = backend.prompt(request["messages"]).to(backend.target.device)
        prefix = torch.cat((prefix, torch.tensor([ar.token_ids[:first_diff]], device=prefix.device)), 1)
        with torch.inference_mode():
            past = backend.target(prefix[:, :-1], use_cache=True).past_key_values
            one = backend.target(prefix[:, -1:], past_key_values=copy.deepcopy(past)).logits[0, 0].float()
            more = torch.tensor([ar.token_ids[first_diff:first_diff + 8]], device=prefix.device)
            block = backend.target(torch.cat((prefix[:, -1:], more), 1), past_key_values=copy.deepcopy(past)).logits[0, 0].float()
            row["single_top2"] = [(int(i), float(v)) for v, i in zip(*one.topk(2))]
            row["block_top2"] = [(int(i), float(v)) for v, i in zip(*block.topk(2))]
            row["max_abs_logit_difference"] = float((one - block).abs().max())
    results.append(row)
    del backend
    torch.cuda.empty_cache()
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps([{k: v for k, v in row.items() if k not in {"ar", "d"}} for row in results], indent=2))
