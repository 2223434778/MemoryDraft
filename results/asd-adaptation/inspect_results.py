"""Posthoc coverage, distribution shift and cost interpretation; no model selection."""
from collections import Counter
import json
from pathlib import Path

from memorydraft.evaluation.runtime_migration import read_rows

root = Path.cwd()
base = root/"results/asd-adaptation"
examples = [r for i in (0, 1) for r in read_rows(base/f"collect-{i}/examples.jsonl")]
sources = [r for i in (0, 1) for r in read_rows(base/f"collect-{i}/sources.jsonl")]
pairs = {}
for r in examples:
    pairs.setdefault((r["request_id"], r["source_round"]), {})[r["source_policy"]] = r
shift = {}
for split in ("train", "calibration"):
    chosen = [p for p in pairs.values() if p["ordinary"]["split"] == split]
    shift[split] = dict(paired_states=len(chosen), requests=len({p["ordinary"]["request_id"] for p in chosen}),
        different_prefix=sum(p["ordinary"]["prefix_ids"] != p["asd-prefix"]["prefix_ids"] for p in chosen),
        same_generated_boundary=sum(p["ordinary"]["generated_before"] == p["asd-prefix"]["generated_before"] for p in chosen),
        prior_asd_relaxed=sum(p["asd-prefix"]["past_relaxed"] > 0 for p in chosen),
        different_teacher_labels=sum(p["ordinary"]["teacher_tokens"] != p["asd-prefix"]["teacher_tokens"] for p in chosen))
curves = {a:list(read_rows(base/f"train-{a}/metrics.jsonl")) for a in ("ordinary", "asd-prefix")}
trainers = {a:json.loads((base/f"train-{a}/provenance.json").read_text(encoding="utf-8")) for a in curves}
collectors = [json.loads((base/f"collect-{i}/provenance.json").read_text(encoding="utf-8")) for i in (0, 1)]
analysis = json.loads((base/"analysis.json").read_text(encoding="utf-8"))
result = dict(posthoc=True, source_generations=len(sources), source_finishes=dict(Counter(r["finish_reason"] for r in sources)),
    examples=len(examples), unique_teacher_prefixes=len({r["prefix_sha256"] for r in examples}), shift=shift,
    feature_cache_bytes=sum(c["cache_bytes"] for c in collectors), curves=curves,
    training={a:dict(seconds=p["seconds"], selected_step=p["selected_step"], initial_loss=p["initial_calibration"]["loss"],
                    selected_loss=p["selected_calibration_loss"], final_loss=curves[a][-1]["calibration"]["loss"],
                    selected_loss_ratio=p["selected_calibration_loss"]/p["initial_calibration"]["loss"],
                    first_gradient_norm=p["first_grad_norm"], peak_gpu_bytes=p["peak_gpu_bytes"]) for a, p in trainers.items()},
    domain=analysis["scopes"]["gsm8k"],
    notes=["Prefix differences include different source round progress, not a pure token intervention.",
           "Teacher suffixes are new continuations under each actual prefix; original ASD output is not the label.",
           "An equal update count is not an equal measured FLOP or wall-time claim.",
           "Natural approximate outputs may change length; use full-prefix work and quality separately."])
(base/"interpretation.json").write_bytes((json.dumps(result, indent=2)+"\n").encode())
print(json.dumps({k:v for k,v in result.items() if k not in {"domain", "curves"}}))
