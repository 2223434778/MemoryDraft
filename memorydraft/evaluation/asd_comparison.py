"""Frozen cross-route comparison and request-level selection, without model calls."""
from collections import defaultdict
import math
from pathlib import Path
import random
import statistics

from memorydraft.evaluation.asd_suite import BENCHMARKS, load_suite

STUDY = "asd-comparison"
VARIANTS = ("base", "asd-prefix", "ordinary-position")
POLICIES = ("DFlash", "ASD", "ASD-Run", "ASD-Utility")
ARMS = tuple(f"{v}/{p}" for v in VARIANTS for p in POLICIES) + (
    "base/Confidence-Utility", "base/tree32")
BASELINES = ("base/DFlash", "base/ASD")
REPEATS = {"screen": {"natural": 2, "prefix": 1}, "confirm": {"natural": 3, "prefix": 2}}
ADAPTERS = {
    "asd-prefix": ("asd-adaptation-v2/artifacts/adaptation-adapters/asd-prefix/selected.safetensors",
                   "885375d37670df6ec60e756bb018b9c5efc0f3121be23ee9c5424e14690822e0"),
    "ordinary-position": ("asd-adaptation-followup/artifacts/followup-adapter/selected.safetensors",
                          "42755672a816b44b727cbc5bcdbbd2b18dd2a19db42e5706f09d540d678b1f6e"),
}


def records(root: Path, phase: str):
    if phase not in REPEATS:
        raise ValueError("Unknown comparison phase")
    pool = load_suite(root/"data/research/asd-suite", "reserve")
    return [r for b in BENCHMARKS for i, r in enumerate(r for r in pool if r["benchmark"] == b)
            if (i < 4) == (phase == "screen")]


def arms_for(benchmark, phase, selection=None):
    if phase == "screen":
        return ARMS
    if phase != "confirm" or selection is None or set(selection) != set(BENCHMARKS):
        raise ValueError("Confirmation requires a complete frozen selection")
    chosen = selection[benchmark]
    if len(chosen) != 2 or len(set(chosen)) != 2 or any(a not in ARMS or a in BASELINES for a in chosen):
        raise ValueError("Invalid confirmation candidates")
    return (*BASELINES, *chosen)


def quality_ok(item):
    quality = item["quality"]
    return quality.get("secondary_correct", quality["correct"])


def aggregate(rows, phase, selection=None):
    """Reject incomplete/repeated cells before computing any winner."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["phase"] != phase:
            raise ValueError("Mixed comparison phases")
        grouped[row["request_id"]][row["stage"], row["arm"]].append(row)
    if not grouped:
        raise ValueError("Empty comparison")
    samples = []
    for identity, cells in sorted(grouped.items()):
        benchmark_set = {r["benchmark"] for values in cells.values() for r in values}
        if len(benchmark_set) != 1:
            raise ValueError("Inconsistent request benchmark")
        benchmark = next(iter(benchmark_set))
        expected = arms_for(benchmark, phase, selection)
        if set(cells) != {(s, a) for s in REPEATS[phase] for a in expected}:
            raise ValueError("Incomplete candidate matrix")
        values = {}
        for (stage, arm), entries in cells.items():
            entries.sort(key=lambda r: r["repeat"])
            if [r["repeat"] for r in entries] != list(range(REPEATS[phase][stage])):
                raise ValueError("Invalid repeat coverage")
            first = entries[0]
            signature = lambda r: (r["token_ids"], r["rounds"], r["asd"], r["finish_reason"], r["prompt_sha256"], r["snapshot_id"])
            if any(signature(r) != signature(first) for r in entries):
                raise ValueError("Unstable repeated trajectory")
            if any(not r["same_input"] or (arm.endswith(("/DFlash", "/tree32")) and not r["strict_token_match"])
                   for r in entries):
                raise ValueError("Input or strict output mismatch")
            if any(r["asd"] and not 0 <= r["asd"]["spent"] <= 8 for r in entries):
                raise ValueError("ASD budget exceeded")
            times = [r["latency_seconds"] for r in entries]
            if any(not math.isfinite(t) or t <= 0 for t in times):
                raise ValueError("Invalid timing")
            values[stage, arm] = dict(seconds=statistics.mean(times), times=times, tokens=len(first["token_ids"]),
                calls=len(first["rounds"]), finish_reason=first["finish_reason"], quality=first.get("quality"),
                spent=first["asd"]["spent"] if first["asd"] else 0,
                relaxed=first["asd"]["relaxed_tokens"] if first["asd"] else 0)
        inputs = {(r["prompt_sha256"], r["snapshot_id"]) for entries in cells.values() for r in entries}
        if len(inputs) != 1:
            raise ValueError("Candidates used different inputs")
        samples.append(dict(request_id=identity, benchmark=benchmark, arms={f"{s}/{a}": v for (s,a),v in values.items()}))
    families = {}
    for benchmark in sorted({s["benchmark"] for s in samples}):
        selected = [s for s in samples if s["benchmark"] == benchmark]
        families[benchmark] = {}
        for arm in arms_for(benchmark, phase, selection):
            result = {}
            for stage in REPEATS[phase]:
                items = [s["arms"][f"{stage}/{arm}"] for s in selected]
                result[stage] = dict(requests=len(items), tokens=sum(v["tokens"] for v in items),
                    calls=sum(v["calls"] for v in items), spent=sum(v["spent"] for v in items),
                    relaxed=sum(v["relaxed"] for v in items))
                for baseline in BASELINES:
                    pairs = [(s["arms"][f"{stage}/{arm}"], s["arms"][f"{stage}/{baseline}"]) for s in selected]
                    # Pairwise equal-token sets retain short-EOS requests in the natural table.
                    common = [(v,b) for v,b in pairs if stage == "natural" or v["tokens"] == b["tokens"] == 128]
                    ratios = [v["seconds"]/b["seconds"] for v,b in common]
                    densities = [(v["calls"]/v["tokens"])/(b["calls"]/b["tokens"]) for v,b in common if b["calls"]]
                    result[stage][baseline] = dict(n=len(common), ratio=statistics.mean(ratios) if ratios else None,
                        ci95=bootstrap(ratios) if ratios else None,
                        length_ratio=statistics.mean(v["tokens"]/b["tokens"] for v,b in common) if common else None,
                        call_density_ratio=statistics.mean(densities) if densities else None,
                        call_density_requests=len(densities),
                        lost=sum(quality_ok(b) and not quality_ok(v) for v,b in pairs) if stage == "natural" else None,
                        gained=sum(not quality_ok(b) and quality_ok(v) for v,b in pairs) if stage == "natural" else None,
                        new_truncations=sum(b["finish_reason"] == "eos" and v["finish_reason"] != "eos" for v,b in pairs))
                if stage == "natural":
                    result[stage].update(correct=sum(v["quality"]["correct"] for v in items),
                        secondary_correct=sum(quality_ok(v) for v in items),
                        truncations=sum(v["finish_reason"] != "eos" for v in items))
            families[benchmark][arm] = result
    return dict(phase=phase, families=families, samples=samples, requests=len(samples), rows=len(rows))


def bootstrap(values):
    rng = random.Random("MemoryDraft-comparison-bootstrap-v1")
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return [means[49], means[1949]]


def select_candidates(summary):
    """One policy/search candidate and one adapted candidate per task family."""
    if summary["phase"] != "screen" or set(summary["families"]) != set(BENCHMARKS):
        raise ValueError("Selection requires all six screening families")
    result = {}
    for benchmark, arms in summary["families"].items():
        def key(arm):
            item = arms[arm]["natural"]["base/ASD"]
            return (item["lost"], item["new_truncations"], item["ratio"], arm)
        policy = [a for a in ARMS if a.startswith("base/") and a not in BASELINES]
        adapted = [a for a in ARMS if not a.startswith("base/")]
        result[benchmark] = [min(policy, key=key), min(adapted, key=key)]
    return result
