"""Frozen four-arm, six-task comparison of cost-controlled ASD trees."""
from collections import Counter, defaultdict
import math
import statistics

from memorydraft.evaluation.asd_comparison import bootstrap, quality_ok
from memorydraft.evaluation.asd_suite import BENCHMARKS, load_suite

ARMS = ("DFlash", "ASD", "BudgetTree", "BudgetTree-ASD")
REPEATS = {"natural": 3, "prefix": 2}
STUDY = "asd-budget-tree"


def records(root):
    return load_suite(root/"data/research/asd-suite", "development")


def aggregate(rows, references):
    refs = {r["request_id"]: r for r in references}
    if len(refs) != len(references):
        raise ValueError("Duplicate AR reference")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["request_id"]][row["stage"], row["arm"]].append(row)
    if set(grouped) != set(refs) or not grouped:
        raise ValueError("Missing request coverage")
    samples = []
    for identity, cells in sorted(grouped.items()):
        if set(cells) != {(s,a) for s in REPEATS for a in ARMS}:
            raise ValueError("Incomplete tree comparison matrix")
        ref = refs[identity]
        values = {}
        for (stage, arm), entries in cells.items():
            entries.sort(key=lambda r:r["repeat"])
            if [r["repeat"] for r in entries] != list(range(REPEATS[stage])):
                raise ValueError("Repeated or missing measurement")
            first = entries[0]
            signature = lambda r:(r["token_ids"], r["rounds"], r["asd"], r["finish_reason"],
                                  r["prompt_sha256"], r["snapshot_id"])
            if any(signature(r) != signature(first) for r in entries):
                raise ValueError("Unstable repeated trajectory")
            if any((r["prompt_sha256"], r["snapshot_id"]) != (ref["prompt_sha256"], ref["snapshot_id"])
                   or r["benchmark"] != ref["benchmark"] for r in entries):
                raise ValueError("Target input mismatch")
            limit = 4096 if stage == "natural" else 128
            if arm in ("DFlash", "BudgetTree") and (first["token_ids"] != ref["token_ids"][:limit]
                    or (stage == "natural" and (first["text"], first["finish_reason"]) != (ref["text"], ref["finish_reason"]))):
                raise ValueError("Strict tree differs from AR")
            if any(r["asd"] and not 0 <= r["asd"]["spent"] <= 8 for r in entries):
                raise ValueError("Exceeded original ASD budget")
            if any(not math.isfinite(r["latency_seconds"]) or r["latency_seconds"] <= 0 for r in entries):
                raise ValueError("Invalid latency")
            values[f"{stage}/{arm}"] = dict(seconds=statistics.mean(r["latency_seconds"] for r in entries),
                tokens=len(first["token_ids"]), calls=len(first["rounds"]), quality=first.get("quality"),
                finish_reason=first["finish_reason"], spent=first["asd"]["spent"] if first["asd"] else 0,
                relaxed=first["asd"]["relaxed_tokens"] if first["asd"] else 0,
                nodes=sum(r.get("tree_nodes",r["proposed"]+1) for r in first["rounds"]),
                node_histogram=dict(Counter(r["tree_nodes"] for r in first["rounds"] if "tree_nodes" in r)),
                same_tree_extra=sum(r["accepted"]-r["strict_accepted"] for r in first["rounds"]
                                    if arm == "BudgetTree-ASD"))
        samples.append(dict(request_id=identity, benchmark=ref["benchmark"], arms=values))
    families = {}
    for benchmark in BENCHMARKS:
        selected = [s for s in samples if s["benchmark"] == benchmark]
        if not selected:
            continue
        families[benchmark] = {}
        for arm in ARMS:
            result = {}
            for stage in REPEATS:
                items = [s["arms"][f"{stage}/{arm}"] for s in selected]
                result[stage] = dict(requests=len(items), tokens=sum(v["tokens"] for v in items),
                    calls=sum(v["calls"] for v in items), nodes=sum(v["nodes"] for v in items),
                    spent=sum(v["spent"] for v in items), relaxed=sum(v["relaxed"] for v in items),
                    same_tree_extra=sum(v["same_tree_extra"] for v in items),
                    node_histogram=dict(sum((Counter(v["node_histogram"]) for v in items),Counter())))
                for baseline in ("DFlash", "ASD", "BudgetTree"):
                    pairs = [(s["arms"][f"{stage}/{arm}"],s["arms"][f"{stage}/{baseline}"]) for s in selected]
                    common = [(a,b) for a,b in pairs if stage == "natural" or a["tokens"] == b["tokens"] == 128]
                    ratios = [a["seconds"]/b["seconds"] for a,b in common]
                    result[stage][baseline] = dict(n=len(common),ratio=statistics.mean(ratios) if ratios else None,
                        ci95=bootstrap(ratios) if ratios else None,
                        length_ratio=statistics.mean(a["tokens"]/b["tokens"] for a,b in common) if common else None)
                if stage == "natural":
                    result[stage].update(correct=sum(quality_ok(v) for v in items),
                        primary_correct=sum(v["quality"]["correct"] for v in items),
                        truncated=sum(v["finish_reason"] == "length" for v in items))
                    for baseline in ("ASD", "BudgetTree"):
                        paired = [(s["arms"][f"natural/{arm}"],s["arms"][f"natural/{baseline}"]) for s in selected]
                        result[stage][baseline].update(lost=[s["request_id"] for s,(a,b) in zip(selected,paired)
                            if quality_ok(b) and not quality_ok(a)],
                            gained=[s["request_id"] for s,(a,b) in zip(selected,paired) if quality_ok(a) and not quality_ok(b)])
            families[benchmark][arm] = result
    return dict(samples=samples,families=families,timed_generations=len(rows),ar_generations=len(references))
