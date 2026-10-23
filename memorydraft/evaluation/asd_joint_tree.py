"""Six-arm comparison isolating fixed spine preservation from regret allocation."""
from pathlib import Path
import gzip
import hashlib
import json
import math
import statistics

from memorydraft.backends.budget_tree import BudgetTreeConfig, RegretBudget, TreeCosts
from memorydraft.evaluation.asd_comparison import bootstrap, quality_ok
from memorydraft.evaluation.asd_tree import ARMS as ORIGINAL_ARMS, REPEATS, aggregate as original_aggregate

STUDY = "asd-joint-tree"
ARMS = (*ORIGINAL_ARMS, "SpineTree-ASD", "JointTree-ASD")
LEGACY = "asd-budget-tree"


def frozen_config(root: Path, shard: int):
    directory = root/"results"/LEGACY
    name = f"part-{shard}/calibration.json"
    raw = (directory/name).read_bytes()
    published = json.loads((directory/"publication.json").read_text())["files"][name]
    if hashlib.sha256(raw).hexdigest() != published["sha256"]:
        raise ValueError("Frozen first-stage cost table changed")
    values = json.loads(raw)["costs"]
    costs = TreeCosts(tuple(values["contexts"]),tuple(values["nodes"]),
        tuple(tuple(row) for row in values["seconds"]),values["draft_seconds"],values["auxiliary_seconds"])
    return BudgetTreeConfig(costs),dict(source=LEGACY,name=name,sha256=published["sha256"])


class LegacyCheck:
    def __init__(self, root: Path):
        self.expected, self.checked = {}, 0
        directory = root/"results"/LEGACY
        published = json.loads((directory/"publication.json").read_text())["files"]
        for shard in (0,1):
            name = f"part-{shard}/rows.jsonl.gz"
            raw = (directory/name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != published[name]["sha256"]:
                raise ValueError("Legacy first-stage rows changed")
            for line in gzip.decompress(raw).decode().splitlines():
                row = json.loads(line)
                if row["repeat"] == 0:
                    self.expected[row["request_id"],row["stage"],row["arm"]] = row

    def __call__(self,row):
        if row["arm"] not in ORIGINAL_ARMS:
            return
        expected = self.expected[row["request_id"],row["stage"],row["arm"]]
        keys = ("token_ids","text","rounds","asd","finish_reason","prompt_sha256","snapshot_id")
        if any(row[k] != expected[k] for k in keys):
            raise ValueError("First-stage trajectory failed to reproduce: "+row["request_id"]+"/"+row["arm"])
        self.checked += 1


def aggregate(rows,references):
    allocation = audit_allocation(rows)
    result = original_aggregate(rows,references,arms=ARMS)
    result["allocation_audit"] = allocation
    for benchmark,family in result["families"].items():
        selected = [s for s in result["samples"] if s["benchmark"] == benchmark]
        for arm in ARMS:
            for stage in REPEATS:
                for baseline in ("BudgetTree-ASD","SpineTree-ASD"):
                    pairs = [(s,s["arms"][f"{stage}/{arm}"],s["arms"][f"{stage}/{baseline}"]) for s in selected]
                    common = [(s,a,b) for s,a,b in pairs if stage == "natural" or a["tokens"] == b["tokens"] == 128]
                    ratios = [a["seconds"]/b["seconds"] for _,a,b in common]
                    value = dict(n=len(common),ratio=statistics.mean(ratios) if ratios else None,
                        ci95=bootstrap(ratios) if ratios else None,
                        length_ratio=statistics.mean(a["tokens"]/b["tokens"] for _,a,b in common) if common else None)
                    if stage == "natural":
                        value.update(lost=[s["request_id"] for s,a,b in pairs if quality_ok(b) and not quality_ok(a)],
                                     gained=[s["request_id"] for s,a,b in pairs if quality_ok(a) and not quality_ok(b)])
                    family[arm][stage][baseline] = value
    return result


def audit_allocation(rows):
    """Reconcile construction snapshots with the actual selected-path ledger."""
    blocks, shortened = 0, 0
    for row in rows:
        if row["arm"] not in ("SpineTree-ASD","JointTree-ASD"):
            continue
        mode = "full" if row["arm"] == "SpineTree-ASD" else "remaining"
        spent = 0.
        for block in row["rounds"]:
            meta = block["tree_budget"]
            horizon = block["proposed"]
            snapshot = RegretBudget(8,max(0.,8-spent),.25,2)
            requested = horizon if mode == "full" else snapshot.spine_length(horizon)
            reserved = meta["spine_reserved"]
            if (meta["spine_mode"] != mode or meta["horizon"] != horizon
                    or meta["spine_requested"] != requested
                    or not math.isclose(meta["regret_before"],snapshot.remaining,abs_tol=1e-9)
                    or not 0 <= reserved <= min(requested,31)
                    or not reserved <= meta["max_depth"] <= horizon
                    or not reserved+1 <= block["tree_nodes"] <= 32
                    or not 0 <= block["relaxed_tokens"] <= 2
                    or not math.isfinite(block["regret"]) or block["regret"] < 0):
                raise ValueError("Invalid budget-conditioned allocation")
            spent += block["regret"]
            if spent > 8:
                raise ValueError("Joint-tree path exceeded actual budget")
            blocks += 1
            shortened += int(mode == "remaining" and requested < horizon)
        if row["asd"] is None or not math.isclose(spent,row["asd"]["spent"],abs_tol=1e-9):
            raise ValueError("Joint-tree selected-path ledger disagrees with summary")
    return dict(checked_blocks=blocks,remaining_shortened_blocks=shortened)
