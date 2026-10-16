"""Recompute pilot coverage from saved logs; never runs models or uses the network."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def audit(rows):
    groups = {}
    for row in rows:
        key = (row["request_id"], row["repeat"])
        group = groups.setdefault(key, {})
        if row["method"] in group:
            raise ValueError(f"Duplicate method for {key}")
        group[row["method"]] = row
    if not groups:
        raise ValueError("Empty audit input")
    methods = set(next(iter(groups.values())))
    for key, group in groups.items():
        if "AR" not in group or set(group) != methods:
            raise ValueError(f"Unpaired methods for {key}")
        reference = group["AR"]
        for row in group.values():
            if row["correct"] is not True or row["token_ids"] != reference["token_ids"]:
                raise ValueError(f"Output mismatch for {key}")
            if (row["snapshot_id"], row["history_condition"]) != (
                reference["snapshot_id"], reference["history_condition"]
            ):
                raise ValueError(f"Snapshot/condition mismatch for {key}")
            events = row["rounds"]
            if row["method"] != "AR":
                if sum(e["committed"] for e in events) != len(row["token_ids"]):
                    raise ValueError(f"Committed token mismatch for {key}")
            for event in events:
                if not 0 <= event["accepted"] <= event["proposed"] or event["committed"] <= 0:
                    raise ValueError(f"Invalid candidate counts for {key}")
    ar_seconds = sum(g["AR"]["latency_seconds"] for g in groups.values())
    result = {}
    for method in sorted(methods):
        subset = [g[method] for g in groups.values()]
        events = [e for row in subset for e in row["rounds"]]
        proposed = sum(e["proposed"] for e in events)
        accepted = sum(e["accepted"] for e in events)
        r_events = [e for e in events if e["route"] == "R"]
        output_tokens = sum(len(row["token_ids"]) for row in subset)
        seconds = sum(row["latency_seconds"] for row in subset)
        result[method] = {
            "requests": len(subset), "output_tokens": output_tokens,
            "latency_seconds": seconds, "speedup_vs_ar": ar_seconds / seconds,
            "rounds": len(events), "route_rounds": dict(Counter(e["route"] for e in events)),
            "proposed": proposed, "accepted": accepted,
            "acceptance_rate": accepted / proposed if proposed else None,
            "accepted_per_round": accepted / len(events) if events else None,
            "r_proposal_round_fraction": len(r_events) / len(events) if events else None,
            "r_accepted": sum(e["accepted"] for e in r_events),
            "r_accepted_output_fraction": sum(e["accepted"] for e in r_events) / output_tokens
            if output_tokens else None,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = {
        "source": args.input.as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
        "note": "Offline recount only. D/H round boundaries are not matched-prefix observations.",
        "methods": audit([json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["methods"], ensure_ascii=False, indent=2))
