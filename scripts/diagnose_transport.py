"""Score isolated prefix decisions; no model inference or online speed claim."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from transformers import AutoTokenizer

from memorydraft.evaluation.runtime_migration import file_hash, input_hashes
from memorydraft.evaluation.transport_probe import probe_inputs, prefix_length, raw_span_oracle
from memorydraft.memory.store import MemoryStore
from memorydraft.retrieval.reuse import ReuseMatcher
from memorydraft.retrieval.suffix import SuffixArrayIndex
from memorydraft.retrieval.transport import AlignedReuseMatcher, AlignmentConfig
from memorydraft.runtime.benchmark import source_digest, write_json


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/qwen3-8b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    originals = {name: input_hashes(ROOT / "results" / name) for name in ("reuse-confirmation", "infra-v0.3")}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tasks, events = probe_inputs(ROOT)
    rows, totals = [], {}
    with MemoryStore(args.output / "history.db") as store:
        store.append(events)
        for task in tasks:
            snapshot = store.snapshot(task.request.user_id, task.request.order)
            if snapshot.snapshot_id != task.snapshot_id:
                raise ValueError("Wrong legal snapshot")
            index = SuffixArrayIndex(snapshot, tokenizer, prompt_text="\n".join(m["content"] for m in task.request.messages))
            matchers = {method: AlignedReuseMatcher(index, method=method) for method in ("ot", "edit")}
            exact = ReuseMatcher(index, cursor=False)
            tokens = task.reference["token_ids"]
            # Every prefix is scored independently; no online feedback/call cap is simulated.
            for position in range(8, len(tokens)):
                prefix, remaining = tokens[:position], tokens[position:]
                proposal = exact.propose(prefix, 32)
                if proposal is not None and proposal.matched < 6:
                    proposal = None
                row = {"request_id": task.request.id, "position": position,
                       "suffix": prefix_length(proposal, remaining), "suffix_available": proposal is not None}
                if proposal is not None:
                    rows.append(row)
                    continue
                pool = matchers["ot"].shortlist(prefix)
                row["pool_oracle"] = raw_span_oracle(index, remaining,
                    ((p, matchers["ot"].bounds[p][1]) for p in pool))
                row["raw_oracle"] = raw_span_oracle(index, remaining)
                for method, matcher in matchers.items():
                    started = time.perf_counter()
                    selected = matcher.rank(prefix, 32)
                    row[method + "_seconds"] = time.perf_counter() - started
                    row[method] = prefix_length(selected, remaining)
                    row[method + "_proposed"] = selected is not None
                rows.append(row)
            selected = [r for r in rows if r["request_id"] == task.request.id]
            misses = [r for r in selected if not r["suffix_available"]]
            totals[task.request.id] = {"positions": len(selected), "misses": len(misses),
                "raw_ge8": sum(r["raw_oracle"] >= 8 for r in misses),
                "pool_ge8": sum(r["pool_oracle"] >= 8 for r in misses),
                **{method: {"proposed": sum(r[method + "_proposed"] for r in misses),
                            "recovered_ge8": sum(r[method] >= 8 for r in misses),
                            "zero_accept": sum(r[method + "_proposed"] and r[method] == 0 for r in misses),
                            "correct_prefix_sum": sum(r[method] for r in misses),
                            "lookup_seconds": sum(r[method + "_seconds"] for r in misses)} for method in matchers}}
    if originals != {name: input_hashes(ROOT / "results" / name) for name in originals}:
        raise ValueError("Original evidence changed")
    write_json(args.output / "offline.json", {"scope": "Post-hoc isolated decisions; all prefixes >=8, no online budget/feedback; no speed claim",
        "original_sha256": originals, "source": source_digest(), "runner_sha256": file_hash(Path(__file__)),
        "protocol_sha256": file_hash(ROOT / "docs/OT_R_PROTOCOL.md"), "config": asdict(AlignmentConfig()),
        "per_request": totals, "positions": rows})
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    main()
