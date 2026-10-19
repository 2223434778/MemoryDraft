"""Reproduce the post-hoc shortlist-miss analysis without loading a model."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import tempfile

from transformers import AutoTokenizer

from memorydraft.evaluation.transport_diagnosis import accepted_prefix
from memorydraft.memory.store import MemoryStore
from memorydraft.retrieval.suffix import SuffixArrayIndex
from memorydraft.retrieval.transport import AlignedReuseMatcher, AlignmentConfig
from scripts.diagnose_transport_online import inputs, write
from scripts.write_transport_diagnosis_report import read_json


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/ot-diagnosis")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/qwen3-8b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks, events = inputs(ROOT)
    task = next(t for t in tasks if t.request.id == "payment-runbook:1")
    row = next(r for r in read_json(args.source / "decisions.json")
               if r["request_id"] == task.request.id and r["method"] == "OT-R")
    call = next(c for c in row["calls"] if c["generated_before"] == 15)
    prefix, remaining = row["token_ids"][:15], row["token_ids"][15:]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    with tempfile.TemporaryDirectory(prefix="memorydraft-ot-diagnosis-") as directory:
        with MemoryStore(Path(directory) / "history.db") as store:
            store.append(events)
            snapshot = store.snapshot(task.request.user_id, task.request.order)
            if snapshot.snapshot_id != task.snapshot_id:
                raise ValueError("Changed legal snapshot")
        index = SuffixArrayIndex(snapshot, tokenizer, prompt_text="\n".join(m["content"] for m in task.request.messages))
    matcher = AlignedReuseMatcher(index, method="edit")
    details = []
    for shift in (2, 3, 4, 5, 8):
        config = replace(AlignmentConfig(), max_shift=shift)
        pool = AlignedReuseMatcher(index, method="edit", config=replace(config, shortlist=32)).shortlist(prefix)
        detail = dict(max_shift=shift, top8=pool[:8], top32=pool,
                      raw_best_in_top8=call["raw_position"] in pool[:8],
                      raw_best_in_top32=call["raw_position"] in pool)
        for name, chosen in (("top8", pool[:8]), ("top32", pool)):
            detail["pool_best_" + name] = max((accepted_prefix(index.tokens[p:min(p + 32, matcher.bounds[p][1])],
                                                               remaining) for p in chosen), default=0)
        detail["selected"] = {}
        for method in ("edit", "ot"):
            proposal = AlignedReuseMatcher(index, method=method, config=config).rank(prefix, 32)
            detail["selected"][method] = dict(position=proposal.position if proposal else None,
                                              accepted=accepted_prefix(proposal.tokens, remaining) if proposal else 0)
        details.append(detail)
    anchors = sorted(range(len(prefix)), key=lambda offset: (
        len(matcher.positions.get(prefix[offset], ())) or 10**12, -offset))[:4]
    value = dict(request=task.request.id, position=15, raw_best_position=call["raw_position"],
                 prefix_tokens=prefix, details=details,
                 anchors=[dict(offset=o, token=prefix[o], text=tokenizer.decode([prefix[o]]),
                               occurrences=[dict(position=p, source=matcher.bounds[p][2],
                                                 required_shift=10 - (p + len(prefix) - o - 1))
                                            for p in matcher.positions.get(prefix[o], ())]) for o in anchors],
                 candidate_examples=[dict(position=p,
                                          history=tokenizer.decode(index.tokens[max(matcher.bounds[p][0], p - 15):p]),
                                          next=tokenizer.decode(index.tokens[p:p + 8])) for p in (10, 202, 203, 204)],
                 all_source_bounds=list(zip(index.starts, index.ends, index.sources)))
    write(args.output, value)


if __name__ == "__main__":
    main()
