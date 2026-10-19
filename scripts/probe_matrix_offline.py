"""Replay saved proposals first; score a common pool without any new Target calls."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path

import ot
from transformers import AutoTokenizer

from memorydraft.evaluation.matrix_probe import SCORED, inputs, observed_rows, protected, state_digest
from memorydraft.evaluation.runtime_migration import file_hash
from memorydraft.evaluation.splice_probe import input_digest, prefix_acceptance
from memorydraft.memory.store import MemoryStore
from memorydraft.retrieval.matrix import assess_alignment
from memorydraft.retrieval.splice import SpliceReuseMatcher
from memorydraft.retrieval.suffix import SuffixArrayIndex
from memorydraft.runtime.benchmark import source_digest
from scripts.diagnose_transport_online import write
from scripts.run_runtime_migration import append


ROOT = Path(__file__).resolve().parents[1]


def score_pool(pool, matcher, draft, remaining):
    candidates, decisions = [], {}
    for item in pool:
        assessments = {}
        for method in SCORED[1:]:
            value = assess_alignment(item["query"], item["history"], matcher.config, ot.sinkhorn,
                                     draft_prefix=item["prefix_length"], tail=method in {"Splice-TailOT", "Splice-AllOT"},
                                     order=method in {"Splice-OrderOT", "Splice-AllOT"},
                                     structure=method in {"Splice-PlanOT", "Splice-AllOT"})
            assessments[method] = dict(score=value.score if math.isfinite(value.score) else None,
                                       admissible=value.admissible, features=value.features)
        candidates.append(dict(tokens=list(item["tokens"]), prefix_length=item["prefix_length"],
                               position=item["position"], source=item["source"], cheap=item["cheap"],
                               assessments=assessments, oracle_accepted=prefix_acceptance(item["tokens"], remaining)))
    for method in SCORED:
        ranked = []
        for i, item in enumerate(candidates):
            score = item["cheap"] if method == "Splice-Edit" else item["assessments"][method]["score"]
            if score is not None and score >= matcher.config.minimum_score:
                ranked.append((score, -item["prefix_length"], -item["position"], i))
        selected = max(ranked)[-1] if ranked else None
        decisions[method] = dict(selected=selected, accepted=candidates[selected]["oracle_accepted"] if selected is not None
                                else prefix_acceptance(draft, remaining), fallback=selected is None)
    return candidates, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks, events = inputs(ROOT)
    keyed = {t.request.id: t for t in tasks}
    hashes = protected(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    info = dict(status="started", source=source_digest(), POT=ot.__version__, inputs_sha256=input_digest(tasks, events),
                runner_sha256=file_hash(Path(__file__)), protocol_sha256=file_hash(ROOT / "docs/OT_MATRIX_PROTOCOL.md"),
                tokenizer_sha256=file_hash(args.tokenizer / "tokenizer.json"), original_sha256=hashes,
                scope="Post-hoc common-state CPU scoring; no Target forward or timing claim")
    write(args.output / "provenance.json", info)
    frames, seen, replayed = [], set(), 0
    try:
        with MemoryStore(args.output / "history.db") as store:
            store.append(events)
            for row in observed_rows(ROOT):
                task = keyed[row["request_id"]]
                snapshot = store.snapshot(task.request.user_id, task.request.order)
                if snapshot.snapshot_id != task.snapshot_id:
                    raise ValueError("Changed offline legal snapshot")
                index = SuffixArrayIndex(snapshot, tokenizer, min_match=3, max_match=32,
                                         prompt_text="\n".join(m["content"] for m in task.request.messages))
                matcher = SpliceReuseMatcher(index, method="edit" if row["method"] == "Splice-Edit" else "ot",
                                             tail_weighted=row["method"] == "Splice-TailOT")
                calls = {c["generated"]: c for c in row["calls"]}
                for step in row["rounds"]:
                    generated = step["generated_before"]
                    confirmed = task.reference["token_ids"][:generated]
                    budget = min(32, task.request.max_new_tokens - generated)
                    if step["lookup"] == "global":
                        proposal = matcher.propose(confirmed, budget)
                        if proposal is None or proposal.matched < 6:
                            raise ValueError("Exact replay changed")
                        matcher.observe(proposal, step["accepted"], [*confirmed, *proposal.tokens[:step["accepted"]]])
                    if generated not in calls:
                        continue
                    call, pool = calls[generated], []
                    draft = call["draft_tokens"]
                    eligible = (len(confirmed) >= 8 and 4 <= len(draft) <= 32 and budget > len(draft)
                                and matcher.stats["guided_lookups"] < 4 and bool(index.tokens))
                    if eligible:
                        pool = matcher.splice_shortlist(confirmed, draft, budget)
                    proposal = matcher.propose_with_draft(confirmed, draft, budget)
                    actual = asdict(proposal) if proposal else None
                    # JSON normalizes tuple/list while preserving all endpoint metadata.
                    if state_digest(actual) != state_digest(call["proposal"]):
                        raise ValueError(f"Saved proposal changed at {task.request.id}:{generated}:{row['method']}")
                    replayed += 1
                    signature = state_digest([task.request.id, generated, draft, matcher.blocked,
                                              [(c["tokens"], c["prefix_length"], c["position"]) for c in pool]])
                    if pool and signature not in seen:
                        seen.add(signature)
                        candidates, decisions = score_pool(pool, matcher, draft, task.reference["token_ids"][generated:])
                        frame = dict(request_id=task.request.id, project_id=task.project_id, generated=generated,
                                     snapshot_id=task.snapshot_id, state_sha256=signature,
                                     draft_accepted=prefix_acceptance(draft, task.reference["token_ids"][generated:]),
                                     candidates=candidates, decisions=decisions)
                        frames.append(frame)
                        append(args.output / "frames.jsonl", frame)
                    if proposal:
                        matcher.observe(proposal, call["accepted"], [*confirmed, *proposal.tokens[:call["accepted"]]])
        changed = {method: sum(f["decisions"][method]["selected"] != f["decisions"]["Splice-OT"]["selected"]
                               for f in frames) for method in SCORED[-3:]}
        summary = dict(frames=len(frames), replayed_calls=replayed, changed=changed,
                       gpu_gate=any(changed.values()), methods={m: dict(
                           accepted=sum(f["decisions"][m]["accepted"] for f in frames),
                           fallback=sum(f["decisions"][m]["fallback"] for f in frames),
                           worse_than_draft=sum(f["decisions"][m]["accepted"] < f["draft_accepted"] for f in frames)) for m in SCORED})
        write(args.output / "summary.json", summary)
        info.update(status="complete", frames=len(frames), replayed_calls=replayed,
                    gpu_gate=summary["gpu_gate"], original_unchanged=hashes == protected(ROOT))
        if not info["original_unchanged"]:
            raise ValueError("Protected evidence changed")
        print(summary, flush=True)
    except Exception as error:
        info.update(status="failed", frames=len(frames), replayed_calls=replayed, error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        write(args.output / "provenance.json", info)


if __name__ == "__main__":
    main()
