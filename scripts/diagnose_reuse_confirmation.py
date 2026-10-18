"""Audit old positive results and quantify alternative candidate sources, without inference."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from transformers import AutoTokenizer

from memorydraft.evaluation.reuse_attribution import (ANCHOR, EXTRA_TOKEN_BUDGET, LIMIT, MINIMUM,
    candidates, coverage, legal_snapshots, resource_sets)
from memorydraft.evaluation.reuse_cases import confirmation_projects, messages
from scripts.run_reuse_pilot import CONFIRMATION_METHODS, digest, save
from scripts.write_reuse_report import audit, read


ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def input_hashes(source):
    return {path.name: sha(path.read_bytes()) for path in sorted(source.iterdir()) if path.is_file()}


def run(source, tokenizer_path, output):
    if output.exists():
        raise ValueError("Preserve existing attribution evidence; choose a new output path")
    started = time.perf_counter()
    before = input_hashes(source)
    projects = json.loads((source / "projects.json").read_text(encoding="utf-8"))
    initial = json.loads((source / "initial-history.json").read_text(encoding="utf-8"))
    refs, replay = read(source / "references.jsonl"), read(source / "replay.jsonl")
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    summary = audit(projects, initial, refs, replay, expected_projects=confirmation_projects(),
                    methods=CONFIRMATION_METHODS, repeats=3)
    if (provenance["status"] != "complete" or provenance["stage"] != "confirmation"
            or summary != json.loads((source / "summary.json").read_text(encoding="utf-8"))):
        raise ValueError("Original confirmation is incomplete or summary changed")
    snapshots = legal_snapshots(projects, initial, refs)
    lookup = {(row["project_id"], row["turn"]): row for row in refs}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    rows, resources, observed = [], [], Counter()
    for project in projects:
        for turn, request in enumerate(project["currents"]):
            row = lookup[project["id"], turn]
            prompt = tokenizer.apply_chat_template(messages(request), add_generation_prompt=True, enable_thinking=False)
            if digest(prompt) != row["prompt_sha256"] or tokenizer.decode(row["token_ids"], skip_special_tokens=True) != row["text"]:
                raise ValueError("Tokenizer/template differs from frozen prompts or outputs")
            snapshot = snapshots[project["id"], turn]
            sets = resource_sets(request, snapshot)
            measured, positions = coverage(row["token_ids"], request, sets, tokenizer)
            generated = candidates(request)
            compiled = generated["Compiled"]
            body = [token for token in row["token_ids"] if token not in tokenizer.all_special_ids]
            match = compiled == row["text"]
            token_match = tokenizer.encode(compiled, add_special_tokens=False) == body
            row_observed = Counter()
            for sample in replay:
                if (sample["project_id"], sample["turn"], sample["method"]) != (project["id"], turn, "Global"):
                    continue
                for event in sample["rounds"]:
                    if event["route"] != "R":
                        continue
                    start = event["generated_before"]
                    accepted = set(range(start, start + event["accepted"]))
                    if accepted and max(accepted) >= len(row["token_ids"]):
                        raise ValueError("Observed accepted tokens extend beyond frozen output")
                    row_observed["accepted"] += len(accepted)
                    for method in ("Structure", "Static", "Expanded", "Compiled"):
                        row_observed[method] += len(accepted & positions[method])
            observed.update(row_observed)
            rows.append(dict(project_id=project["id"], turn=turn, output_tokens=len(row["token_ids"]),
                request_sha256=sha(request.encode()), prompt_sha256=row["prompt_sha256"], snapshot_id=snapshot.snapshot_id,
                history=[dict(record_id=r.record_id, order=r.order, text_sha256=sha(r.text.encode())) for r in snapshot.records],
                compiled_text_exact=match, compiled_tokens_exact=token_match,
                compiled_semantic_exact=json.loads(compiled) == json.loads(row["text"]),
                expanded_equals_history=generated["Expanded"] == snapshot.records[0].text,
                expanded_semantic_exact=json.loads(generated["Expanded"]) == json.loads(row["text"]),
                coverage=measured, observed_r_tokens=dict(row_observed)))
            resources.append(dict(project_id=project["id"], turn=turn,
                candidates={name: dict(text=text, sha256=sha(text.encode()), tokens=len(tokenizer.encode(text, add_special_tokens=False)))
                            for name, text in generated.items()}))
    total = sum(row["output_tokens"] for row in rows)
    totals = {method: dict(covered=sum(row["coverage"][method]["covered"] for row in rows),
                          extra_tokens=sum(row["coverage"][method]["extra_tokens"] for row in rows))
              for method in rows[0]["coverage"]}
    extra = {method: sum(row["coverage"][method]["history_outside"] for row in rows)
             for method in ("Context", "Structure", "Static", "Expanded", "Compiled")}
    original_events = {method: Counter(event["route"] for row in replay if row["method"] == method for event in row["rounds"])
                       for method in ("Adapted", "Global", "Context")}
    matched_sizes = {method: sum(row["coverage"][method]["extra_tokens"] == row["coverage"]["Global"]["extra_tokens"]
                                for row in rows) for method in ("Static", "Expanded", "Compiled")}
    report = dict(post_run_exploratory=True, measured_new_speedup=False,
        original_source_sha256=provenance["source"]["source_sha256"], input_files_sha256=before,
        analysis_files_sha256={str(p.relative_to(ROOT)).replace("\\", "/"): sha(p.read_bytes())
            for p in (Path(__file__), ROOT / "memorydraft/evaluation/reuse_attribution.py")},
        tokenizer=str(tokenizer_path), oracle=dict(anchor=ANCHOR, limit=LIMIT, minimum=MINIMUM,
            extra_token_ceiling=EXTRA_TOKEN_BUDGET, exact_token_matched=False, candidate_boundaries=True),
        summary=dict(requests=len(rows), projects=len(projects), output_tokens=total, coverage=totals,
            history_outside=extra, history_outside_fraction={key: count / total for key, count in extra.items()},
            compiled_text_exact=sum(row["compiled_text_exact"] for row in rows),
            compiled_tokens_exact=sum(row["compiled_tokens_exact"] for row in rows),
            compiled_semantic_exact=sum(row["compiled_semantic_exact"] for row in rows),
            expanded_equals_history=sum(row["expanded_equals_history"] for row in rows),
            expanded_semantic_exact=sum(row["expanded_semantic_exact"] for row in rows),
            equal_history_token_count_requests=matched_sizes,
            observed_r_tokens=dict(observed), original_times=summary["all"]["times"], original_route_counts=original_events),
        resources=resources, rows=rows, offline_seconds=time.perf_counter() - started)
    if input_hashes(source) != before:
        raise ValueError("Original evidence changed during analysis")
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, report)
    save(output.parent / "decision.json", dict(post_run_exploratory=True, new_gpu_timing_started=False,
        all_compiled_text_and_tokens_exact=all(row["compiled_text_exact"] and row["compiled_tokens_exact"] for row in rows),
        history_outside_compiled=extra["Compiled"], measured_static_speedup=False,
        decision="stop_after_offline" if all(row["compiled_text_exact"] and row["compiled_tokens_exact"] for row in rows)
                 and extra["Compiled"] == 0 else "review_remaining_opportunity",
        analysis_sha256=sha(output.read_bytes())))
    print(json.dumps(report["summary"], indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/reuse-confirmation")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/qwen3-8b")
    parser.add_argument("--output", type=Path, default=ROOT / "results/raw/research-stage2/reuse-attribution/offline-final.json")
    args = parser.parse_args()
    run(args.source, args.tokenizer, args.output)
