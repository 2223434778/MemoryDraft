"""Post-run diagnosis only; never change the preregistered scores or stop gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memorydraft.evaluation.release_triage import (AREAS, CRITICAL, continuation_map, load_dataset,
                                                  oracle_coverage, request_text, sha)
from scripts.run_reuse_pilot import save
from scripts.write_release_triage_report import DATA, ROOT, audit, references


def schema_examples():
    """Two fictional schema renderings, independent of source notes, labels and outputs."""
    examples = []
    for status in ("not_mentioned", "review"):
        blocks = []
        for area in AREAS:
            evidence = '[]' if status == "not_mentioned" else '["L001"]'
            blocks.append('    {\n      "area": ' + json.dumps(area) + ',\n      "status": ' + json.dumps(status)
                          + ',\n      "evidence": ' + evidence + '\n    }')
        examples.append('{\n  "version": "0.0.0",\n  "checks": [\n' + ',\n'.join(blocks) + '\n  ]\n}')
    return examples


def classify_diagnosis(text, release, gold):
    value = json.loads(text)
    checks = value["checks"]
    base = (value["version"] == release["version"] and set(value) == {"version", "checks"}
            and len(checks) == 5 and [c["area"] for c in checks] == list(AREAS))
    status_correct, unsupported, cap = [], {}, []
    critical_false_negative, critical_false_positive = [], []
    for area, check in zip(AREAS, checks):
        expected = "review" if area in gold else "not_mentioned"
        if check["status"] == expected:
            status_correct.append(area)
        if len(check["evidence"]) > 2:
            cap.append(area)
        if area in gold:
            bad = [line for line in check["evidence"] if line not in gold[area]["lines"]]
            if bad:
                unsupported[area] = bad
        elif check["evidence"]:
            unsupported[area] = check["evidence"]
        if area in CRITICAL and check["status"] != expected:
            (critical_false_negative if area in gold else critical_false_positive).append(area)
    return dict(base_json_identity_and_area_order=base, status_correct=status_correct,
                status_all_correct=len(status_correct) == 5, evidence_count_violations=cap,
                unsupported_evidence=unsupported, critical_status_false_negative=critical_false_negative,
                critical_status_false_positive=critical_false_positive)


def run(source):
    from transformers import AutoTokenizer

    projects, labels, _ = load_dataset(DATA)
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    rows, raw_hash = references(source)
    audit(projects, rows, provenance["eos_token_ids"])
    lookup = {(r["project_id"], r["turn"]): r for r in rows}
    tokenizer = AutoTokenizer.from_pretrained(provenance["target"]["model"], local_files_only=True)
    examples = schema_examples()
    static = continuation_map(examples, tokenizer)
    diagnoses = []
    for project in projects:
        for turn, release in enumerate(project["releases"]):
            row = lookup[project["id"], turn]
            entry = dict(project_id=project["id"], turn=turn,
                         **classify_diagnosis(row["text"], release, labels[project["id"] + ":" + release["tag"]]))
            if turn:
                context = continuation_map([request_text(project, turn)], tokenizer)
                merged = {key: value | static.get(key, set()) for key, value in context.items()}
                for key, value in static.items():
                    merged.setdefault(key, value)
                baseline, _ = oracle_coverage(row["token_ids"], merged)
                history, _ = oracle_coverage(row["token_ids"], continuation_map(
                    [lookup[project["id"], t]["text"] for t in range(turn)], tokenizer))
                entry.update(output_tokens=len(row["token_ids"]), schema_plus_prompt_covered=len(baseline),
                             history_beyond_schema=len(history - baseline))
            diagnoses.append(entry)
    warm = [d for d in diagnoses if d["turn"]]
    summary = dict(base_json_identity_and_area_order=sum(d["base_json_identity_and_area_order"] for d in diagnoses),
        status_correct_requests=sum(d["status_all_correct"] for d in diagnoses),
        status_correct_categories=sum(len(d["status_correct"]) for d in diagnoses),
        evidence_count_violation_requests=sum(bool(d["evidence_count_violations"]) for d in diagnoses),
        unsupported_evidence_requests=sum(bool(d["unsupported_evidence"]) for d in diagnoses),
        critical_status_false_negatives=sum(len(d["critical_status_false_negative"]) for d in diagnoses),
        critical_status_false_positives=sum(len(d["critical_status_false_positive"]) for d in diagnoses),
        warm_output_tokens=sum(d["output_tokens"] for d in warm),
        history_beyond_schema=sum(d["history_beyond_schema"] for d in warm),
        static_example_tokens=sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in examples))
    summary["history_beyond_schema_fraction"] = summary["history_beyond_schema"] / summary["warm_output_tokens"]
    result = dict(post_run_exploratory=True, does_not_change_gate=True, measured_speedup=False,
        references_sha256=raw_hash, runtime_source_sha256=provenance["source"]["source_sha256"],
        analysis_script_sha256=sha(Path(__file__).read_bytes()),
        data_sha256=sha((DATA / "manifest.json").read_bytes()),
        static_example_sha256=[sha(s.encode()) for s in examples], static_examples=examples,
        summary=summary, rows=diagnoses)
    save(source / "posthoc-diagnosis.json", result)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/release-triage-feasibility")
    args = parser.parse_args()
    run(args.source)
