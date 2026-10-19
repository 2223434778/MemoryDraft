"""Explicit continuation over all observed references, with task quality reported."""
from __future__ import annotations

import json

from memorydraft.evaluation.prefix_risk_probe import protected as previous_protected
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.evaluation.transport_validation import attach_references


def protected(root):
    hashes = previous_protected(root)
    directory = root / "results/prefix-risk"
    hashes.update({p.relative_to(root).as_posix(): file_hash(p) for p in sorted(directory.rglob("*"))
                   if p.is_file() and p.relative_to(directory).parts[0] != "continuation"})
    return hashes


def reference_source(root, tasks):
    directory = root / "results/prefix-risk/mechanism"
    info = json.loads((directory / "provenance.json").read_text())
    if (info["status"] != "stopped_target_contract" or info["references"] != 12
            or any(info["frozen_sha256"][name] != file_hash(root / name) for name in
                   ("data/research/prefix-risk/cases.json", "data/research/prefix-risk/calibration.json"))):
        raise ValueError("Continuation requires the unchanged original inputs and recorded reference set")
    references = read_rows(directory / "fresh-ar.jsonl")
    attached = attach_references(tasks, references)
    metadata = dict(reference_sha256=file_hash(directory / "fresh-ar.jsonl.gz"),
                    provenance_sha256=file_hash(directory / "provenance.json"),
                    source_package_sha256=info["source"]["sha256"], references=len(references),
                    contract_correct=sum(r["contract"]["artifact_correct"] for r in references),
                    historical_ar_seconds=sum(r["latency_seconds"] for r in references))
    return attached, metadata, info["engine"]["config"]


def check_reference_profile(previous, current):
    keys = ("backend", "target_revision", "dtype", "attention", "threads", "template_options",
            "max_input_tokens", "max_new_tokens", "reuse")
    if any(previous[k] != current[k] for k in keys):
        raise ValueError("Continuation changed the frozen reference inference profile")
