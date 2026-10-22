"""Position-matched adaptation controls without changing the frozen first study."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from memorydraft.evaluation.adaptation import digest, prefix_digest
from memorydraft.evaluation.runtime_migration import read_rows

STUDY = "asd-adaptation-followup"
VARIANTS = ("base", "ordinary", "ordinary-position", "asd-prefix")
POLICIES = ("DFlash", "ASD")
PREVIOUS_COMMIT = "4d7ac6eca2ab9970e9f71314fc77b9b9de79e140"
ADAPTER_SHA = {
    "ordinary": "b5bb3c12a87a37cdead2e9ce75479e5e38266f8be0999934ce3716a0232fc943",
    "asd-prefix": "885375d37670df6ec60e756bb018b9c5efc0f3121be23ee9c5424e14690822e0",
}


def previous_rows(root: Path, name: str):
    return [r for i in (0, 1) for r in read_rows(root/f"results/asd-adaptation/collect-{i}/{name}.jsonl")]


def aligned_controls(examples, sources):
    """Take strict tokens at each ASD state's exact generated-token position."""
    indexed = {}
    for row in sources:
        key = row["request_id"], row["source_policy"]
        if key in indexed:
            raise ValueError("Duplicate source trajectory")
        indexed[key] = row
    controls, seen = [], set()
    for row in examples:
        if row["source_policy"] != "asd-prefix":
            continue
        key = row["request_id"], row["source_round"]
        if key in seen:
            raise ValueError("Duplicate ASD pairing key")
        seen.add(key)
        n = row["generated_before"]
        prefix = row["prefix_ids"]
        if not 0 < n < len(prefix) or prefix_digest(prefix) != row["prefix_sha256"]:
            raise ValueError("Invalid actual ASD prefix")
        strict = indexed[row["request_id"], "ordinary"]
        asd = indexed[row["request_id"], "asd-prefix"]
        if (n > len(strict["token_ids"]) or (n == len(strict["token_ids"]) and strict["finish_reason"] == "eos")
                or n > len(asd["token_ids"]) or prefix[-n:] != asd["token_ids"][:n]):
            raise ValueError("Fixed matched position unavailable before EOS")
        prompt = prefix[:-n]
        ids = prompt+strict["token_ids"][:n]
        controls.append(dict(request_id=row["request_id"], split=row["split"], source_round=row["source_round"],
                             source_policy="ordinary-position", generated_before=n, prefix_ids=ids,
                             prefix_sha256=prefix_digest(ids), matched_asd_prefix_sha256=row["prefix_sha256"],
                             prompt_tokens=len(prompt), matched_content_equal=ids == prefix,
                             past_spent=0., past_relaxed=0))
    if not controls:
        raise ValueError("Empty position-matched training source")
    return sorted(controls, key=lambda r: (r["request_id"], r["source_round"]))


def records(root: Path):
    base = root/f"data/research/{STUDY}"
    meta = json.loads((base/"manifest.json").read_text(encoding="utf-8"))
    raw = (base/"records.json").read_bytes()
    if digest(raw) != meta["records_sha256"]:
        raise ValueError("Changed frozen followup questions")
    rows = json.loads(raw)
    if (len(rows) != 24 or len({r["id"] for r in rows}) != 24
            or any(r["benchmark"] != "gsm8k" for r in rows)
            or {s:sum(r["study_split"] == s for r in rows) for s in ("observed", "independent")} != dict(observed=8, independent=16)):
        raise ValueError("Changed observed/independent evaluation coverage")
    return rows


def check_previous_publication(root: Path):
    meta = json.loads((root/"results/asd-adaptation/publication.json").read_text(encoding="utf-8"))
    if meta["execution_commit"] != PREVIOUS_COMMIT:
        raise ValueError("Changed prior study execution")
    selected = {n:h for n, h in meta["files"].items() if n.startswith(("results/asd-adaptation/", "data/research/asd-adaptation/"))
                or n == "memorydraft/evaluation/adaptation.py"}
    for name, sha in selected.items():
        if digest((root/name).read_bytes()) != sha:
            raise ValueError("Previous public evidence changed: "+name)
    return len(selected)


def previous_root(root: Path):
    previous = root.parent/"asd-adaptation-v2"
    if not previous.is_dir() or previous == root:
        raise ValueError("Read-only previous feature/adapter root unavailable")
    return previous


def load_examples(root: Path, rows, *, previous: bool):
    base = previous_root(root) if previous else root
    cache, result = {}, []
    for row in rows:
        path = (base/row["feature_path"]).resolve()
        if not path.is_relative_to((base/"artifacts").resolve()):
            raise ValueError("Feature path must stay inside its artifact root")
        if row["feature_path"] not in cache:
            if digest(path.read_bytes()) != row["feature_sha256"]:
                raise ValueError("Changed teacher features")
            cache[row["feature_path"]] = torch.load(path, map_location="cpu", weights_only=True)
        sample = cache[row["feature_path"]]
        if (sample["labels"][0].tolist() != row["teacher_tokens"]
                or sample["prefix_ids"][0].tolist() != row["prefix_ids"]
                or prefix_digest(row["prefix_ids"]) != row["prefix_sha256"]):
            raise ValueError("Teacher features disagree with public metadata")
        result.append((row, sample))
    return result


def adapter_paths(root: Path):
    result = {}
    for arm in ("ordinary", "asd-prefix", "ordinary-position"):
        if arm == "ordinary-position":
            meta = json.loads((root/f"results/raw/{STUDY}/train/provenance.json").read_text(encoding="utf-8"))
            if meta["status"] != "complete":
                raise ValueError("Incomplete matched adaptation")
            base, expected = root, meta["adapter_sha256"]
        else:
            meta = json.loads((root/f"results/asd-adaptation/train-{arm}/provenance.json").read_text(encoding="utf-8"))
            base, expected = previous_root(root), ADAPTER_SHA[arm]
        path = (base/meta["adapter_path"]).resolve()
        if not path.is_relative_to((base/"artifacts").resolve()) or digest(path.read_bytes()) != expected:
            raise ValueError("Selected adapter identity changed")
        result[arm] = (path, dict(adapter_sha256=expected, selected_step=meta["selected_step"],
                                 selected_calibration_loss=meta["selected_calibration_loss"]))
    return result
