"""Freeze unseen GSM test questions using source metadata, never model outcomes."""
import argparse
import json
from pathlib import Path
import shutil

from memorydraft.evaluation.adaptation import digest
from memorydraft.evaluation.adaptation_followup import STUDY, aligned_controls, previous_rows
from scripts.prepare_asd_adaptation import normalized

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"


def old_questions(value):
    if isinstance(value, dict):
        if isinstance(value.get("question"), str) and (value.get("benchmark") == "gsm8k" or str(value.get("id", "")).startswith("gsm8k")):
            yield value
        for child in value.values():
            yield from old_questions(child)
    elif isinstance(value, list):
        for child in value:
            yield from old_questions(child)


def prepare(source, output):
    raw = source.read_bytes()
    if digest(raw) != SOURCE_SHA:
        raise ValueError("Changed official GSM test source")
    exclusion_files, used_ids, used_text = {}, set(), set()
    for path in sorted((ROOT/"data/research").rglob("*.json")):
        if path.is_relative_to(output):
            continue
        found = list(old_questions(json.loads(path.read_text(encoding="utf-8"))))
        if found:
            exclusion_files[path.relative_to(ROOT).as_posix()] = digest(path.read_bytes())
            used_ids.update(r["id"] for r in found)
            used_text.update(normalized(r["question"]) for r in found)
    pool = [dict(r, id=f"gsm8k:{i}", benchmark="gsm8k", source_index=i, study_split="independent")
            for i, r in enumerate(json.loads(s) for s in raw.decode().splitlines())
            if i >= 32 and f"gsm8k:{i}" not in used_ids and normalized(r["question"]) not in used_text]
    pool.sort(key=lambda r: digest(("MemoryDraft-adaptation-followup-v1:"+r["id"]).encode()))
    chosen = pool[:16]
    if len(chosen) != 16 or len({normalized(r["question"]) for r in chosen}) != 16:
        raise ValueError("Insufficient independent questions")
    suite = json.loads((ROOT/"data/research/asd-suite/gsm8k.json").read_text(encoding="utf-8"))
    observed = [dict(r, study_split="observed") for r in suite if r["split"] == "development"]
    controls = aligned_controls(previous_rows(ROOT, "examples"), previous_rows(ROOT, "sources"))
    if len(controls) != 80 or {s:sum(r["split"] == s for r in controls) for s in ("train", "calibration")} != dict(train=64, calibration=16):
        raise ValueError("Cannot preserve the complete original ASD training pairs")
    output.mkdir(parents=True, exist_ok=False)
    data = (json.dumps([*observed, *chosen], ensure_ascii=False, indent=2)+"\n").encode()
    (output/"records.json").write_bytes(data)
    shutil.copyfile(ROOT/"data/research/asd-adaptation/GSM8K_LICENSE.txt", output/"GSM8K_LICENSE.txt")
    manifest = dict(schema=1, source_sha256=SOURCE_SHA, source_commit="3101c7d5072418e28b9008a6636bde82a006892c",
                    source_path="grade_school_math/data/test.jsonl", records_sha256=digest(data),
                    selection="SHA256 rank MemoryDraft-adaptation-followup-v1:<source ID>, first 16 after exclusions; no output filtering",
                    independent_ids=[r["id"] for r in chosen], observed_ids=[r["id"] for r in observed],
                    exclusion_files=exclusion_files, excluded_ids=sorted(used_ids), source_index_minimum=32,
                    exact_text_exclusion=True, matched_states=len(controls), matched_content_equal=sum(r["matched_content_equal"] for r in controls),
                    training_data="Original 32 train/8 calibration unchanged; no test questions used for updates or checkpoint selection",
                    exposure="Independent means no prior project selection/generation; public benchmark and upstream model exposure not excluded")
    (output/"manifest.json").write_bytes((json.dumps(manifest, indent=2)+"\n").encode())
    print(json.dumps(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT/f"data/research/{STUDY}")
    args = parser.parse_args()
    prepare(args.source, args.output)
