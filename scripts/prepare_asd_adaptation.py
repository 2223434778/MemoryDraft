"""Offline question-only training split; never read generated evaluation outcomes."""
import argparse
import json
from pathlib import Path
import re
import shutil

from memorydraft.evaluation.adaptation import digest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"


def normalized(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def prepare(source, output):
    raw = source.read_bytes()
    if digest(raw) != SOURCE_SHA:
        raise ValueError("Changed pinned official GSM8K train source")
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    excluded = {normalized(r["question"]) for r in json.loads((ROOT/"data/research/asd-suite/gsm8k.json").read_text(encoding="utf-8"))}
    # The complete official test source also covers every old selected GSM task.
    test = ROOT/"artifacts/asd-attribution-gsm-source.jsonl"
    if digest(test.read_bytes()) != "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14":
        raise ValueError("Changed pinned official GSM test source")
    excluded.update(normalized(json.loads(line)["question"]) for line in test.read_text(encoding="utf-8").splitlines())
    candidates = [dict(id=f"gsm8k-train:{i}", benchmark="gsm8k", source_index=i, question=r["question"])
                  for i, r in enumerate(rows) if normalized(r["question"]) not in excluded]
    candidates.sort(key=lambda r: digest(("MemoryDraft-adaptation-v1:"+r["id"]).encode()))
    selected = [{**r, "split": "train" if i < 32 else "calibration"} for i, r in enumerate(candidates[:40])]
    if len(selected) != 40 or len({normalized(r["question"]) for r in selected}) != 40:
        raise ValueError("Insufficient unique nonoverlapping training questions")
    output.mkdir(parents=True, exist_ok=False)
    data = (json.dumps(selected, ensure_ascii=False, indent=2)+"\n").encode()
    (output/"questions.json").write_bytes(data)
    shutil.copyfile(ROOT/"data/research/asd-attribution/GSM8K_LICENSE.txt", output/"GSM8K_LICENSE.txt")
    info = dict(schema=1, source_commit="3101c7d5072418e28b9008a6636bde82a006892c",
                repository="https://github.com/openai/grade-school-math", path="grade_school_math/data/train.jsonl",
                source_sha256=SOURCE_SHA, questions_sha256=digest(data), license="MIT", train=32, calibration=8,
                selection="SHA256 rank MemoryDraft-adaptation-v1:<source ID>, question-only; no answers or generated outcomes",
                exclusion="Normalized exact question overlap with complete official GSM8K test and suite removed",
                excluded_source_questions=len(rows)-len(candidates), ids={s:[r["id"] for r in selected if r["split"]==s] for s in ("train", "calibration")})
    (output/"manifest.json").write_bytes((json.dumps(info, indent=2)+"\n").encode())
    print(json.dumps(info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT/"data/research/asd-adaptation")
    args = parser.parse_args()
    prepare(args.source, args.output)
