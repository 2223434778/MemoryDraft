import gzip
import json
from pathlib import Path
import shutil

import pytest

from memorydraft.evaluation.runtime_migration import file_hash
from scripts.write_asd_bench_report import analyze, render

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/asd-bench"


def test_report_corrects_numeric_wrappers_without_rewriting_original_scores():
    before = {p: file_hash(p) for p in SOURCE.rglob("*") if p.is_file()}
    a, info = analyze(SOURCE)
    assert a["stages"]["mechanism"]["gsm8k"]["DFlash"]["quality_correct"] == 8
    assert a["stages"]["mechanism"]["gsm8k"]["DFlash-ASD"]["quality_correct"] == 8
    assert a["stages"]["mechanism"]["mbpp"]["DFlash-ASD"]["quality_correct"] == 5
    assert a["score_corrections"]
    assert a["fixed_mbpp_continuation"]["DFlash"]["fraction"] > .7
    assert all(v["natural_prefixes_match"] for v in a["fixed_mbpp_continuation"].values())
    assert before == {p: file_hash(p) for p in before}
    assert render(a, info) == (ROOT / "docs/ASD_BENCH_RESULTS.md").read_text(encoding="utf-8")


def test_public_task_report_rejects_overspent_budget(tmp_path):
    source = tmp_path / "probe"
    shutil.copytree(SOURCE, source)
    path = source / "natural/rows.jsonl.gz"
    rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
    row = next(r for r in rows if r["method"] == "DFlash-ASD")
    row["asd"]["spent"] = 9.
    path.write_bytes(gzip.compress(("\n".join(json.dumps(r) for r in rows) + "\n").encode()))
    with pytest.raises(ValueError, match="Overspent"):
        analyze(source)
