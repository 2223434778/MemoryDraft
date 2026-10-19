import gzip
import json
from pathlib import Path
import shutil

import pytest

from scripts.write_asd_splice_report import analyze, render

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/asd-splice"


def test_completed_asd_report_counts_and_excludes_post_eos_opportunities():
    analysis, stages, info = analyze(SOURCE)
    assert [len(stages[s]) for s in ("mechanism", "natural", "fixed")] == [72, 144, 144]
    assert all(v["strict_identity"] == 8 for v in analysis["stages"]["mechanism"].values())
    assert all(v["budget8_eligible"] == 0 for v in analysis["blockers"].values())
    assert analysis["blockers"]["Splice"]["count"] == 22
    assert render(analysis, info) == (ROOT / "docs/ASD_SPLICE_RESULTS.md").read_text(encoding="utf-8")


def test_report_rejects_overspent_approximate_request(tmp_path):
    shutil.copytree(SOURCE, tmp_path / "probe")
    source = tmp_path / "probe"
    path = source / "natural/rows.jsonl.gz"
    rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
    row = next(r for r in rows if r["method"] == "Splice-ASD")
    row["asd"]["spent"] = row["asd"]["budget"] + 1
    path.write_bytes(gzip.compress(("\n".join(json.dumps(r) for r in rows) + "\n").encode()))
    with pytest.raises(ValueError, match="Invalid request budget"):
        analyze(source)
