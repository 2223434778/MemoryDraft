import gzip
import json
from pathlib import Path
import shutil

import pytest

from scripts.write_splice_report import build


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/ot-splice"


def test_published_report_rebuilds_from_actual_replay():
    report, diagnostic = build(SOURCE)
    assert report.encode() == (ROOT / "docs/OT_SPLICE_RESULTS.md").read_bytes()
    assert diagnostic == json.loads((SOURCE / "diagnostic.json").read_text())
    assert diagnostic["mechanism_gate"]["passed"]


@pytest.mark.parametrize("change", ["source", "oracle", "prefix", "summary", "missing_timing"])
def test_report_rejects_tampered_execution_or_attribution(tmp_path, change):
    destination = tmp_path / "evidence"
    shutil.copytree(SOURCE, destination)
    if change == "source":
        path = destination / "mechanism/provenance.json"
        value = json.loads(path.read_text())
        value["source"]["files"]["retrieval/splice.py"] = "changed"
    elif change == "summary":
        path = destination / "mechanism/summary.json"
        value = json.loads(path.read_text())
        value["all"]["Splice-OT"]["counters"]["rounds"] += 1
    else:
        path = destination / ("timing/rows.jsonl.gz" if change == "missing_timing" else "mechanism/rows.jsonl.gz")
        rows = [json.loads(l) for l in gzip.decompress(path.read_bytes()).decode().splitlines()]
        if change == "missing_timing":
            rows.pop()
        else:
            row = next(r for r in rows if r["method"] == "Splice-OT" and any(c["proposal"] for c in r["calls"]))
            call = next(c for c in row["calls"] if c["proposal"])
            if change == "oracle":
                call["oracle_accepted"] += 1
            else:
                call["proposal"]["draft_prefix"] = 1000
        path.write_bytes(gzip.compress(("\n".join(json.dumps(r) for r in rows) + "\n").encode(), mtime=0))
        with pytest.raises(ValueError):
            build(destination)
        return
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        build(destination)
