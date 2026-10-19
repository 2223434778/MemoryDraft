import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.write_matrix_report import build


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/ot-matrix"


def test_report_rebuilds_from_actual_common_states_and_gpu_results():
    report, diagnostic = build(SOURCE)
    assert report.encode() == (ROOT / "docs/OT_MATRIX_RESULTS.md").read_bytes()
    assert diagnostic == json.loads((SOURCE / "diagnostic.json").read_text())
    assert diagnostic["offline"]["replayed_calls"] == 135
    assert diagnostic["timing_counter_paths_match_mechanism"]


@pytest.mark.parametrize("change", ["source", "oracle", "decision", "features", "missing", "solves"])
def test_report_rejects_tampered_evidence(tmp_path, change):
    destination = tmp_path / "evidence"
    shutil.copytree(SOURCE, destination)
    if change == "source":
        path = destination / "offline/provenance.json"
        value = json.loads(path.read_text())
        value["source"]["sha256"] = "wrong"
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path = destination / ("mechanism/rows.jsonl.gz" if change in {"missing", "solves"} else "offline/frames.jsonl.gz")
        rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
        if change == "oracle":
            rows[0]["candidates"][0]["oracle_accepted"] += 1
        elif change == "decision":
            rows[0]["decisions"]["Splice-PlanOT"]["accepted"] += 1
        elif change == "features":
            rows[0]["candidates"][0]["assessments"]["Splice-PlanOT"]["features"]["support"] = 2.
        elif change == "missing":
            rows.pop()
        else:
            next(r for r in rows if r["method"] == "Splice-AllOT")["counters"]["matrix_solves"] = 1000
        path.write_bytes(gzip.compress(("\n".join(json.dumps(r) for r in rows) + "\n").encode(), mtime=0))
    with pytest.raises(ValueError):
        build(destination)


def test_cli_preserves_equivalent_published_diagnostic_bytes(tmp_path):
    destination = tmp_path / "evidence"
    shutil.copytree(SOURCE, destination)
    path = destination / "diagnostic.json"
    path.write_text(json.dumps(json.loads(path.read_text()), indent=1), encoding="utf-8")
    before = path.read_bytes()
    subprocess.run([sys.executable, "-m", "scripts.write_matrix_report", "--source", str(destination),
                    "--report", str(tmp_path / "report.md")], cwd=ROOT, check=True, capture_output=True,
                   env={**os.environ, "PYTHONHASHSEED": "321"})
    assert path.read_bytes() == before
