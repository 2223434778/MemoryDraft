import json
from pathlib import Path
import shutil
import os
import subprocess
import sys

import pytest

from scripts.write_splice_cooldown_report import build


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/ot-splice-cooldown"


def test_repair_report_rebuilds_and_original_evidence_is_preserved():
    import hashlib

    report, diagnostic = build(SOURCE)
    assert report.encode() == (ROOT / "docs/OT_SPLICE_COOLDOWN_RESULTS.md").read_bytes()
    assert diagnostic == json.loads((SOURCE / "diagnostic.json").read_text())
    prior = json.loads((ROOT / "results/ot-splice/publication.json").read_text())
    for name, sha in prior["files"].items():
        assert hashlib.sha256((ROOT / "results/ot-splice" / name).read_bytes()).hexdigest() == sha


@pytest.mark.parametrize("change", ["source", "scope", "summary"])
def test_repair_report_rejects_wrong_scope_or_execution(tmp_path, change):
    destination = tmp_path / "repair"
    shutil.copytree(SOURCE, destination)
    path = destination / "mechanism/provenance.json"
    value = json.loads(path.read_text())
    if change == "source":
        value["source"]["sha256"] = "wrong"
    elif change == "scope":
        value["cooldown_check"] = False
    else:
        path = destination / "timing/summary.json"
        value = json.loads(path.read_text())
        value["all"]["Splice-OT"]["seconds"] = 1.
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        build(destination)


@pytest.mark.parametrize("repair", [False, True])
def test_report_cli_preserves_equivalent_json_bytes_across_hash_seeds(tmp_path, repair):
    source = SOURCE if repair else ROOT / "results/ot-splice"
    destination = tmp_path / "evidence"
    shutil.copytree(source, destination)
    path = destination / "diagnostic.json"
    # Legal equivalent serialization deliberately differs from the writer's format.
    path.write_text(json.dumps(json.loads(path.read_text()), sort_keys=True, indent=1), encoding="utf-8")
    before = path.read_bytes()
    module = "scripts.write_splice_cooldown_report" if repair else "scripts.write_splice_report"
    subprocess.run([sys.executable, "-m", module, "--source", str(destination), "--report", str(tmp_path / "report.md")],
                   cwd=ROOT, check=True, capture_output=True, env={**os.environ, "PYTHONHASHSEED": "123"})
    assert path.read_bytes() == before
