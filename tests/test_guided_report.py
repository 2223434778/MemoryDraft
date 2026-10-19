import json
from pathlib import Path
import shutil

import pytest

from memorydraft.evaluation.runtime_migration import read_rows
from scripts.write_guided_report import rebuild


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("change", [None, "tokens", "missing", "forwards", "source"])
def test_report_reaudits_paired_execution_and_rejects_tampering(tmp_path, change):
    source = tmp_path / "evidence"
    shutil.copytree(ROOT / "results/ot-guided", source)
    development = source / "development"
    rows = read_rows(development / "rows.jsonl")
    if change == "tokens":
        rows[0]["token_ids"] = [999]
    elif change == "missing":
        rows.pop()
    elif change == "forwards":
        next(r for r in rows if r["method"] == "Guided-OT")["counters"]["draft_forward_rounds"] = 0
    elif change == "source":
        path = source / "holdout/provenance.json"
        info = json.loads(path.read_text(encoding="utf-8"))
        info["source"]["sha256"] = "different-implementation"
        path.write_text(json.dumps(info), encoding="utf-8")
    (development / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    if change is None:
        result = rebuild(source, tmp_path / "report.md")
        assert result["status"] in {"complete", "stopped_target_contract"}
        assert b"\r" not in (tmp_path / "report.md").read_bytes()
    else:
        with pytest.raises(ValueError):
            rebuild(source, tmp_path / "report.md")
