from copy import deepcopy
import gzip
import json
import shutil

import pytest

from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from scripts.write_asd_completion_report import ROOT, analyze, render


@pytest.fixture
def evidence(tmp_path):
    source = ROOT / "results/asd-completion"
    if not source.exists():
        pytest.skip("GPU completion evidence is not yet published")
    target = tmp_path / "evidence"
    shutil.copytree(source,target)
    return target


def test_published_report_rebuilds_and_does_not_rewrite_rows(evidence):
    path=evidence/"part-a/mechanism/rows.jsonl.gz"
    original=file_hash(path)
    a,infos=analyze(evidence)
    assert len(a["requests"])==3
    assert all(t["correct"]==3 and t["truncated"]==0 for t in a["methods"].values())
    assert all(r["previous_prefix"]["exact_previous_prefix"] for rows in a["requests"].values() for r in rows.values())
    assert render(a,infos)==(ROOT/"docs/ASD_COMPLETION_RESULTS.md").read_text(encoding="utf-8")
    assert file_hash(path)==original


@pytest.mark.parametrize("kind", ["budget","duplicate","prefix","trajectory","policy"])
def test_corrupt_completion_evidence_is_rejected(evidence,kind):
    if kind=="policy":
        path=evidence/"part-a/provenance.json"
        value=json.loads(path.read_text(encoding="utf-8"))
        value["max_new_tokens"]=512
        path.write_text(json.dumps(value),encoding="utf-8")
    else:
        base=evidence/"part-a"/("natural" if kind=="trajectory" else "mechanism")/"rows.jsonl"
        rows=read_rows(base)
        if kind=="budget":rows[0]["asd"]["spent"]=9
        elif kind=="duplicate":rows.append(deepcopy(rows[0]))
        elif kind=="prefix":rows[0]["previous_prefix"]["common_prefix_tokens"]=0
        else:rows[1]["token_ids"][600]=999999
        base.with_suffix(".jsonl.gz").write_bytes(gzip.compress("".join(json.dumps(r)+"\n" for r in rows).encode(),mtime=0))
    with pytest.raises(ValueError):
        analyze(evidence)
