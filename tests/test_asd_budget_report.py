from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from memorydraft.evaluation.asd_budget import load_tasks, score
from memorydraft.evaluation.runtime_migration import file_hash
from scripts.run_asd_budget_probe import POLICIES
from scripts.write_asd_budget_report import analyze, render


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def completed(tmp_path):
    data = "data/research/asd-budget/records.json"
    records = load_tasks(data)
    for benchmark in ("gsm8k", "math500"):
        root = tmp_path / benchmark
        root.mkdir()
        info = dict(status="complete", original_unchanged=True, data_sha256=file_hash(data),
                    policies={m:asdict(c) for m,c in POLICIES.items()}, source=dict(sha256="runtime"))
        write(root / "provenance.json", info)
        refs = {}
        for stage, repeats in (("mechanism", 1), ("natural", 2)):
            directory = root / stage
            directory.mkdir()
            rows = []
            for record in records:
                if record["benchmark"] != benchmark:
                    continue
                text = record["answer"] if benchmark == "gsm8k" else "\\boxed{" + record["answer"] + "}"
                for method, config in POLICIES.items():
                    row = dict(request_id=record["id"], method=method, benchmark=benchmark, token_ids=[1,2],
                               text=text, prompt_sha256="prompt-"+record["id"], snapshot_id="empty", strict_identity=True,
                               quality=score(record,text), finish_reason="eos", latency_seconds=1., counters=dict(rounds=1),
                               asd=dict(budget=config.budget, spent=0., relaxed_tokens=0, blocks=1, value=config.value,
                                        selection=config.selection, regret_price=config.regret_price,
                                        shortened_blocks=0, foregone_proposal_tokens=0, approximate=config.budget>0),
                               events=[dict(accepted=1, strict_accepted=1, mismatch=[False], regrets=[0.], values=[1.],
                                            feasible_accepted=1, relaxed=0, regret=0., spent=0., generated_before=1)])
                    if method == "DFlash":
                        refs[record["id"]] = deepcopy(row)
                    for repeat in range(repeats):
                        rows.append({**row,"repeat":repeat})
            (directory / "rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
        write(root / "references.json", refs)
    return tmp_path


def test_complete_report_and_raw_rows_are_unchanged(completed):
    raw = completed / "gsm8k/natural/rows.jsonl"
    before = file_hash(raw)
    a, info = analyze(completed)
    assert a["benchmarks"]["gsm8k"]["ASD-Utility"]["quality_correct"] == 16
    assert a["benchmarks"]["math500"]["ASD"]["quality_correct"] == 8
    assert a["comparisons"]["gsm8k"]["ASD-Utility"]["latency_change_percent"] == 0
    assert "没有忽略EOS的强制续写" in render(a,info)
    assert file_hash(raw) == before


@pytest.mark.parametrize("kind", ["budget", "duplicate", "trajectory", "ledger"])
def test_bad_evidence_is_rejected(completed, kind):
    path = completed / "gsm8k/natural/rows.jsonl"
    if kind == "ledger":
        path = completed / "gsm8k/mechanism/rows.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    if kind == "budget":
        rows[0]["asd"]["spent"] = 9
    elif kind == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif kind == "trajectory":
        rows[1]["token_ids"] = [3]
    else:
        rows[0]["events"][0]["regret"] = 1
    path.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
    with pytest.raises(ValueError):
        analyze(completed)


def test_published_budget_report_rebuilds_from_complete_evidence():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    source = root / "results/asd-budget"
    if not source.exists():
        pytest.skip("GPU evidence not yet published in this checkout")
    a, infos = analyze(source)
    assert render(a, infos) == (root / "docs/ASD_BUDGET_RESULTS.md").read_text(encoding="utf-8")
