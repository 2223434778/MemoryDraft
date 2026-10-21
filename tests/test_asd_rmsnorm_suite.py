from copy import deepcopy

import pytest

from memorydraft.evaluation.runtime_migration import read_rows
from scripts import run_asd_rmsnorm_suite_probe as runner
from scripts import write_asd_rmsnorm_suite_report as report


def synthetic(row, method, profile):
    """Prior real trajectories with SYNTHETIC timing; never performance evidence."""
    r = deepcopy(row)
    latency = 1. if profile == "eager" else .97
    r.update(profile=profile, engine_latency_seconds=latency, latency_seconds=latency,
             policy_setup_seconds=0., profile_switch_seconds=0., graphs_before=4, graphs_after=4)
    if method not in {"AR", "DFlash"}:
        r["cpu_execution"] = dict(fast_calls=len(r["rounds"]), fallback_calls=0)
    return r


def scoring_map(previous):
    return {(q, r["text"]): deepcopy(r["quality"]) for (q, _), r in previous.items()}


def test_full_development_shards_and_frozen_pair_order():
    records, positions = runner.records_and_positions()
    assert len(records) == 48 and {r["split"] for r in records} == {"development"}
    for shard in (0, 1):
        selected = [r for r in records if positions[r["id"]]%2 == shard]
        assert len(selected) == 24
        assert all(sum(r["benchmark"] == b for r in selected) == 4 for b in report.BENCHMARKS)
    order = runner.timing_order(records[0]["id"], 0)
    assert order == runner.timing_order(records[0]["id"], 0) and len(set(order)) == 8
    assert all(order[i][0] == order[i+1][0] and order[i][1] != order[i+1][1] for i in range(0, 8, 2))


def test_question_26_generations_scores_after_pairs_and_reuses_scores(monkeypatch, tmp_path):
    previous = runner.previous_rows()
    record = runner.records_and_positions()[0][0]
    quality = scoring_map(previous)
    events = []

    def execute(engine, compiler, r, method, profile, limit):
        events.append("generate")
        assert limit == 4096
        return synthetic(previous[r["id"], method], method, profile)

    def score(r, text):
        events.append("score")
        assert events.count("generate")%2 == 0
        return quality[r["id"], text]

    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(runner, "score", score)
    counts = runner.run_question(None, None, record, previous, tmp_path)
    assert counts["reference"] == 2 and counts["timed"] == 24 and counts["quality_rows"] == 10
    assert events.count("generate") == 26 and events.count("score") == counts["unique_texts_scored"]
    assert len(read_rows(tmp_path/"rows.jsonl")) == 24
    assert len(read_rows(tmp_path/"references.jsonl")) == 2


@pytest.mark.parametrize("drift", ["eager", "repeat"])
def test_question_rejects_old_eager_or_repeat_drift(monkeypatch, tmp_path, drift):
    previous = runner.previous_rows()
    record = runner.records_and_positions()[0][0]
    quality, calls = scoring_map(previous), {}

    def execute(engine, compiler, r, method, profile, limit):
        row = synthetic(previous[r["id"], method], method, profile)
        key = method, profile
        calls[key] = calls.get(key, 0)+1
        if (drift == "eager" and profile == "eager") or (drift == "repeat" and calls[key] > 1):
            row["token_ids"][0] += 1
        return row

    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(runner, "score", lambda r, text: quality[r["id"], text])
    with pytest.raises(ValueError, match="changed trajectory"):
        runner.run_question(None, None, record, previous, tmp_path)


def test_question_preserves_compiled_wrong_answer_instead_of_filtering(monkeypatch, tmp_path):
    previous = runner.previous_rows()
    record = runner.records_and_positions()[0][0]
    quality = scoring_map(previous)

    def execute(engine, compiler, r, method, profile, limit):
        row = synthetic(previous[r["id"], method], method, profile)
        if method == "ASD" and profile != "eager":
            row["token_ids"][0] += 1
            row["text"] = "Synthetic incorrect answer"
        return row

    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(runner, "score", lambda r, text: dict(correct=False) if text.startswith("Synthetic") else quality[r["id"], text])
    assert runner.run_question(None, None, record, previous, tmp_path)["timed"] == 24
    changed = [r for r in read_rows(tmp_path/"rows.jsonl") if r["method"] == "ASD" and r["profile"] != "eager"]
    assert len(changed) == 3 and all(not r["quality"]["correct"] for r in changed)


@pytest.fixture
def matrix(monkeypatch):
    previous = runner.previous_rows()
    records = {r["id"]: r for r in runner.records_and_positions()[0]}
    quality = scoring_map(previous)
    monkeypatch.setattr(report, "score", lambda r, text: deepcopy(quality[r["id"], text]))
    rows, refs = [], []
    for q in records:
        for method in ("AR", *runner.METHODS):
            for profile in runner.PROFILES:
                for repeat in range(1 if method == "AR" else runner.REPEATS):
                    row = synthetic(previous[q, method], method, profile)
                    order = runner.timing_order(q, -1, ("AR",)) if method == "AR" else runner.timing_order(q, repeat)
                    row.update(stage="reference" if method == "AR" else "natural", repeat=repeat,
                               slot=order.index((method, profile)))
                    (refs if method == "AR" else rows).append(row)
    return rows, refs, records, previous, {151643, 151645}


def test_full_report_audits_1248_and_48_question_statistics(matrix):
    rows, refs, records, previous, stops = matrix
    result = report.audit(rows, refs, records, previous, stops)
    assert result["timed"] == 1152 and result["references"] == 96 and result["quality_rows"] == 480
    assert result["compiled_token_matches"] == 240
    summary = report.summarize(rows, refs, records)
    assert all(v["n"] == 48 and v["ratio"] == pytest.approx(.97) for v in summary["overall"].values())
    assert all(v["new36"]["n"] == 36 and v["old12"]["n"] == 12 for v in summary["subsets"].values())
    assert all(v["n"] == 8 for family in summary["families"].values() for v in family.values())


@pytest.mark.parametrize("change", ["missing", "profile", "repeat", "quality", "cost", "input", "eos", "calls", "order"])
def test_report_rejects_corrupted_natural_measurements(matrix, change):
    rows, refs, records, previous, stops = matrix
    if change == "missing":
        rows.pop()
    elif change == "profile":
        rows[0]["profile"] = "mlp-inductor"
    elif change == "repeat":
        rows[1]["token_ids"][0] += 1
    elif change == "quality":
        rows[0]["quality"]["correct"] = not rows[0]["quality"]["correct"]
    elif change == "cost":
        rows[0]["policy_setup_seconds"] = -1.
        rows[0]["engine_latency_seconds"] += 1.
    elif change == "input":
        refs[0]["prompt_sha256"] = "changed"
    elif change == "eos":
        rows[0]["token_ids"].insert(0, next(iter(stops)))
    elif change == "calls":
        next(r for r in rows if r["method"] == "ASD")["cpu_execution"]["fast_calls"] += 1
    else:
        rows[0]["slot"] = -1
    with pytest.raises(ValueError):
        report.audit(rows, refs, records, previous, stops)


def test_drift_and_formal_compile_are_retained_in_primary_matrix(matrix):
    rows, refs, records, previous, stops = matrix
    q = next(iter(records))
    changed = [r for r in rows if r["request_id"] == q and r["method"] == "DFlash" and r["profile"] != "eager"]
    for r in changed:
        r["token_ids"][0] += 1
    changed[0]["graphs_after"] += 1
    result = report.audit(rows, refs, records, previous, stops)
    assert result["compiled_token_matches"] == 239 and result["strict_matches_AR"][runner.PROFILES[1]] == 47
    assert result["compile_events_during_formal"] == 1
