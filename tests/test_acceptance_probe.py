import pytest

from memorydraft.evaluation.benefit_probe import calibration
from memorydraft.evaluation.runtime_migration import read_rows
from scripts.evaluate_acceptance_probe import ROOT, metrics, old_curve


def test_reconstructs_every_original_prediction_without_current_project():
    samples = calibration(ROOT)
    rows = read_rows(ROOT / "results/ot-benefit/offline/rows.jsonl")
    assert len(rows) == 14
    for row in rows:
        curve = old_curve(samples, row)
        changed = [{**s, "outcomes": {k: [0, 32] for k in s["outcomes"]}}
                   if s["project"] == row["project_id"] else s for s in samples]
        assert old_curve(changed, row) == curve
        assert sum(curve) == pytest.approx(row["predicted_accepted"])


def test_survival_metrics_are_length_and_position_sensitive():
    assert metrics([1., 1., 0.], 2) == dict(predicted=2., actual=2, error=0., bias=0., brier=0.)
    assert metrics([1., .5, .5], 2)["brier"] == pytest.approx(1 / 6)
    with pytest.raises(ValueError):
        metrics([.3, .8], 1)
    with pytest.raises(ValueError):
        metrics([float("nan")], 1)
