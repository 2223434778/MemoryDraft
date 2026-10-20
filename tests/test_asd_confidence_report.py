from dataclasses import asdict
import json
import shutil

import pytest
import torch

from memorydraft.backends.asd import ASDConfig
from memorydraft.backends.asd_confidence import ConfidenceUtilityVerifier, token_confidence
from scripts.write_asd_confidence_report import ROOT, analyze, decision, render


def test_independent_scalar_prices_match_tensor_including_eos():
    torch.set_num_threads(1)
    torch.manual_seed(142)
    for alpha in (0, .5):
        for n in range(40):
            tokens = torch.randint(0, 8, (7,))
            logits = torch.randn(7, 8)
            config = ASDConfig(budget=8, gate=1, selection="utility", regret_price=(n % 4))
            verifier = ConfidenceUtilityVerifier(config, alpha=alpha, record=True)
            verifier.spent = float(n % 8)
            before = verifier.spent
            stop = [2] if n % 2 else []
            observed = verifier.decide(tokens, logits, stop_token_ids=stop)
            event = verifier.events[0]
            event.update(proposal_eos=[i + 1 for i, t in enumerate(tokens.tolist()) if t in stop],
                         pair_mass=token_confidence(tokens, logits)["pair_mass"].tolist())
            expected = decision(event, config, before, alpha=alpha)
            assert all(expected[k] == v for k, v in asdict(observed).items())


@pytest.fixture
def evidence(tmp_path):
    source = ROOT / "results/asd-confidence"
    if not source.exists():
        pytest.skip("GPU confidence evidence not yet published")
    target = tmp_path / "evidence"
    shutil.copytree(source, target)
    return target


def test_published_report_rebuilds_exactly(evidence):
    a = analyze(evidence)
    assert {k: v for k, v in a["counts"].items() if k != "branches"} == dict(mechanism=28, natural=56, prefix=56)
    assert 0 < a["counts"]["branches"] <= 16 and a["counts"]["branches"] % 2 == 0
    assert render(a) == (ROOT / "docs/ASD_CONFIDENCE_RESULTS.md").read_text(encoding="utf-8")


def test_changed_alpha_or_protocol_rejected(evidence):
    path = evidence / "math-a/provenance.json"
    info = json.loads(path.read_text())
    info["alpha"] = .8
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="Frozen configuration"):
        analyze(evidence)


def test_outcome_based_intervention_selection_rejected(evidence):
    path = evidence / "math-a/selections.json"
    data = json.loads(path.read_text())
    data["math500:4"]["matched"] = not data["math500:4"]["matched"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="selected using changed rule"):
        analyze(evidence)
