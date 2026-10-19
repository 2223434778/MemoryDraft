#!/usr/bin/env bash
set -euo pipefail
cd /media/a6000/D/ZhaoBinHe/MemoryDraft/prefix-risk-continuation
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=.
mkdir -p artifacts
../.venv-dflash/bin/python -m pytest -q tests/test_prefix_risk_continuation.py tests/test_prefix_risk.py tests/test_prefix_risk_probe.py tests/test_prefix_risk_report.py tests/test_runtime.py tests/test_dflash_reuse.py --basetemp artifacts/test-tmp --junitxml=artifacts/continuation-tests-server.xml
../.venv-dflash/bin/python -u -m scripts.run_prefix_risk_probe --continue-evaluation --stage mechanism --config configs/runtime-ot-benefit.json --output results/raw/prefix-risk-continuation/mechanism
../.venv-dflash/bin/python -u -m scripts.run_prefix_risk_probe --continue-evaluation --stage timing --config configs/runtime-ot-benefit.json --mechanism results/raw/prefix-risk-continuation/mechanism --output results/raw/prefix-risk-continuation/timing
echo PIPELINE_COMPLETE
