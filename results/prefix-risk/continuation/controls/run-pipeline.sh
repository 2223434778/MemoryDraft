#!/usr/bin/env bash
set -euo pipefail
cd /media/a6000/D/ZhaoBinHe/MemoryDraft/prefix-risk-controls
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=.
../.venv-dflash/bin/python -m pytest -q tests/test_prefix_risk_continuation.py tests/test_prefix_risk_continuation_report.py tests/test_prefix_risk_report.py --basetemp artifacts/test-tmp --junitxml=artifacts/controls-tests-server.xml
../.venv-dflash/bin/python -u -m scripts.run_prefix_risk_controls --config configs/runtime-ot-benefit.json --output results/raw/prefix-risk-controls
echo PIPELINE_COMPLETE
