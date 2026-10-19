#!/usr/bin/env bash
set -euo pipefail
cd /media/a6000/D/ZhaoBinHe/MemoryDraft/prefix-risk
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=.
mkdir -p artifacts
../.venv-dflash/bin/python -m pytest -q tests/test_prefix_risk.py tests/test_prefix_risk_probe.py tests/test_runtime.py tests/test_dflash_reuse.py --basetemp artifacts/test-tmp --junitxml=artifacts/prefix-risk-tests-server.xml
../.venv-dflash/bin/python -u -m scripts.run_prefix_risk_probe --stage mechanism --config configs/runtime-ot-benefit.json --output results/raw/prefix-risk/mechanism
if ../.venv-dflash/bin/python -c 'import json; p=json.load(open("results/raw/prefix-risk/mechanism/provenance.json")); raise SystemExit(0 if p["status"]=="complete" and p["timing_gate"]["passed"] else 1)'; then
  ../.venv-dflash/bin/python -u -m scripts.run_prefix_risk_probe --stage timing --config configs/runtime-ot-benefit.json --mechanism results/raw/prefix-risk/mechanism --output results/raw/prefix-risk/timing
else
  echo STOPPED_AT_FROZEN_GATE
fi
echo PIPELINE_COMPLETE
