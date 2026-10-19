#!/usr/bin/env bash
set -euo pipefail
cd /media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-acceptance
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=.
mkdir -p artifacts
../.venv-dflash/bin/python -m pytest -q tests/test_acceptance.py tests/test_acceptance_probe.py tests/test_dflash_reuse.py --basetemp artifacts/test-tmp --junitxml=artifacts/acceptance-tests-server.xml
../.venv-dflash/bin/python -u -m scripts.collect_acceptance_probe --config configs/runtime-ot-benefit.json --output results/raw/acceptance/collection
../.venv-dflash/bin/python -m scripts.evaluate_acceptance_probe --input results/raw/acceptance/collection --output results/raw/acceptance/evaluation
echo PIPELINE_COMPLETE
