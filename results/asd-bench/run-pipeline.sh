#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=.
../.venv-dflash/bin/python -m pytest -q tests/test_asd.py tests/test_asd_bench.py --basetemp artifacts/test-tmp --junitxml=artifacts/tests-server.xml
../.venv-dflash/bin/python -u -m scripts.run_asd_bench_probe --config configs/runtime-ot-benefit.json --output results/raw/asd-bench
