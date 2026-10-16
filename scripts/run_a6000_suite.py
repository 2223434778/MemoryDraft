"""Reproduce the bounded synthetic validation suite on the provisioned A6000."""
import os
import subprocess
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[1]
env = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", PYTHONPATH=str(root))


def run(*args):
    subprocess.run([sys.executable, *args], cwd=root, env=env, check=True)


run("scripts/make_pilot.py")
run("-m", "memorydraft", "import", "--db", "pilot.db", "--input", "data/synthetic/history.jsonl")
for name, config, methods, requests, condition in [
    ("ar-fallback", "a6000-validation", ["AR", "RHD"], "requests", "relevant"),
    ("semantic", "a6000-semantic", ["AR", "D", "R", "RD", "H", "RHD"], "requests", "relevant"),
    ("ablation-empty", "a6000-semantic", ["AR", "D", "H"], "h_cases", "empty"),
    ("ablation-unrelated", "a6000-semantic", ["AR", "D", "H"], "h_cases", "unrelated"),
]:
    run("-m", "memorydraft", "replay", "--config", f"configs/{config}.json", "--db", "pilot.db",
        "--requests", f"data/synthetic/{requests}.jsonl", "--output", f"results/{name}/replay.jsonl",
        "--max-new-tokens", "64", "--block-size", "8", "--history-condition", condition, "--methods", *methods)
run("-m", "memorydraft", "import", "--db", "demo.db", "--input", "data/synthetic/history.jsonl")
run("-m", "memorydraft", "generate", "--config", "configs/a6000-semantic.json", "--db", "demo.db",
    "--request", "data/synthetic/request.json", "--method", "RHD", "--max-new-tokens", "32",
    "--commit-order", "30", "--output", "results/demo/answer.json")
run("scripts/check_demo.py")
(root / "results/suite-complete.txt").write_text("All suite commands completed successfully.\n")
