"""Sequential bounded GPU suite; start after checksum-verified weights arrive."""
import json
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path


root = Path(__file__).resolve().parents[1]
env = dict(os.environ, OMP_NUM_THREADS="4", PYTHONPATH=str(root), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
out = root / "results/qwen3-pilot"
out.mkdir(parents=True, exist_ok=True)


def run(name, *args, required=True):
    log = (out / (name + ".log")).open("w")
    command = [sys.executable, "-u", "scripts/run_research.py", "--output", str(out / name), *args]
    print("command", command, flush=True)
    code = subprocess.call(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    if required and code:
        raise RuntimeError(f"Stage failed: {name}; evidence saved")
    return code


if __name__ == "__main__":
    manifest = json.loads((root / "configs/qwen3-download-manifest.json").read_text())
    # Downloader only renames .partial after full SHA-256 validation.
    deadline = time.monotonic() + 7200
    marker = root / "models/qwen3-weights-verified.json"
    expected_manifest = hashlib.sha256((root / "configs/qwen3-download-manifest.json").read_bytes()).hexdigest()
    while not (marker.exists() and json.loads(marker.read_text())["manifest_sha256"] == expected_manifest and
               all((root / "models" / x["destination"] / x["filename"]).exists() for x in manifest)):
        if time.monotonic() > deadline:
            raise RuntimeError("Pinned weights not ready within this run's two-hour limit")
        time.sleep(10)
    profile = ["--dtype", "bfloat16", "--attention", "eager"]
    code = run("smoke-bf16", "--stage", "smoke", "--max-new-tokens", "64", *profile, required=False)
    if code:
        failure = json.loads((out / "smoke-bf16/provenance.json").read_text())
        if "Greedy mismatch" not in failure.get("error", ""):
            raise RuntimeError("BF16 smoke failed for a non-numerical reason; inspect evidence")
        profile = ["--dtype", "float32", "--attention", "eager"]
        run("smoke-fp32", "--stage", "smoke", "--max-new-tokens", "64", *profile)
    # Same development requests and output budget for all candidate lengths.
    run("calibration", "--stage", "smoke", "--max-new-tokens", "64", "--blocks", "2", "4",
        "--methods", "AR", "D_native", "R", *profile)
    smoke_name = "smoke-fp32" if profile[1] == "float32" else "smoke-bf16"
    choices = [(json.loads((out / smoke_name / "summary-k8.json").read_text())["D_native"]["speedup_vs_ar"], 8)]
    for k in (2, 4):
        summary = json.loads((out / "calibration" / f"summary-k{k}.json").read_text())
        choices.append((summary["D_native"]["speedup_vs_ar"], k))
    block = max(choices)[1]
    (out / "selection.json").write_text(json.dumps(dict(profile=profile, block=block, dev_choices=choices), indent=2))
    print("frozen test configuration", profile, "block", block, flush=True)
    run("memory-test", "--stage", "measure", "--split", "test", "--suite", "memory",
        "--max-new-tokens", "128", "--repeats", "3", "--blocks", str(block),
        "--methods", "AR", "D_native", "D", "PLD", "R", "RD", "H", "RHD", *profile)
    run("prefix-test", "--stage", "prefix", "--split", "test", "--suite", "memory",
        "--max-new-tokens", "128", "--blocks", str(block), *profile)
    run("public-test", "--stage", "measure", "--split", "test", "--suite", "public",
        "--max-new-tokens", "128", "--repeats", "3", "--blocks", str(block),
        "--methods", "AR", "D_native", "PLD", "R", *profile)
    (out / "complete.txt").write_text("All stages completed; inspect summaries before making performance claims.\n")
