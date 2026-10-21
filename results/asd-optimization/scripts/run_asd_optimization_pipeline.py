"""Frozen bounded optimization matrix and conditional transfer on two GPUs."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

from memorydraft.evaluation.runtime_migration import read_rows


def select_transfer(rows):
    """Frozen gate, without transfer labels or tuning; caller passes stage A only."""
    def metrics(profile, arm):
        selected = [r for r in rows if r["experiment_profile"] == profile and r["arm"] == arm]
        natural = [r for r in selected if r["max_new_tokens"] == 2048]
        prefix = [r for r in selected if r["max_new_tokens"] == 128]
        if len(natural) != 4 or len(prefix) != 8:
            raise ValueError("Incomplete frozen development matrix")
        return dict(seconds=sum(r["latency_seconds"] for r in prefix),
                    complete=all(len(r["token_ids"]) == 128 for r in prefix),
                    correct=sum(r["quality"]["correct"] for r in natural),
                    truncated=sum(r["finish_reason"] != "eos" for r in natural))

    base = metrics("fp32-eager", "D-fast")
    profiles = []
    for p in ("fp32-sdpa", "bf16-sdpa"):
        m = metrics(p, "D-fast")
        if m["complete"] and base["complete"] and m["seconds"] <= .95 * base["seconds"] and m["correct"] >= base["correct"]:
            profiles.append((m["seconds"], p))
    selected_profile = min(profiles)[1] if profiles else None
    arms = ["D-fast", "ASD-fast", "U-fast"]
    reasons = {}
    for arm in ("D-verify7", "U-verify7", "D-draft8", "U-draft8", "U-recent", "U-budget"):
        reference = metrics("fp32-eager", "D-fast" if arm.startswith("D-") else "U-fast")
        m = metrics("fp32-eager", arm)
        passed = (m["complete"] and reference["complete"] and m["seconds"] <= .95 * reference["seconds"]
                  and m["correct"] >= reference["correct"] and m["truncated"] <= reference["truncated"])
        reasons[arm] = dict(passed=passed, metrics=m, reference=reference)
        if passed:
            arms.append(arm)
    return dict(selected_profile=selected_profile, fp32_arms=arms, length_gates=reasons)


def main():
    root = Path.cwd()
    name = "ASD_OPTIMIZATION_SOURCE_MANIFEST.json"
    manifest = json.loads((root / name).read_text(encoding="utf-8"))

    def verify():
        for p, digest in manifest["files"].items():
            if hashlib.sha256((root / p).read_bytes()).hexdigest() != digest:
                raise ValueError("Execution file changed: " + p)
        return len(manifest["files"])

    info = dict(status="started", commit=manifest["commit"], files_verified_before=verify())
    exe = str(root.parent / ".venv-dflash/bin/python")
    env = dict(os.environ, OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1", PYTHONPATH=".")
    destination = root / "results/raw/asd-optimization"
    destination.mkdir(parents=True, exist_ok=False)
    (root / "artifacts").mkdir(exist_ok=True)

    def worker(job):
        gpu, profile, stage, ids, arms = job
        part = f"{stage}-{profile}-{gpu}"
        command = [exe, "-u", "-m", "scripts.run_asd_optimization_probe", "--config", "configs/runtime-dflash.json",
                   "--profile", profile, "--task-ids", *ids, "--output", str(destination / part)]
        if stage == "transfer":
            command += ["--transfer", "--arms", *arms]
        print("START", part, json.dumps(command), flush=True)
        with (destination / (part + ".log")).open("w") as log:
            result = subprocess.run(command, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT)
        print("FINISH", part, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(part + " failed " + str(result.returncode))

    try:
        subprocess.run([exe, "-m", "pytest", "-q", "tests/test_asd_optimization.py", "tests/test_asd.py",
                        "tests/test_dflash_reuse.py", "--basetemp", "artifacts/test-tmp",
                        "--junitxml=artifacts/tests-server.xml"], env=env, check=True)
        for profile in ("fp32-eager", "fp32-sdpa", "bf16-sdpa"):
            jobs = [(0, profile, "development", ["math500:1", "math500:4"], None),
                    (1, profile, "development", ["math500:7", "gsm8k:21"], None)]
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(worker, jobs))
        rows = []
        for p in sorted(destination.glob("development-*")):
            if p.is_dir():
                rows.extend({**r, "experiment_profile": p.name[len("development-"):-2]} for r in read_rows(p / "rows.jsonl"))
        selection = select_transfer(rows)
        (destination / "transfer-selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        profiles = [("fp32-eager", selection["fp32_arms"])]
        if selection["selected_profile"]:
            profiles.append((selection["selected_profile"], ["D-fast", "ASD-fast", "U-fast"]))
        for profile, arms in profiles:
            jobs = [(0, profile, "transfer", ["math500:8", "gsm8k:24"], arms),
                    (1, profile, "transfer", ["math500:9", "gsm8k:25"], arms)]
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(worker, jobs))
        info.update(status="complete", files_verified_after=verify())
    except Exception as error:
        info.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        (destination / "execution-verification.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        with zipfile.ZipFile(root.parent / "asd-optimization-results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(destination.rglob("*")):
                if p.is_file():
                    n = p.relative_to(destination).as_posix()
                    if n.endswith(".jsonl"):
                        archive.writestr(n + ".gz", gzip.compress(p.read_bytes(), mtime=0))
                    else:
                        archive.write(p, n)
            for n in (name, "scripts/run_asd_optimization_pipeline.py", "artifacts/tests-server.xml"):
                archive.write(root / n, n.replace("artifacts/", ""))
    print("PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
