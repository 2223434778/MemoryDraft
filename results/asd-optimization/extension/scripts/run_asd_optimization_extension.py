"""Frozen follow-up: precision/attention attribution and BF16 length interactions."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

from memorydraft.evaluation.runtime_migration import read_rows
from scripts.write_asd_optimization_report import summarize


def choose_lengths(rows):
    table = summarize(rows)
    selected, gates = ["D-fast", "ASD-fast", "U-fast"], {}
    for arm in ("D-verify7", "U-verify7", "D-draft8", "U-draft8", "U-recent", "U-budget"):
        m = table["bf16-sdpa/" + arm]
        baseline = table["bf16-sdpa/" + ("D-fast" if arm.startswith("D-") else "U-fast")]
        passed = (m["prefix_complete"] and baseline["prefix_complete"] and
                  m["prefix_seconds"] <= .95 * baseline["prefix_seconds"] and
                  m["correct"] >= baseline["correct"] and m["truncated"] <= baseline["truncated"])
        gates[arm] = dict(passed=passed, prefix_ratio=m["prefix_seconds"] / baseline["prefix_seconds"],
                          correct=m["correct"], reference_correct=baseline["correct"])
        if passed:
            selected.append(arm)
    return dict(arms=selected, gates=gates)


def main():
    root = Path.cwd()
    manifest_name = "ASD_OPTIMIZATION_EXTENSION_SOURCE_MANIFEST.json"
    manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    base = root / "results/raw/asd-optimization-extension"
    base.mkdir(parents=True, exist_ok=False)
    (root / "artifacts").mkdir(exist_ok=True)

    def verify():
        for p, digest in manifest["files"].items():
            if hashlib.sha256((root / p).read_bytes()).hexdigest() != digest:
                raise ValueError("Execution file changed: " + p)
        return len(manifest["files"])

    info = dict(status="started", commit=manifest["commit"], files_verified_before=verify())
    exe = str(root.parent / ".venv-dflash/bin/python")
    env = dict(os.environ, OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1", PYTHONPATH=".")
    core = ["D-fast", "ASD-fast", "U-fast"]
    lengths = ["D-verify7", "U-verify7", "D-draft8", "U-draft8", "U-recent", "U-budget"]

    def worker(job):
        gpu, profile, stage, ids, arms = job
        part = f"{stage}-{profile}-{gpu}"
        command = [exe, "-u", "-m", "scripts.run_asd_optimization_probe", "--config", "configs/runtime-dflash.json",
                   "--profile", profile, "--task-ids", *ids, "--output", str(base / part), "--arms", *arms,
                   "--transfer" if stage == "transfer" else "--attribution"]
        print("START", part, json.dumps(command), flush=True)
        with (base / (part + ".log")).open("w") as log:
            result = subprocess.run(command, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT)
        print("FINISH", part, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(part + " failed " + str(result.returncode))

    def run(profile, stage, arms):
        ids = (["math500:1", "math500:4"], ["math500:7", "gsm8k:21"]) if stage == "attribution" else (
            ["math500:8", "gsm8k:24"], ["math500:9", "gsm8k:25"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, [(i, profile, stage, ids[i], arms) for i in (0, 1)]))

    try:
        subprocess.run([exe, "-m", "pytest", "-q", "tests/test_asd_optimization.py", "tests/test_asd_optimization_report.py",
                        "--basetemp", "artifacts/test-tmp", "--junitxml=artifacts/tests-server.xml"], env=env, check=True)
        run("bf16-sdpa", "attribution", core + lengths)
        run("bf16-eager", "attribution", core)
        rows = []
        for p in sorted(base.glob("attribution-bf16-sdpa-*/rows.jsonl")):
            rows.extend(dict(r, experiment_profile="bf16-sdpa") for r in read_rows(p))
        choice = choose_lengths(rows)
        (base / "length-selection.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")
        if len(choice["arms"]) > len(core):
            run("bf16-sdpa", "transfer", choice["arms"])
        info.update(status="complete", files_verified_after=verify())
    except Exception as error:
        info.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        (base / "execution-verification.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        with zipfile.ZipFile(root.parent / "asd-optimization-extension-results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    n = p.relative_to(base).as_posix()
                    if n.endswith(".jsonl"):
                        archive.writestr(n + ".gz", gzip.compress(p.read_bytes(), mtime=0))
                    else:
                        archive.write(p, n)
            for n in (manifest_name, "scripts/run_asd_optimization_extension.py", "artifacts/tests-server.xml"):
                archive.write(root / n, n.replace("artifacts/", ""))
    print("EXTENSION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
