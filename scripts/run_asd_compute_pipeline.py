"""Run frozen compute forks on two isolated GPUs and package public evidence."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile


def main():
    root = Path.cwd()
    name = "ASD_COMPUTE_SOURCE_MANIFEST.json"
    manifest = json.loads((root / name).read_text(encoding="utf-8"))

    def verify():
        for p, digest in manifest["files"].items():
            if hashlib.sha256((root / p).read_bytes()).hexdigest() != digest:
                raise ValueError("Execution source changed: " + p)
        return len(manifest["files"])

    verification = dict(status="started", commit=manifest["commit"], files_verified_before=verify())
    exe = str(root.parent / ".venv-dflash/bin/python")
    env = dict(os.environ, OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1", PYTHONPATH=".")
    (root / "artifacts").mkdir(exist_ok=True)
    subprocess.run([exe, "-m", "pytest", "-q", "tests/test_asd_compute.py", "tests/test_asd_utility_causal.py",
                    "tests/test_asd.py", "--basetemp", "artifacts/test-tmp", "--junitxml=artifacts/tests-server.xml"],
                   env=env, check=True)

    def worker(gpu):
        part, ids = (("math-a", ["math500:1", "math500:4"]) if gpu == 0
                     else ("mixed-b", ["math500:7", "gsm8k:21"]))
        command = [exe, "-u", "-m", "scripts.run_asd_compute_probe", "--config", "configs/runtime-dflash.json",
                   "--task-ids", *ids, "--output", "results/raw/asd-compute/" + part]
        print("START", gpu, part, json.dumps(command), flush=True)
        with (root / (part + ".log")).open("w") as log:
            process = subprocess.run(command, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                                     stdout=log, stderr=subprocess.STDOUT)
        print("FINISH", gpu, part, process.returncode, flush=True)
        if process.returncode:
            raise RuntimeError(part + " failed " + str(process.returncode))

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, [0, 1]))
        verification.update(status="complete", files_verified_after=verify())
    except Exception as error:
        verification.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        (root / "execution-verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
        with zipfile.ZipFile(root.parent / "asd-compute-results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            directory = root / "results/raw/asd-compute"
            for p in sorted(directory.rglob("*")):
                if p.is_file():
                    n = p.relative_to(directory).as_posix()
                    if n.endswith(".jsonl"):
                        archive.writestr(n + ".gz", gzip.compress(p.read_bytes(), mtime=0))
                    else:
                        archive.write(p, n)
            for n in (name, "execution-verification.json", "scripts/run_asd_compute_pipeline.py",
                      "math-a.log", "mixed-b.log", "artifacts/tests-server.xml"):
                if (root / n).exists():
                    archive.write(root / n, n.replace("artifacts/", ""))
    print("PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
