"""Execute the frozen two-task diagnostic and export compact public evidence."""
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
    manifest_name = "ASD_UTILITY_CAUSAL_SOURCE_MANIFEST.json"
    manifest = json.loads((root / manifest_name).read_text())

    def verify():
        for name, digest in manifest["files"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Execution source changed: " + name)
        return len(manifest["files"])

    count = verify()
    verification = dict(status="started", commit=manifest["commit"], files_verified_before=count)
    (root / "artifacts").mkdir(exist_ok=True)
    exe = str(root.parent / ".venv-dflash/bin/python")
    env = dict(os.environ, OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1", PYTHONPATH=".")
    test_command = [exe, "-m", "pytest", "-q", "tests/test_asd_utility_causal.py", "tests/test_asd_confidence.py",
                    "tests/test_asd.py", "--basetemp", "artifacts/test-tmp", "--junitxml=artifacts/tests-server.xml"]
    subprocess.run(test_command, env=env, check=True)

    def worker(gpu):
        part, task = (("gsm", "gsm8k:21") if gpu == 0 else ("math", "math500:7"))
        command = [exe, "-u", "-m", "scripts.run_asd_utility_causal_probe", "--config", "configs/runtime-dflash.json",
                   "--task-id", task, "--output", "results/raw/asd-utility-causal/" + part]
        print("START", gpu, part, json.dumps(command), flush=True)
        with (root / (part + ".log")).open("w") as log:
            result = subprocess.run(command, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                                    stdout=log, stderr=subprocess.STDOUT)
        print("FINISH", gpu, part, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(part + " failed " + str(result.returncode))

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, [0, 1]))
        verification.update(status="complete", files_verified_after=verify())
    except Exception as error:
        verification.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        (root / "execution-verification.json").write_text(json.dumps(verification, indent=2))
        with zipfile.ZipFile(root.parent / "asd-utility-causal-results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            directory = root / "results/raw/asd-utility-causal"
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    name = path.relative_to(directory).as_posix()
                    if name.endswith(".jsonl"):
                        archive.writestr(name + ".gz", gzip.compress(path.read_bytes(), mtime=0))
                    else:
                        archive.write(path, name)
            for name in (manifest_name, "execution-verification.json", "scripts/run_asd_utility_causal_pipeline.py",
                         "gsm.log", "math.log", "artifacts/tests-server.xml"):
                if (root / name).exists():
                    archive.write(root / name, name.replace("artifacts/", ""))
    print("PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
