"""Bounded two-GPU run with owned process groups and isolated compile caches."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import zipfile


def main(*, study="compile", extra_tests=()):
    root = Path.cwd()
    name = "ASD_"+study.upper()+"_SOURCE_MANIFEST.json"
    manifest = json.loads((root/name).read_text(encoding="utf-8"))
    dest = root/"results/raw"/("asd-"+study)
    dest.mkdir(parents=True, exist_ok=False)
    (root/"artifacts").mkdir(exist_ok=True)

    def verify():
        for path, digest in manifest["files"].items():
            if hashlib.sha256((root/path).read_bytes()).hexdigest() != digest:
                raise ValueError("Execution source changed: "+path)
        return len(manifest["files"])

    info = dict(status="started", commit=manifest["commit"], files_verified_before=verify(), worker_timeout_seconds=1800)
    exe = str(root.parent/".venv-dflash/bin/python")
    env = dict(os.environ, OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1", PYTHONPATH=".")

    def worker(shard):
        cache = {"TORCHINDUCTOR_CACHE_DIR": str(root/f"artifacts/inductor-{shard}"),
                 "TRITON_CACHE_DIR": str(root/f"artifacts/triton-{shard}")}
        for folder in cache.values():
            Path(folder).mkdir(parents=True, exist_ok=False)
        command = [exe, "-u", "-m", "scripts.run_asd_"+study+"_probe", "--config", "configs/runtime-dflash.json",
                   "--shard", str(shard), "--output", str(dest/f"part-{shard}")]
        print("START", shard, json.dumps(dict(argv=command, cache=cache)), flush=True)
        with (dest/f"part-{shard}.log").open("w") as log:
            process = subprocess.Popen(command, env=dict(env, **cache, CUDA_VISIBLE_DEVICES=str(shard),
                                       TORCHINDUCTOR_COMPILE_THREADS="2"), stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                # Only terminate the new session created for this worker.
                if process.poll() is None:
                    assert os.getpgid(process.pid) == process.pid
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise RuntimeError(f"Compiler worker {shard} exceeded 1800 seconds")
        print("FINISH", shard, code, flush=True)
        if code:
            raise RuntimeError(f"Compiler worker {shard} failed: {code}")

    try:
        subprocess.run([exe, "-m", "pytest", "-q", "tests/test_regional_compile.py", "tests/test_asd_compile.py",
                        "tests/test_asd_cpu.py", "tests/test_dflash_reuse.py", *extra_tests, "--basetemp", "artifacts/test-tmp",
                        "--junitxml=artifacts/tests-server.xml"], env=env, check=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, [0, 1]))
        info.update(status="complete", files_verified_after=verify())
    except Exception as error:
        info.update(status="failed", error=type(error).__name__+": "+str(error))
        raise
    finally:
        (dest/"execution-verification.json").write_bytes((json.dumps(info, indent=2)+"\n").encode())
        with zipfile.ZipFile(root.parent/("asd-"+study+"-results.zip"), "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(dest.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(dest).as_posix()
                    if relative.endswith(".jsonl"):
                        archive.writestr(relative+".gz", gzip.compress(path.read_bytes(), mtime=0))
                    else:
                        archive.write(path, relative)
            for path in (name, "scripts/run_asd_compile_pipeline.py", "artifacts/tests-server.xml"):
                archive.write(root/path, path.replace("artifacts/", ""))
    print("REGIONAL "+study.upper()+" COMPLETE", flush=True)


if __name__ == "__main__":
    main()
