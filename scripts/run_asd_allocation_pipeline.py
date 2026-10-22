"""Bounded two-card finite-action check with canonical source verification."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import zipfile

from scripts.run_rmsnorm_runtime_pipeline import run_owned


def main():
    root = Path.cwd()
    manifest_name = "ASD_ALLOCATION_SOURCE_MANIFEST.json"
    manifest = json.loads((root/manifest_name).read_text())
    dest = root/"results/raw/asd-allocation"
    dest.mkdir(parents=True, exist_ok=False)
    (root/"artifacts").mkdir(exist_ok=True)
    exe = str(root.parent/".venv-dflash/bin/python")
    env = dict(os.environ, PYTHONPATH=".", OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1")
    def verify():
        for n, h in manifest["files"].items():
            if hashlib.sha256((root/n).read_bytes()).hexdigest() != h:
                raise ValueError("Execution source changed: "+n)
        return len(manifest["files"])
    info = dict(status="started", commit=manifest["commit"], files_verified_before=verify(), worker_timeout_seconds=2400)
    def worker(i):
        cache = {"TORCHINDUCTOR_CACHE_DIR": str(root/f"artifacts/inductor-{i}"), "TRITON_CACHE_DIR": str(root/f"artifacts/triton-{i}")}
        for directory in cache.values():
            Path(directory).mkdir(parents=True, exist_ok=False)
        run_owned([exe, "-u", "-m", "scripts.run_asd_allocation_probe", "--config", "configs/runtime-dflash-rmsnorm.json",
                   "--shard", str(i), "--output", str(dest/f"part-{i}")],
                  dict(env, **cache, CUDA_VISIBLE_DEVICES=str(i), TORCHINDUCTOR_COMPILE_THREADS="2"),
                  dest/f"part-{i}.log", timeout=2400)
    try:
        run_owned([exe, "-m", "pytest", "-q", "tests/test_allocation.py", "tests/test_dflash_reuse.py",
                   "tests/test_runtime_rmsnorm.py", "tests/test_asd_allocation_report.py", "--basetemp", "artifacts/test-tmp",
                   "--junitxml="+str(dest/"tests-server.xml")], env, dest/"tests.log", timeout=300)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, [0, 1]))
        info.update(status="complete", files_verified_after=verify())
    except Exception as error:
        info.update(status="failed", error=type(error).__name__+": "+str(error))
        raise
    finally:
        (dest/"execution-verification.json").write_text(json.dumps(info, indent=2)+"\n")
        with zipfile.ZipFile(root.parent/"asd-allocation-results.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(dest.rglob("*")):
                if path.is_file():
                    name = path.relative_to(dest).as_posix()
                    if path.suffix == ".jsonl":
                        z.writestr(name+".gz", gzip.compress(path.read_bytes(), mtime=0))
                    else:
                        z.write(path, name)
            z.write(root/manifest_name, manifest_name)
    print("ASD ALLOCATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
