"""Collect, equal-budget train, then evaluate two adapters in an isolated root."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import zipfile

from memorydraft.evaluation.adaptation import ARMS
from scripts.run_rmsnorm_runtime_pipeline import run_owned


def main():
    root = Path.cwd()
    manifest_name = "ASD_ADAPTATION_SOURCE_MANIFEST.json"
    manifest = json.loads((root/manifest_name).read_text(encoding="utf-8"))
    dest = root/"results/raw/asd-adaptation"
    dest.mkdir(parents=True, exist_ok=False)
    (root/"artifacts").mkdir(exist_ok=True)
    exe = str(root.parent/".venv-dflash/bin/python")
    env = dict(os.environ, PYTHONPATH=".", OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1")
    def verify():
        for name, h in manifest["files"].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != h:
                raise ValueError("Frozen adaptation source changed: "+name)
        return len(manifest["files"])
    info = dict(status="started", commit=manifest["commit"], files_verified_before=verify(),
                phases=[], worker_timeouts=dict(collect=1800, train=1200, evaluate=3600))
    def parallel(module, phase, timeout):
        def worker(i):
            extra = ["--arm", ARMS[i]] if phase == "train" else ["--shard", str(i)]
            cache = {}
            if phase == "evaluate":
                cache = {"TORCHINDUCTOR_CACHE_DIR": str(root/f"artifacts/inductor-{i}"), "TRITON_CACHE_DIR": str(root/f"artifacts/triton-{i}")}
                for directory in cache.values():
                    Path(directory).mkdir(exist_ok=False)
            run_owned([exe, "-u", "-m", module, *extra],
                      dict(env, **cache, CUDA_VISIBLE_DEVICES=str(i), TORCHINDUCTOR_COMPILE_THREADS="2"),
                      dest/f"{phase}-{i}.log", timeout=timeout)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, [0, 1]))
        info["phases"].append(phase)
    try:
        run_owned([exe, "-m", "pytest", "-q", "tests/test_adaptation.py", "tests/test_asd_adaptation_report.py",
                   "tests/test_dflash_reuse.py", "tests/test_runtime_rmsnorm.py", "--basetemp", "artifacts/test-tmp",
                   "--junitxml="+str(dest/"tests-server.xml")], env, dest/"tests.log", timeout=300)
        parallel("scripts.run_asd_adaptation_collect", "collect", 1800)
        caches = sum(p.stat().st_size for p in (root/"artifacts").glob("adaptation-features-*/*.pt"))
        if caches > 16*2**30:
            raise ValueError("Aggregate feature cache exceeds 16GiB")
        info["feature_cache_bytes"] = caches
        parallel("scripts.run_asd_adaptation_train", "train", 1200)
        parallel("scripts.run_asd_adaptation_eval", "evaluate", 3600)
        info.update(status="complete", files_verified_after=verify())
    except Exception as error:
        info.update(status="failed", error=type(error).__name__+": "+str(error))
        raise
    finally:
        (dest/"execution-verification.json").write_text(json.dumps(info, indent=2)+"\n", encoding="utf-8")
        with zipfile.ZipFile(root.parent/"asd-adaptation-results.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(dest.rglob("*")):
                if path.is_file():
                    if path.suffix not in {".json", ".jsonl", ".xml", ".log"}:
                        raise ValueError("Unexpected evidence type; never archive weights/features")
                    name = path.relative_to(dest).as_posix()
                    if path.suffix == ".jsonl":
                        z.writestr(name+".gz", gzip.compress(path.read_bytes(), mtime=0))
                    else:
                        z.write(path, name)
            z.write(root/manifest_name, manifest_name)
    print("ASD ADAPTATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
