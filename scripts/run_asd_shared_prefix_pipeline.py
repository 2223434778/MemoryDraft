"""Counterbalanced same-card shared-prefix and fresh whole-prefix HTTP control."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import zipfile

from scripts.run_rmsnorm_runtime_pipeline import run_owned

STUDY = "asd-shared-prefix"


def main():
    root = Path.cwd()
    name = "ASD_SHARED_PREFIX_SOURCE_MANIFEST.json"
    manifest = json.loads((root/name).read_text(encoding="utf-8"))
    output = root/f"results/raw/{STUDY}"
    output.mkdir(parents=True,exist_ok=False)
    def verify():
        for n,h in manifest["files"].items():
            if hashlib.sha256((root/n).read_bytes()).hexdigest() != h:raise ValueError("Frozen source changed: "+n)
        return len(manifest["files"])
    info = dict(status="started",commit=manifest["commit"],files_verified_before=verify())
    exe = str(root.parent/".venv-dflash/bin/python")
    env = dict(os.environ,HF_HUB_OFFLINE="1",PYTHONPATH=".",OMP_NUM_THREADS="4",TORCHINDUCTOR_COMPILE_THREADS="2")
    try:
        run_owned([exe,"-m","pytest","-q","tests/test_shared_prefix.py","tests/test_shared_prefix_report.py","tests/test_prefix_cache.py",
            "tests/test_prefix_cache_study.py","tests/test_runtime.py","tests/test_runtime_asd.py","tests/test_serving.py",
            "--basetemp","artifacts/tests-tmp","--junitxml="+str(output/"tests-server.xml")],env,output/"tests.log",timeout=300)
        def worker(variant,shard):
            cache = dict(TORCHINDUCTOR_CACHE_DIR=str(root/f"artifacts/inductor-{variant}-{shard}"),
                         TRITON_CACHE_DIR=str(root/f"artifacts/triton-{variant}-{shard}"))
            for directory in cache.values():Path(directory).mkdir(exist_ok=False)
            config = "configs/runtime-dflash-asd-shared-prefix.json" if variant == "shared" else "configs/runtime-dflash-asd-prefix-cache.json"
            mode = ["--mode","eval"] if variant == "shared" else ["--mode","http-only","--load-states","cache"]
            run_owned([exe,"-u","-m","scripts.run_asd_prefix_cache","--shard",str(shard),"--config",config,
                "--levels","4096","--study",STUDY,"--variant",variant,*mode],dict(env,**cache,CUDA_VISIBLE_DEVICES=str(shard)),
                output/f"eval-{variant}-{shard}.log",timeout=2400)
        for jobs in ((("shared",0),("whole",1)),(("whole",0),("shared",1))):
            with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda job:worker(*job),jobs))
        caches = sum(p.stat().st_size for p in (root/"artifacts").rglob("*") if p.is_file())
        if caches > 16*2**30:raise ValueError("Compiler cache disk budget exceeded")
        info.update(status="complete",files_verified_after=verify(),cache_bytes=caches)
    except Exception as error:
        info.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        (output/"execution-verification.json").write_bytes((json.dumps(info,indent=2)+"\n").encode())
        with zipfile.ZipFile(root.parent/(root.name+"-results.zip"),"x",zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    if path.suffix not in {".json",".jsonl",".xml",".log"}:raise ValueError("Unexpected evidence type")
                    archive.write(path,path.relative_to(output).as_posix())
            archive.write(root/name,name)
    print("ASD SHARED PREFIX COMPLETE",flush=True)


if __name__ == "__main__":
    main()
