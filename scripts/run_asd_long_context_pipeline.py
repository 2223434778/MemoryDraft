"""Fresh deployment, resource gate, counterbalanced GPU assignment; no training."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import zipfile

from memorydraft.evaluation.long_context import LEVELS, STUDY
from scripts.run_rmsnorm_runtime_pipeline import run_owned
from scripts.validate_asd_runtime import ADAPTER_SHA


def main():
    root = Path.cwd()
    manifest_name = "ASD_LONG_CONTEXT_SOURCE_MANIFEST.json"
    manifest = json.loads((root/manifest_name).read_text(encoding="utf-8"))
    output = root/f"results/raw/{STUDY}"
    output.mkdir(parents=True,exist_ok=False)
    exe = str(root.parent/".venv-dflash/bin/python")
    env = dict(os.environ,PYTHONPATH=".",OMP_NUM_THREADS="4",HF_HUB_OFFLINE="1",TORCHINDUCTOR_COMPILE_THREADS="2")
    private = root.parent/"asd-adaptation-followup/artifacts/followup-adapter/selected.safetensors"
    def verify():
        if not all(hashlib.sha256((root/n).read_bytes()).hexdigest() == h for n,h in manifest["files"].items()):
            raise ValueError("Frozen long-context source changed")
        return len(manifest["files"])
    info = dict(status="started",commit=manifest["commit"],files_verified_before=verify())
    try:
        if hashlib.sha256(private.read_bytes()).hexdigest() != ADAPTER_SHA:raise ValueError("Original adapter changed")
        run_owned([exe,"-m","scripts.prepare_asd_runtime_adapter","--source",str(private)],env,output/"adapter-prepare.log",timeout=120)
        run_owned([exe,"-m","pytest","-q","tests/test_long_context.py","tests/test_runtime_asd.py","tests/test_runtime.py",
                   "--basetemp","artifacts/test-tmp","--junitxml="+str(output/"tests-server.xml")],env,output/"tests.log",timeout=300)
        def worker(mode,variant,shard,levels):
            cache = dict(TORCHINDUCTOR_CACHE_DIR=str(root/f"artifacts/inductor-{mode}-{variant}-{shard}"),
                         TRITON_CACHE_DIR=str(root/f"artifacts/triton-{mode}-{variant}-{shard}"))
            for directory in cache.values():Path(directory).mkdir(exist_ok=False)
            run_owned([exe,"-u","-m","scripts.run_asd_long_context","--mode",mode,"--variant",variant,"--shard",str(shard),
                       "--levels",*map(str,levels)],dict(env,**cache,CUDA_VISIBLE_DEVICES=str(shard)),
                      output/f"{mode}-{variant}-{shard}.log",timeout=2400)
        def waves(mode,levels):
            for jobs in ((("base",0),("position",1)),(("position",0),("base",1))):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(lambda job:worker(mode,*job,levels),jobs))
        waves("preflight",LEVELS)
        checks = [json.loads((output/f"preflight-{variant}-{shard}/provenance.json").read_text())
                  for variant in ("base","position") for shard in (0,1)]
        admitted = [level for level in LEVELS if all(level in r["admitted"] for r in checks)]
        write = lambda name,value:(output/name).write_bytes((json.dumps(value,indent=2)+"\n").encode())
        write("admission.json",dict(levels=admitted,excluded=[v for v in LEVELS if v not in admitted],
                                   rule="Intersection across both cards/configs; resource failures retained; no outcome-based exclusion"))
        if not admitted:raise ValueError("No input length met resource budget")
        waves("eval",admitted)
        caches = sum(p.stat().st_size for p in (root/"artifacts").rglob("*") if p.is_file() and any(
                     part.startswith(("inductor-","triton-")) for part in p.relative_to(root/"artifacts").parts))
        if caches > 8*4*2**30:raise ValueError("Compiler cache budget exceeded")
        if hashlib.sha256(private.read_bytes()).hexdigest() != ADAPTER_SHA:raise ValueError("Original adapter changed")
        info.update(status="complete",files_verified_after=verify(),original_adapter_unchanged=True,admitted=admitted,cache_bytes=caches)
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
            archive.write(root/manifest_name,manifest_name)
    print("ASD LONG CONTEXT COMPLETE",flush=True)


if __name__ == "__main__":
    main()
