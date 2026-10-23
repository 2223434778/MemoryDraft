"""Bounded two-stage comparison; selection is frozen before any confirmation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import zipfile

from memorydraft.evaluation.asd_comparison import STUDY, records
from scripts.run_asd_rmsnorm_validation_probe import exposure_check
from scripts.run_rmsnorm_runtime_pipeline import run_owned


def main():
    root = Path.cwd()
    name = "ASD_COMPARISON_SOURCE_MANIFEST.json"
    manifest = json.loads((root/name).read_text())
    output = root/f"results/raw/{STUDY}"
    output.mkdir(parents=True,exist_ok=False)
    (root/"artifacts").mkdir(exist_ok=True)
    def verify():
        for n,h in manifest["files"].items():
            if hashlib.sha256((root/n).read_bytes()).hexdigest() != h:
                raise ValueError("Frozen comparison source changed: "+n)
        return len(manifest["files"])
    info = dict(status="started", commit=manifest["commit"], files_verified_before=verify())
    exe = str(root.parent/".venv-dflash/bin/python")
    env = dict(os.environ,HF_HUB_OFFLINE="1",PYTHONPATH=".",OMP_NUM_THREADS="4",TORCHINDUCTOR_COMPILE_THREADS="2")
    try:
        ids = {r["id"] for phase in ("screen","confirm") for r in records(root,phase)}
        info["initial_isolation"] = exposure_check(root,ids)
        run_owned([exe,"-m","pytest","-q","tests/test_asd_comparison.py","tests/test_asd.py","tests/test_asd_cpu.py",
            "tests/test_asd_confidence.py","tests/test_draft_tree.py","--basetemp","artifacts/tests-tmp",
            "--junitxml="+str(output/"tests-server.xml")],env,output/"tests.log",timeout=300)
        def worker(phase,shard):
            cache = dict(TORCHINDUCTOR_CACHE_DIR=str(root/f"artifacts/inductor-{phase}-{shard}"),
                         TRITON_CACHE_DIR=str(root/f"artifacts/triton-{phase}-{shard}"))
            for directory in cache.values():Path(directory).mkdir(exist_ok=False)
            run_owned([exe,"-u","-m","scripts.run_asd_comparison","--phase",phase,"--shard",str(shard)],
                dict(env,**cache,CUDA_VISIBLE_DEVICES=str(shard)),output/f"{phase}-{shard}.log",timeout=14400)
        for phase in ("screen","confirm"):
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda shard:worker(phase,shard),(0,1)))
            run_owned([exe,"-m","scripts.write_asd_comparison_report","--input",str(output),
                *(["--screen-only"] if phase == "screen" else [])],env,output/f"report-{phase}.log",timeout=300)
            info["completed_phase"] = phase
            (output/"execution-verification.json").write_text(json.dumps(info,indent=2)+"\n")
        info.update(status="complete",files_verified_after=verify())
    except Exception as error:
        info.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        (output/"execution-verification.json").write_text(json.dumps(info,indent=2)+"\n")
        with zipfile.ZipFile(root.parent/(root.name+"-results.zip"),"x",zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    if path.suffix not in {".json",".jsonl",".xml",".log",".md"}:
                        raise ValueError("Unexpected evidence type")
                    archive.write(path,path.relative_to(output).as_posix())
            archive.write(root/name,name)
    print("ASD COMPARISON COMPLETE",flush=True)


if __name__ == "__main__":
    main()
