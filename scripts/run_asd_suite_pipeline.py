"""Deploy-frozen stage-1 baselines, one independent worker per physical GPU."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

from memorydraft.evaluation.asd_suite import BENCHMARKS


def main():
    root=Path.cwd(); name="ASD_SUITE_SOURCE_MANIFEST.json"
    manifest=json.loads((root/name).read_text(encoding="utf-8"))
    destination=root/"results/raw/asd-suite-stage1"; destination.mkdir(parents=True,exist_ok=False)
    (root/"artifacts").mkdir(exist_ok=True)

    def verify():
        for p,digest in manifest["files"].items():
            if hashlib.sha256((root/p).read_bytes()).hexdigest()!=digest:
                raise ValueError("Execution file changed: "+p)
        return len(manifest["files"])

    info=dict(status="started",commit=manifest["commit"],files_verified_before=verify())
    exe=str(root.parent/".venv-dflash/bin/python")
    env=dict(os.environ,OMP_NUM_THREADS="4",HF_HUB_OFFLINE="1",PYTHONPATH=".")

    def worker(shard):
        command=[exe,"-u","-m","scripts.run_asd_suite","--config","configs/runtime-dflash.json",
                 "--benchmarks",*BENCHMARKS,"--shard",str(shard),"--output",str(destination/f"part-{shard}")]
        print("START",shard,json.dumps(command),flush=True)
        with (destination/f"part-{shard}.log").open("w") as log:
            result=subprocess.run(command,env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),stdout=log,stderr=subprocess.STDOUT)
        print("FINISH",shard,result.returncode,flush=True)
        if result.returncode:
            raise RuntimeError(f"Worker {shard} failed: {result.returncode}")

    try:
        subprocess.run([exe,"-m","pytest","-q","tests/test_asd_suite.py","tests/test_asd_suite_report.py",
                        "tests/test_asd.py","tests/test_dflash_reuse.py","--basetemp","artifacts/test-tmp",
                        "--junitxml=artifacts/tests-server.xml"],env=env,check=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker,[0,1]))
        info.update(status="complete",files_verified_after=verify())
    except Exception as error:
        info.update(status="failed",error=type(error).__name__+": "+str(error)); raise
    finally:
        (destination/"execution-verification.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
        with zipfile.ZipFile(root.parent/"asd-suite-stage1-results.zip","w",zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(destination.rglob("*")):
                if p.is_file():
                    n=p.relative_to(destination).as_posix()
                    if n.endswith(".jsonl"):
                        archive.writestr(n+".gz",gzip.compress(p.read_bytes(),mtime=0))
                    else:
                        archive.write(p,n)
            for n in (name,"scripts/run_asd_suite_pipeline.py","artifacts/tests-server.xml"):
                archive.write(root/n,n.replace("artifacts/",""))
    print("PIPELINE COMPLETE",flush=True)


if __name__=="__main__":
    main()
