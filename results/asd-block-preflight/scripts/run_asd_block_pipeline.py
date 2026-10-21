"""Frozen two-worker pipeline for the real longer-draft preflight."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile


def main():
    root=Path.cwd();name="ASD_BLOCK_SOURCE_MANIFEST.json"
    manifest=json.loads((root/name).read_text(encoding="utf-8"))
    dest=root/"results/raw/asd-block-preflight";dest.mkdir(parents=True,exist_ok=False)
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
        command=[exe,"-u","-m","scripts.run_asd_block_probe","--config","configs/runtime-dflash.json",
                 "--shard",str(shard),"--output",str(dest/f"part-{shard}")]
        print("START",shard,json.dumps(command),flush=True)
        with (dest/f"part-{shard}.log").open("w") as log:
            r=subprocess.run(command,env=dict(env,CUDA_VISIBLE_DEVICES=str(shard)),stdout=log,stderr=subprocess.STDOUT)
        print("FINISH",shard,r.returncode,flush=True)
        if r.returncode:raise RuntimeError(f"Block worker {shard} failed: {r.returncode}")

    try:
        subprocess.run([exe,"-m","pytest","-q","tests/test_asd_block_probe.py","tests/test_asd.py",
                        "tests/test_dflash_reuse.py","--basetemp","artifacts/test-tmp",
                        "--junitxml=artifacts/tests-server.xml"],env=env,check=True)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,[0,1]))
        info.update(status="complete",files_verified_after=verify())
    except Exception as error:
        info.update(status="failed",error=type(error).__name__+": "+str(error));raise
    finally:
        (dest/"execution-verification.json").write_bytes((json.dumps(info,indent=2)+"\n").encode())
        with zipfile.ZipFile(root.parent/"asd-block-preflight-results.zip","w",zipfile.ZIP_DEFLATED) as z:
            for p in sorted(dest.rglob("*")):
                if p.is_file():
                    n=p.relative_to(dest).as_posix()
                    if n.endswith(".jsonl"):z.writestr(n+".gz",gzip.compress(p.read_bytes(),mtime=0))
                    else:z.write(p,n)
            for n in (name,"scripts/run_asd_block_pipeline.py","artifacts/tests-server.xml"):
                z.write(root/n,n.replace("artifacts/",""))
    print("BLOCK PREFLIGHT COMPLETE",flush=True)


if __name__=="__main__":main()
