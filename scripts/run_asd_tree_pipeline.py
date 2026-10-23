"""Two owned A6000 workers, immutable source verification and bounded archives."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import zipfile

from memorydraft.evaluation.asd_tree import STUDY
from scripts.run_rmsnorm_runtime_pipeline import run_owned


def main(*, study=STUDY, worker_module="scripts.run_asd_tree", report_module="scripts.write_asd_tree_report",
         manifest_name="ASD_TREE_SOURCE_MANIFEST.json", extra_tests=(),worker_timeout=7200,post_worker_module=None):
    root = Path.cwd()
    name = manifest_name
    manifest = json.loads((root/name).read_text())
    out = root/f"results/raw/{study}"
    out.mkdir(parents=True,exist_ok=False)
    (root/"artifacts").mkdir(exist_ok=True)
    def verify():
        for n,h in manifest["files"].items():
            if hashlib.sha256((root/n).read_bytes()).hexdigest() != h:
                raise ValueError("Frozen tree source changed: "+n)
        return len(manifest["files"])
    info = dict(status="started",commit=manifest["commit"],files_verified_before=verify(),worker_timeout_seconds=worker_timeout)
    exe = str(root.parent/".venv-dflash/bin/python")
    env = dict(os.environ,HF_HUB_OFFLINE="1",PYTHONPATH=".",OMP_NUM_THREADS="4",TORCHINDUCTOR_COMPILE_THREADS="2")
    try:
        run_owned([exe,"-m","pytest","-q","tests/test_budget_tree.py","tests/test_draft_tree.py",
            "tests/test_dflash_reuse.py","tests/test_asd_cpu.py","tests/test_asd_tree_report.py",*extra_tests,
            "--basetemp","artifacts/test-tmp","--junitxml="+str(out/"tests-server.xml")],
            env,out/"tests.log",timeout=300)
        def worker(shard):
            cache = dict(TORCHINDUCTOR_CACHE_DIR=str(root/f"artifacts/inductor-{shard}"),
                         TRITON_CACHE_DIR=str(root/f"artifacts/triton-{shard}"))
            for directory in cache.values():Path(directory).mkdir(exist_ok=False)
            run_owned([exe,"-u","-m",worker_module,"--shard",str(shard)],
                dict(env,**cache,CUDA_VISIBLE_DEVICES=str(shard)),out/f"part-{shard}.log",timeout=worker_timeout)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,(0,1)))
        if post_worker_module:
            cache=dict(TORCHINDUCTOR_CACHE_DIR=str(root/'artifacts/inductor-post'),
                       TRITON_CACHE_DIR=str(root/'artifacts/triton-post'))
            for directory in cache.values():Path(directory).mkdir(exist_ok=False)
            run_owned([exe,'-u','-m',post_worker_module],dict(env,**cache,CUDA_VISIBLE_DEVICES='0'),
                      out/'post-worker.log',timeout=3600)
        run_owned([exe,"-m",report_module,"--input",str(out)],env,out/"report.log",timeout=300)
        info.update(status="complete",files_verified_after=verify())
    except Exception as error:
        info.update(status="failed",error=type(error).__name__+": "+str(error))
        raise
    finally:
        (out/"execution-verification.json").write_text(json.dumps(info,indent=2)+"\n")
        with zipfile.ZipFile(root.parent/(root.name+"-results.zip"),"x",zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    if path.suffix not in {".json",".jsonl",".xml",".log",".md"}:
                        raise ValueError("Unexpected public evidence type")
                    relative = path.relative_to(out).as_posix()
                    if path.suffix == ".jsonl":archive.writestr(relative+".gz",gzip.compress(path.read_bytes(),mtime=0))
                    else:archive.write(path,relative)
            archive.write(root/name,name)
    print("ASD BUDGET TREE COMPLETE",flush=True)


if __name__ == "__main__":main()
