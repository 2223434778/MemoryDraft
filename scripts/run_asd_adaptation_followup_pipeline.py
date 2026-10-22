"""Bounded position control and independent evaluation in a fresh deployment."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import zipfile

from memorydraft.evaluation.adaptation import digest
from memorydraft.evaluation.adaptation_followup import STUDY, ADAPTER_SHA, check_previous_publication, previous_root, previous_rows
from scripts.run_rmsnorm_runtime_pipeline import run_owned


def private_audit(root):
    previous = previous_root(root)
    manifest = json.loads((previous/"ASD_ADAPTATION_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for name,sha in manifest["files"].items():
        if digest((previous/name).read_bytes()) != sha:
            raise ValueError("Previous execution source changed: "+name)
    files = {r["feature_path"]:r["feature_sha256"] for r in previous_rows(root,"examples")}
    for arm,sha in ADAPTER_SHA.items():
        for checkpoint in ("selected","final"):
            files[f"artifacts/adaptation-adapters/{arm}/{checkpoint}.safetensors"] = sha
    for name,sha in files.items():
        if digest((previous/name).read_bytes()) != sha:
            raise ValueError("Previous private artifact changed: "+name)
    return dict(source_commit=manifest["commit"],source_files=len(manifest["files"]),files=files,
                total_bytes=sum((previous/n).stat().st_size for n in files))


def main():
    root = Path.cwd()
    manifest_name = "ASD_ADAPTATION_FOLLOWUP_SOURCE_MANIFEST.json"
    manifest = json.loads((root/manifest_name).read_text(encoding="utf-8"))
    dest = root/f"results/raw/{STUDY}"
    dest.mkdir(parents=True,exist_ok=False)
    (root/"artifacts").mkdir(exist_ok=True)
    exe = str(root.parent/".venv-dflash/bin/python")
    env = dict(os.environ,PYTHONPATH=".",OMP_NUM_THREADS="4",HF_HUB_OFFLINE="1")
    def verify():
        for name,sha in manifest["files"].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != sha:
                raise ValueError("Frozen followup source changed: "+name)
        return len(manifest["files"])
    info = dict(status="started",commit=manifest["commit"],files_verified_before=verify(),
                phases=[],previous_public_files_verified=check_previous_publication(root))
    def parallel(module,phase,timeout):
        def worker(i):
            cache = {}
            if phase == "evaluate":
                cache = {"TORCHINDUCTOR_CACHE_DIR":str(root/f"artifacts/inductor-{i}"),"TRITON_CACHE_DIR":str(root/f"artifacts/triton-{i}")}
                for directory in cache.values():
                    Path(directory).mkdir(exist_ok=False)
            run_owned([exe,"-u","-m",module,"--shard",str(i)],
                      dict(env,**cache,CUDA_VISIBLE_DEVICES=str(i),TORCHINDUCTOR_COMPILE_THREADS="2"),
                      dest/f"{phase}-{i}.log",timeout=timeout)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker,[0,1]))
        info["phases"].append(phase)
    try:
        initial_private = private_audit(root)
        (dest/"previous-private-audit.json").write_bytes((json.dumps(initial_private,indent=2)+"\n").encode())
        run_owned([exe,"-m","pytest","-q","tests/test_adaptation_followup.py","tests/test_adaptation.py",
                   "tests/test_asd_adaptation_report.py","tests/test_dflash_reuse.py","tests/test_runtime_rmsnorm.py",
                   "--basetemp","artifacts/test-tmp","--junitxml="+str(dest/"tests-server.xml")],env,dest/"tests.log",timeout=300)
        parallel("scripts.run_asd_adaptation_followup_collect","collect",1800)
        caches = sum(p.stat().st_size for p in (root/"artifacts").glob("followup-features-*/*.pt"))
        if caches > 16*2**30:
            raise ValueError("Aggregate new feature budget exceeded")
        info["feature_cache_bytes"] = caches
        run_owned([exe,"-u","-m","scripts.run_asd_adaptation_followup_train"],dict(env,CUDA_VISIBLE_DEVICES="0"),dest/"train.log",timeout=1200)
        info["phases"].append("train")
        parallel("scripts.run_asd_adaptation_followup_eval","evaluate",5400)
        if private_audit(root) != initial_private:
            raise ValueError("Previous private source/features/adapters changed")
        info.update(status="complete",files_verified_after=verify(),previous_private_assets_unchanged=True)
    except Exception as error:
        info.update(status="failed",error=type(error).__name__+": "+str(error))
        raise
    finally:
        (dest/"execution-verification.json").write_bytes((json.dumps(info,indent=2)+"\n").encode())
        with zipfile.ZipFile(root.parent/f"{STUDY}-results.zip","w",zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(dest.rglob("*")):
                if path.is_file():
                    if path.suffix not in {".json",".jsonl",".xml",".log"}:
                        raise ValueError("Unexpected evidence type; no tensors in archive")
                    name = path.relative_to(dest).as_posix()
                    if path.suffix == ".jsonl":
                        archive.writestr(name+".gz",gzip.compress(path.read_bytes(),mtime=0))
                    else:
                        archive.write(path,name)
            archive.write(root/manifest_name,manifest_name)
    print("ASD ADAPTATION FOLLOWUP COMPLETE",flush=True)


if __name__ == "__main__":
    main()
