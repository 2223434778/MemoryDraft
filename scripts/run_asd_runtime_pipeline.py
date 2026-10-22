"""Two isolated runtime profiles, bounded HTTP acceptance and fresh CLI processes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import zipfile

from scripts.run_rmsnorm_runtime_pipeline import run_owned
from scripts.validate_asd_runtime import ADAPTER_SHA, TASK_IDS, check_result, references


def check_results(root, output, cache_bytes, *, cli_spent_atol=0.):
    checks = {}
    for variant,name in (("base","base"),("ordinary-position","position")):
        directory = output/name
        info = json.loads((directory/"acceptance.json").read_text(encoding="utf-8"))
        cli = json.loads((directory/"cli.json").read_text(encoding="utf-8"))
        rows = json.loads((directory/"rows.json").read_text(encoding="utf-8"))
        expected_sha = ADAPTER_SHA if name == "position" else None
        compiler = info["compiler"]
        if (info["status"] != "complete" or info["variant"] != variant or info["rows"] != 16 or info["ar_references"] != 4
                or not info["cancellation_recovered"] or not info["profile_restored_after_close"]
                or not info["strict_benchmark_guard"] or not info["strict_benchmark"]["methods"]["DFlash"]["strict_correct"]
                or not info["http"]["disconnect_cancelled"] or not info["http"]["recovered"] or info["http"]["load"]["failed"]
                or info["http"]["metrics"]["failed"] or info["formal_new_graphs"] != 0):
            raise ValueError("Incomplete ASD runtime acceptance: "+name)
        if (compiler["graph_count"] > 8 or compiler["backend_seconds"] > 300 or compiler["allow_tf32"]
                or compiler["float32_matmul_precision"] != "highest" or cache_bytes[name] > 4*2**30
                or not 0 < info["peak_gpu_bytes"] <= 46*2**30):
            raise ValueError("ASD runtime resource/precision budget failed")
        old = references(variant)
        expected = {(task,stage,method) for task in TASK_IDS for stage in ("natural","prefix") for method in ("DFlash","ASD")}
        if len(rows) != 16 or {(r["request_id"],r["stage"],r["method"]) for r in rows} != expected:
            raise ValueError("Missing or duplicated direct runtime results")
        for row in rows:
            check_result(row,old[row["request_id"],row["stage"],row["method"]],adapter_sha=expected_sha)
        check_result(cli,old["gsm8k:1196","natural","ASD"],adapter_sha=expected_sha,spent_atol=cli_spent_atol)
        if not cli["approximate"] or cli["counters"]["compile_graphs_before"] != 0 or not 0 < cli["counters"]["compile_graphs_after"] <= 8:
            raise ValueError("Fresh CLI did not exercise cold approximate execution")
        checks[name] = dict(status="complete",research_matching_rows=len(rows),fresh_cli_matches=True,
                            direct_cancel_recovers=True,http_disconnect_recovers=True,http_load_successful=info["http"]["load"]["successful"],
                            formal_new_graphs=0,graphs=compiler["graph_count"],cache_bytes=cache_bytes[name],adapter_sha256=expected_sha)
    return checks


def main():
    root = Path.cwd()
    manifest_name = "ASD_RUNTIME_SOURCE_MANIFEST.json"
    manifest = json.loads((root/manifest_name).read_text(encoding="utf-8"))
    dest = root/"results/raw/asd-runtime"
    dest.mkdir(parents=True,exist_ok=False)
    (root/"artifacts").mkdir(exist_ok=True)
    exe = str(root.parent/".venv-dflash/bin/python")
    env = dict(os.environ,PYTHONPATH=".",OMP_NUM_THREADS="4",HF_HUB_OFFLINE="1",TORCHINDUCTOR_COMPILE_THREADS="2")
    private = root.parent/"asd-adaptation-followup/artifacts/followup-adapter/selected.safetensors"
    def verify():
        for n,h in manifest["files"].items():
            if hashlib.sha256((root/n).read_bytes()).hexdigest() != h:
                raise ValueError("Frozen runtime source changed: "+n)
        return len(manifest["files"])
    def private_sha():return hashlib.sha256(private.read_bytes()).hexdigest()
    info = dict(status="started",commit=manifest["commit"],files_verified_before=verify(),phases=[])
    caches = {}
    for name in ("base","position"):
        for stage in ("worker","cli"):
            cache = {"TORCHINDUCTOR_CACHE_DIR":str(root/f"artifacts/inductor-{name}-{stage}"),
                     "TRITON_CACHE_DIR":str(root/f"artifacts/triton-{name}-{stage}")}
            for directory in cache.values():Path(directory).mkdir(exist_ok=False)
            caches[name,stage] = cache
    configs = dict(base="configs/runtime-dflash-asd.json",position="configs/runtime-dflash-asd-position.json")
    try:
        if private_sha() != ADAPTER_SHA:raise ValueError("Original accepted adapter changed")
        run_owned([exe,"-m","scripts.prepare_asd_runtime_adapter","--source",str(private)],env,dest/"adapter-prepare.log",timeout=120)
        run_owned([exe,"-m","pytest","-q","tests/test_runtime_asd.py","tests/test_asd_runtime_acceptance.py","tests/test_runtime.py",
                   "tests/test_serving.py","tests/test_runtime_rmsnorm.py","tests/test_asd.py","--basetemp","artifacts/test-tmp",
                   "--junitxml="+str(dest/"tests-server.xml")],env,dest/"tests.log",timeout=300)
        info["phases"].append("cpu")
        def worker(index):
            name = ("base","position")[index]
            run_owned([exe,"-u","-m","scripts.validate_asd_runtime","--config",configs[name],"--output",str(dest/name)],
                      dict(env,**caches[name,"worker"],CUDA_VISIBLE_DEVICES=str(index)),dest/f"worker-{name}.log",timeout=1200)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,[0,1]))
        info["phases"].append("acceptance")
        def cli(index):
            name = ("base","position")[index]
            run_owned([exe,"-u","-m","memorydraft","infer","--config",configs[name],"--request","examples/request-asd.json",
                       "--method","ASD","--output",str(dest/name/"cli.json")],
                      dict(env,**caches[name,"cli"],CUDA_VISIBLE_DEVICES=str(index)),dest/f"cli-{name}.log",timeout=300)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(cli,[0,1]))
        info["phases"].append("cli")
        cache_bytes = {name:sum(p.stat().st_size for stage in ("worker","cli") for directory in caches[name,stage].values()
                               for p in Path(directory).rglob("*") if p.is_file()) for name in configs}
        # Only the ledger's FP32 sum may round differently on the cold CLI graph.
        checks = check_results(root,dest,cache_bytes,cli_spent_atol=1e-5)
        if private_sha() != ADAPTER_SHA:raise ValueError("Original accepted adapter changed during acceptance")
        info.update(status="complete",files_verified_after=verify(),original_adapter_unchanged=True,checks=checks)
    except Exception as error:
        info.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        archive_path = root.parent/(root.name+"-results.zip")
        info["archive_path"] = str(archive_path)
        (dest/"execution-verification.json").write_bytes((json.dumps(info,indent=2)+"\n").encode())
        with zipfile.ZipFile(archive_path,"x",zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(dest.rglob("*")):
                if path.is_file():
                    if path.suffix not in {".json",".jsonl",".xml",".log"}:raise ValueError("Unexpected runtime evidence type")
                    archive.write(path,path.relative_to(dest).as_posix())
            archive.write(root/manifest_name,manifest_name)
    print("ASD RUNTIME ACCEPTANCE COMPLETE",flush=True)


if __name__ == "__main__":
    main()
