"""Bounded single-GPU runtime acceptance; run only after validation admission."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import zipfile


def run_owned(argv, env, log, *, timeout=1800):
    """Reap only the fresh process session owned by this runner."""
    with log.open("w") as stream:
        process = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                assert os.getpgid(process.pid) == process.pid
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise RuntimeError("Runtime acceptance process exceeded its time budget")
    if code:
        raise RuntimeError(f"Runtime acceptance process failed ({code}); see {log.name}")


def check_results(output, cache_bytes):
    acceptance = json.loads((output/"acceptance/acceptance.json").read_text())
    cli = json.loads((output/"cli.json").read_text())
    rows = [json.loads(line) for line in (output/"acceptance/benchmark/rows.jsonl").read_text().splitlines()]
    cold = acceptance["cold_start"]
    compiler = acceptance["compiler"]
    if (acceptance["status"] != "complete" or not acceptance["history_unchanged"]
            or not acceptance["cancellation_recovered"] or acceptance["reuse_accepted_tokens"] <= 0
            or acceptance["http_load"]["failed"] or not acceptance["profiler"]["cuda_kernel_events"]):
        raise ValueError("Runtime acceptance failed")
    if (len(rows) != 12 or not all(r["correct"] for r in rows)
            or set(acceptance["http_streams"]) != {"AR", "DFlash", "R+DFlash"}
            or not all(v["correct"] and v["execution_profile"] == "rmsnorm-inductor"
                       for v in acceptance["http_streams"].values())):
        raise ValueError("Incomplete benchmark or HTTP streams")
    if not all(r["execution_profile"] == "rmsnorm-inductor" for r in [*cold, *rows, cli]):
        raise ValueError("Execution profile mismatch")
    if (len(cold) != 2 or cold[0]["counters"]["compile_graphs_before"] != 0
            or cold[0]["counters"]["compile_graphs_after"] == 0
            or any(r["counters"]["compile_graphs_before"] != r["counters"]["compile_graphs_after"] for r in rows)):
        raise ValueError("Cold compilation or warmed graph stability failed")
    if (compiler["graph_count"] > 8 or compiler["backend_seconds"] > 300 or cache_bytes > 4*2**30
            or max(r["peak_gpu_bytes"] for r in [*cold, *rows, cli]) > 46*2**30
            or compiler["float32_matmul_precision"] != "highest" or compiler["allow_tf32"]):
        raise ValueError("Runtime resource or precision bounds failed")
    if not cli["token_ids"] or cli["method"] != "DFlash":
        raise ValueError("CLI did not generate a real DFlash output")
    return dict(benchmark_rows=len(rows), all_rows_correct=True, warmed_benchmark_new_graphs=0,
                cli_tokens=len(cli["token_ids"]), graphs=compiler["graph_count"], cache_bytes=cache_bytes)


def main():
    root = Path.cwd()
    manifest_name = "RMSNORM_RUNTIME_SOURCE_MANIFEST.json"
    manifest = json.loads((root/manifest_name).read_text())
    gate = json.loads((root/"results/asd-rmsnorm-validation/analysis.json").read_text())["admission"]
    if not gate["eligible_for_explicit_profile"] or not all(gate["checks"].values()):
        raise ValueError("Independent validation has not admitted this runtime candidate")
    output = root/"results/raw/rmsnorm-runtime"
    output.mkdir(parents=True, exist_ok=False)
    cache = {"TORCHINDUCTOR_CACHE_DIR": str(root/"artifacts/inductor-runtime"),
             "TRITON_CACHE_DIR": str(root/"artifacts/triton-runtime")}
    for directory in cache.values():
        Path(directory).mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, **cache, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1",
               PYTHONPATH=".", TORCHINDUCTOR_COMPILE_THREADS="2")
    exe = str(root.parent/".venv-dflash/bin/python")

    def verify():
        for name, digest in manifest["files"].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
                raise ValueError("Execution source changed: "+name)
        return len(manifest["files"])

    info = dict(status="started", commit=manifest["commit"], files_verified_before=verify(),
                process_timeout_seconds=1800, initial_cache_files=0, validation_admission=gate)
    try:
        run_owned([exe, "-m", "pytest", "-q", "tests/test_runtime_rmsnorm.py", "tests/test_runtime.py",
                   "tests/test_serving.py", "tests/test_regional_compile.py", "--basetemp", "artifacts/test-tmp",
                   "--junitxml="+str(output/"tests-server.xml")], env, output/"tests.log", timeout=300)
        run_owned([exe, "-u", "-m", "scripts.validate_infra", "--config", "configs/runtime-dflash-rmsnorm.json",
                   "--output", str(output/"acceptance")], env, output/"acceptance.log")
        # A fresh process releases the first engine's GPU weights before CLI load.
        run_owned([exe, "-u", "-m", "memorydraft", "infer", "--config", "configs/runtime-dflash-rmsnorm.json",
                   "--request", "examples/request.json", "--method", "DFlash", "--output", str(output/"cli.json")],
                  env, output/"cli.log")
        cache_bytes = sum(p.stat().st_size for d in cache.values() for p in Path(d).rglob("*") if p.is_file())
        info.update(status="complete", checks=check_results(output, cache_bytes), files_verified_after=verify())
    except Exception as error:
        info.update(status="failed", error=type(error).__name__+": "+str(error))
        raise
    finally:
        (output/"execution-verification.json").write_text(json.dumps(info, indent=2)+"\n")
        with zipfile.ZipFile(root.parent/"rmsnorm-runtime-results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                # Raw Chrome trace, databases, weights and caches stay on the server.
                if path.is_file() and path.name != "trace.json" and path.suffix in {".json", ".jsonl", ".xml", ".log"}:
                    archive.write(path, path.relative_to(output).as_posix())
            archive.write(root/manifest_name, manifest_name)
    print("RMSNORM RUNTIME ACCEPTANCE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
