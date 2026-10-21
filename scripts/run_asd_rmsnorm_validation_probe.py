"""Frozen RMSNorm candidate on validation; reserve is not generated."""
import gzip
import hashlib
import json
from pathlib import Path

from memorydraft.evaluation.asd_suite import BENCHMARKS, load_suite
from scripts import run_asd_rmsnorm_suite_probe as common

ROOT = Path(__file__).resolve().parents[1]
WORKER_TIMEOUT = 7200
CANDIDATE_COMMIT = "e958532bb4dc7544692011115895f90df2bd3f43"
CANDIDATE_FILES = {
    "memorydraft/backends/regional_compile.py": "a96509554f3891ee625d618aa1a4ee51927ae3595876379286a8c09636dbbd5e",
    "memorydraft/backends/asd_cpu.py": "588edb3fd70b519391cff05fa091681cdb66780dd02524d6e702d8405d0f33c8",
    "memorydraft/backends/asd.py": "bd307a1fb0cb2307e6fa2fbccf4cb39030b66754704b5d6ec50973c2feb59f84",
    "memorydraft/backends/dflash_reuse.py": "f9989b6239ff3fdb4afc5e939feea5ad33c95e262308c23050c06b04e26f4891",
    "memorydraft/runtime/engine.py": "adafb9191bdfb8070e4d56daaf5d9861ba590663e00e757cae4ffe239ce33f2e",
    "memorydraft/runtime/config.py": "bbcf05d7430d9c01cf263294a7c94ece2627e7e57e7df5e093897d7f8ea70800",
    "configs/runtime-dflash.json": "13444cd758dc8793fc92e42c1b5331affd31f5666bfed42c683df23f94754f65",
    "scripts/run_asd_suite.py": "8d854524078f2736f003b8eb3dd5abe54ad87cb92fef4b0f91af5e33fe4fc9ba",
    "scripts/run_asd_budget_probe.py": "fbfd0b1005a5b8cc0ebaf608ec6a3f026bb35e9b09337b94ad9abaa5e05766d8",
    "memorydraft/evaluation/asd_suite.py": "1bf7a0d3fad234e5361d9fcde7c3cc79148b315755315eb0e16aab93fcfa991b",
    "data/research/asd-suite/manifest.json": "957b67af7223aee63e360ad0f8b6eb195c1143726fb724c1c90837e4fce4c136",
}


def records_and_positions():
    records = load_suite(ROOT/"data/research/asd-suite", "validation")
    positions = {r["id"]: i for b in BENCHMARKS for i, r in enumerate(r for r in records if r["benchmark"] == b)}
    return records, positions


def candidate_check(root, files=CANDIDATE_FILES):
    for name, digest in files.items():
        if hashlib.sha256((root/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != digest:
            raise ValueError("Frozen execution candidate changed: "+name)
    return dict(commit=CANDIDATE_COMMIT, files=files)


def exposure_check(root, protected_ids):
    """Inspect repository result payloads, excluding only this run's live output."""
    checked, payloads, found = 0, 0, []
    def walk(node, path, inherited=None):
        nonlocal payloads
        if isinstance(node, list):
            for item in node:
                walk(item, path, inherited)
        elif isinstance(node, dict):
            identity = next((node[k] for k in ("request_id", "id", "task_id") if k in node), inherited)
            if isinstance(identity, (str, int)) and node.get("benchmark") in BENCHMARKS and ":" not in str(identity):
                identity = node["benchmark"]+":"+str(identity)
            generated = any(isinstance(node.get(k), list) and node[k] for k in
                            ("token_ids", "generated_token_ids", "output_ids", "generated_ids"))
            if generated:
                payloads += 1
                if isinstance(identity, str) and identity in protected_ids:
                    found.append(dict(path=path, request_id=identity))
            for item in node.values():
                if isinstance(item, (dict, list)):
                    walk(item, path, identity)
    for path in sorted((root/"results").rglob("*")):
        relative = path.relative_to(root/"results")
        if relative.parts[:2] == ("raw", "asd-rmsnorm-validation"):
            continue
        if not path.is_file() or not (path.name.endswith(".json") or path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz")):
            continue
        checked += 1
        text = gzip.decompress(path.read_bytes()).decode("utf-8-sig") if path.name.endswith(".gz") else path.read_text(encoding="utf-8-sig")
        if path.name.endswith((".jsonl", ".jsonl.gz")):
            for line in text.splitlines():
                if line.strip():
                    walk(json.loads(line), relative.as_posix())
        else:
            walk(json.loads(text), relative.as_posix())
    if found:
        raise ValueError("Validation/reserve generation already present: "+json.dumps(found[:10]))
    return dict(result_files_scanned=checked, generation_payloads_checked=payloads, protected_generation_matches=0,
                scope="Repository result JSON/JSONL token payloads; current validation raw directory excluded")


def isolation_check(root):
    root = Path(root)
    candidate = candidate_check(root)
    development, validation, reserve = ({r["id"] for r in load_suite(root/"data/research/asd-suite", split)}
                                        for split in ("development", "validation", "reserve"))
    if development & validation or validation & reserve or development & reserve:
        raise ValueError("Suite split overlap")
    return dict(candidate=candidate, validation_ids_sha256=hashlib.sha256(json.dumps(sorted(validation)).encode()).hexdigest(),
                reserve_ids_sha256=hashlib.sha256(json.dumps(sorted(reserve)).encode()).hexdigest(),
                **exposure_check(root, validation | reserve))


if __name__ == "__main__":
    common.main(record_loader=records_and_positions, prior_loader=None, split="validation",
                protocol="docs/ASD_RMSNORM_VALIDATION_PROTOCOL.md", worker_timeout=WORKER_TIMEOUT,
                entrypoint=__file__, precheck=isolation_check)
