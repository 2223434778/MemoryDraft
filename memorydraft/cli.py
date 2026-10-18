from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from memorydraft.memory.store import MemoryEvent, MemoryStore


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="MemoryDraft inference, serving and correctness evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    from memorydraft.runtime.cli import add_commands
    add_commands(sub)
    env = sub.add_parser("env")
    env.add_argument("--output")
    imp = sub.add_parser("import")
    imp.add_argument("--db", required=True)
    imp.add_argument("--input", required=True)
    report = sub.add_parser("report")
    report.add_argument("--input", required=True)
    report.add_argument("--output", required=True)
    for name in ("replay", "generate"):
        run = sub.add_parser(name)
        run.add_argument("--config", required=True)
        run.add_argument("--db", required=True)
        run.add_argument("--output", required=True)
        run.add_argument("--max-new-tokens", type=int, default=64)
        run.add_argument("--block-size", type=int, default=8)
        run.add_argument("--temperature", type=float, default=0.0)
        if name == "replay":
            run.add_argument("--requests", required=True)
            run.add_argument("--methods", nargs="+", default=["AR", "D", "R", "RD", "H", "RHD"])
            run.add_argument("--repeats", type=int, default=1)
            run.add_argument("--history-condition", choices=["relevant", "empty", "unrelated"], default="relevant",
                             help="Offline controlled ablation; never changes Target messages")
        else:
            run.add_argument("--request", required=True, help="JSON file with id/user_id/order/messages")
            run.add_argument("--method", default="RHD")
            run.add_argument("--commit-order", type=int, help="Persist final turn only, at this later availability order")
    args = parser.parse_args(argv)
    if args.command in {"infer", "benchmark", "profile", "serve", "load-test"}:
        from memorydraft.runtime.cli import run
        run(args)
        return
    if args.command == "import":
        with MemoryStore(args.db) as store:
            print(json.dumps({"imported": store.import_jsonl(args.input)}))
        return
    if args.command == "report":
        from memorydraft.evaluation.replay import read_jsonl, summarize
        write_json(args.output, summarize(read_jsonl(args.input)))
        return
    if args.command == "env":
        import torch
        import transformers
        result = {"python": sys.version, "platform": platform.platform(),
                  "torch": torch.__version__, "transformers": transformers.__version__,
                  "cuda_available": torch.cuda.is_available(),
                  "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        print(json.dumps(result, indent=2))
        if args.output:
            write_json(args.output, result)
        return
    from memorydraft.backends.transformers import TransformersBackend
    from memorydraft.evaluation.replay import read_jsonl, replay
    from memorydraft.retrieval.pool import HybridEncoder, LexicalEncoder, SemanticEncoder
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.temperature != 0:
        parser.error("Sampling is not implemented; use --temperature 0")
    retrieval = config.get("retrieval", {"kind": "lexical"})
    if retrieval["kind"] not in {"semantic", "hybrid", "lexical"}:
        parser.error("Unknown retrieval kind")
    encoder = SemanticEncoder(retrieval["model"], retrieval["revision"]) if retrieval["kind"] != "lexical" else LexicalEncoder()
    if retrieval["kind"] == "hybrid":
        encoder = HybridEncoder(encoder)
    backend = TransformersBackend.load(config)
    options = dict(max_new_tokens=args.max_new_tokens, block_size=args.block_size, temperature=args.temperature)
    for key in ("token_budget", "history_budget", "top_k", "min_match", "r_scope"):
        if key in config:
            options[key] = config[key]
    # Warm model kernels without reading any replay answers into memory.
    with MemoryStore(args.db) as store:
        backend.generate([{"role": "user", "content": "Say hello."}], store.snapshot("__warmup__", 0),
                         method="D", max_new_tokens=8, encoder=encoder)
        if args.command == "replay":
            summary = replay(backend, store, read_jsonl(args.requests), args.methods, args.output,
                             repeats=args.repeats, encoder=encoder, history_condition=args.history_condition, **options)
            write_json(str(args.output) + ".summary.json", summary)
            print(json.dumps(summary, indent=2))
        else:
            request = json.loads(Path(args.request).read_text(encoding="utf-8"))
            if args.commit_order is not None and args.commit_order <= request["order"]:
                parser.error("commit-order must be later than request order")
            snapshot_start = time.perf_counter()
            snapshot = store.snapshot(request["user_id"], request["order"])
            snapshot_seconds = time.perf_counter() - snapshot_start
            result = backend.generate(request["messages"], snapshot,
                                      method=args.method, encoder=encoder, **options)
            result.latency_seconds += snapshot_seconds
            if result.ttft_seconds is not None:
                result.ttft_seconds += snapshot_seconds
            result.stage_seconds["snapshot_shared"] = snapshot_seconds
            if args.commit_order is not None:
                writeback_start = time.perf_counter()
                store.append([MemoryEvent(request["user_id"], request["id"] + ":" + role,
                                          args.commit_order, text, source_role=role)
                              for role, text in (("user", request["messages"][-1]["content"]), ("assistant", result.text))])
                writeback_seconds = time.perf_counter() - writeback_start
                result.stage_seconds["writeback"] = writeback_seconds
                result.latency_seconds += writeback_seconds
            write_json(args.output, asdict(result))
            print(result.text)
    provenance = {
        "config": config, "tokenizer_fingerprint": backend.fingerprint,
        "target_commit": getattr(backend.target.config, "_commit_hash", None),
        "draft_commit": getattr(backend.draft.config, "_commit_hash", None),
        "retriever": encoder.name, "max_new_tokens": args.max_new_tokens, "block_size": args.block_size,
        "history_condition": getattr(args, "history_condition", "relevant"),
        "timing": "Request tokenization/retrieval/prompts/candidate setup/generation included; model and encoder load/warmup excluded. Per-phase host times overlap CUDA execution; only synchronized request latency is authoritative.",
    }
    manifest_path = Path(__file__).resolve().parents[1] / "SOURCE_MANIFEST.json"
    if manifest_path.exists():
        provenance["source_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[1]
    provenance["git_revision"] = None
    # A source archive may sit inside an unrelated checkout on the server.
    # Never walk into that checkout merely to attach provenance to this run.
    if (repo_root / ".git").exists():
        try:
            provenance["git_revision"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL, text=True,
            ).strip()
            provenance["git_dirty"] = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.DEVNULL, text=True,
            ).strip())
        except (OSError, subprocess.CalledProcessError):
            provenance["git_revision"] = None
    write_json(str(args.output) + ".provenance.json", provenance)


if __name__ == "__main__":
    main()
