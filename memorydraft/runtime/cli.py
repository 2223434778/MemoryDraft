from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest


def add_commands(sub):
    load = sub.add_parser("load-test")
    load.add_argument("--url", default="http://127.0.0.1:8000")
    load.add_argument("--request", required=True)
    load.add_argument("--method", default="AR")
    load.add_argument("--count", type=int, default=20)
    load.add_argument("--concurrency", type=int, default=2)
    load.add_argument("--timeout", type=float, default=120.0)
    load.add_argument("--output", required=True)
    for name in ("infer", "benchmark", "profile", "serve"):
        parser = sub.add_parser(name)
        parser.add_argument("--config", required=True, help="Runtime config; paths are relative to this file")
        if name in {"infer", "profile"}:
            parser.add_argument("--request", required=True)
            parser.add_argument("--method", default="AR")
            parser.add_argument("--output", required=True)
        elif name == "benchmark":
            parser.add_argument("--requests", required=True, help="JSONL generation requests")
            parser.add_argument("--methods", nargs="+", default=["AR"])
            parser.add_argument("--repeats", type=int, default=3)
            parser.add_argument("--baseline", help="Previous compatible real-model benchmark directory")
            parser.add_argument("--max-slowdown", type=float, default=.10)
            parser.add_argument("--output", required=True, help="New artifact directory; never overwritten")
        else:
            parser.add_argument("--host", default="127.0.0.1")
            parser.add_argument("--port", type=int, default=8000)
            parser.add_argument("--queue-capacity", type=int, default=8)
            parser.add_argument("--event-capacity", type=int, default=32)


def run(args):
    if args.command == "load-test":
        from memorydraft.serving.loadtest import load_test
        body = json.loads(Path(args.request).read_text(encoding="utf-8"))
        result = load_test(args.url, {**body, "method": args.method}, args.output,
                           count=args.count, concurrency=args.concurrency, timeout_seconds=args.timeout)
        print(json.dumps(result, indent=2))
        return
    config = RuntimeConfig.load(args.config)
    if args.command == "serve":
        import uvicorn
        from memorydraft.serving.app import create_app
        app = create_app(lambda: load_engine(config), queue_capacity=args.queue_capacity, event_capacity=args.event_capacity)
        uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)
        return
    if Path(args.output).exists():
        raise FileExistsError("Preserve existing artifacts; choose a new output path: " + args.output)
    engine = load_engine(config)
    try:
        from memorydraft.runtime.benchmark import benchmark, check_regression, profile_request
        if args.command == "benchmark":
            requests = [GenerationRequest.from_dict(json.loads(line))
                        for line in Path(args.requests).read_text(encoding="utf-8").splitlines() if line.strip()]
            result = benchmark(engine, requests, args.methods, args.output, repeats=args.repeats)
            if args.baseline:
                result["regression"] = check_regression(args.output, args.baseline, max_slowdown=args.max_slowdown)
                if not result["regression"]["passed"]:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    raise SystemExit(1)
        else:
            request = GenerationRequest.from_dict(json.loads(Path(args.request).read_text(encoding="utf-8")))
            if args.command == "profile":
                result = profile_request(engine, request, args.method, args.output)
            else:
                result = asdict(engine.generate(request, args.method))
                destination = Path(args.output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("x", encoding="utf-8") as stream:
                    json.dump(result, stream, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        engine.close()
