"""Paired off/cold/hit measurements, then gated HTTP checks, on one GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import socket
import threading
import time
from unittest.mock import patch

import httpx
import torch
import uvicorn

from memorydraft.evaluation.asd_suite import score
from memorydraft.evaluation.long_context import LEVELS, check_input, load_data, request_for
from memorydraft.evaluation.prefix_cache import STUDY, analyze, compare
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.runtime.benchmark import provenance, write_json
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import RequestCancelled
from memorydraft.serving.app import create_app
from scripts.run_runtime_migration import append
from scripts.validate_asd_runtime import sse_events

ROOT = Path(__file__).resolve().parents[1]


def http_checks(engine,items,references,output,*,load_states=("off","cache")):
    if not load_states or len(set(load_states)) != len(load_states) or not set(load_states) <= {"off","cache"}:
        raise ValueError("Invalid HTTP load states")
    chosen = [r for r in items if r["level"] == 4096][:4]
    app = create_app(lambda:engine,queue_capacity=4)
    sock = socket.socket()
    sock.bind(("127.0.0.1",0));sock.listen(128)
    url = "http://127.0.0.1:"+str(sock.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config(app,log_level="warning",access_log=False))
    thread = threading.Thread(target=server.run,kwargs={"sockets":[sock]},daemon=True)
    thread.start()
    summary = dict(status="started",mode="closed_loop",concurrency=2,requests_per_state=20,load_states=list(load_states))
    obj = engine._prefix_cache
    try:
        deadline = time.perf_counter()+30
        while not server.started:
            if not thread.is_alive() or time.perf_counter()>deadline:raise RuntimeError("HTTP startup failed")
            time.sleep(.02)
        with httpx.Client(trust_env=False,timeout=120) as client:
            item = chosen[0]
            body = {**asdict(request_for(item,128)),"method":"ASD"}
            actual = client.post(url+"/v1/generate",json=body)
            actual.raise_for_status()
            row = actual.json()
            if compare(references[item["request"]["id"],"ASD"],row)["changed"]:raise ValueError("HTTP JSON changed output")
            write_json(output/"http-json.json",row)
            response = client.post(url+"/v1/generate",json={**body,"stream":True})
            events = list(sse_events(response))
            emitted = [t for e,v in events if e == "tokens" for t in v["token_ids"]]
            complete = next(v for e,v in events if e == "complete")
            if emitted != complete["token_ids"] or compare(row,complete)["changed"]:
                raise ValueError("HTTP stream changed output")
            write_json(output/"http-stream.json",complete)
            # Disconnect while generating; the subsequent queued request must recover.
            before = client.get(url+"/metrics").json()
            with client.stream("POST",url+"/v1/generate",json={**body,"stream":True,"max_new_tokens":4096}) as response:
                response.raise_for_status()
                event = None
                for line in response.iter_lines():
                    if line.startswith("event: "):event = line[7:]
                    elif event == "tokens" and line.startswith("data: "):break
            deadline = time.perf_counter()+30
            while app.state.worker.metrics()["running"]:
                if time.perf_counter()>deadline:raise RuntimeError("HTTP cancellation did not release worker")
                time.sleep(.02)
            if client.get(url+"/metrics").json()["cancelled"] != before["cancelled"]+1:
                raise ValueError("Disconnect not recorded as cancellation")
            recovered = client.post(url+"/v1/generate",json=body)
            recovered.raise_for_status()
            if compare(row,recovered.json())["changed"]:raise ValueError("Cancellation polluted public cache")
            for state in load_states:
                obj.clear()
                context = patch.object(engine,"_prefix_cache",None) if state == "off" else nullcontext()
                with context:
                    def send(i):
                        item = chosen[i%len(chosen)]
                        payload = {**asdict(request_for(item,128)),"id":f"queue-{state}-{i}",
                            "method":"ASD","user_id":item["request"]["user_id"],"stream":True}
                        started = time.perf_counter()
                        first,emitted,complete = None,[],None
                        with client.stream("POST",url+"/v1/generate",json=payload) as response:
                            response.raise_for_status()
                            event = None
                            for line in response.iter_lines():
                                if line.startswith("event: "):event = line[7:]
                                elif line.startswith("data: "):
                                    value = json.loads(line[6:])
                                    if event == "tokens":
                                        if first is None:first = time.perf_counter()-started
                                        emitted.extend(value["token_ids"])
                                    elif event == "complete":complete = value
                                    elif event == "error":raise RuntimeError(str(value))
                        if complete is None or emitted != complete["token_ids"]:
                            raise ValueError("Incomplete queued HTTP stream")
                        expected = references[item["request"]["id"],"ASD"]
                        if compare(expected,complete)["changed"]:raise ValueError("Queued HTTP output changed")
                        return dict(index=i,state=state,task_id=item["task_id"],
                            client_seconds=time.perf_counter()-started,client_ttft_seconds=first,result=complete)
                    started = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=2) as pool:rows = list(pool.map(send,range(20)))
                    elapsed = time.perf_counter()-started
                    for r in rows:append(output/"http-load.jsonl",r)
                    summary[state] = dict(wall_seconds=elapsed,requests=len(rows),
                        tokens=sum(len(r["result"]["token_ids"]) for r in rows),
                        hits=sum(r["result"]["counters"].get("prefix_cache_hit",0) for r in rows),
                        mean_client_seconds=sum(r["client_seconds"] for r in rows)/len(rows),
                        mean_client_ttft_seconds=sum(r["client_ttft_seconds"] for r in rows)/len(rows),
                        mean_queue_seconds=sum(r["result"]["queue_seconds"] for r in rows)/len(rows))
            summary.update(status="complete",json_and_sse_match=True,disconnect_cancelled=True,recovered=True,
                           metrics=client.get(url+"/metrics").json())
    finally:
        server.should_exit = True
        thread.join(60)
        sock.close()
        write_json(output/"http-summary.json",summary)
        if thread.is_alive():raise RuntimeError("HTTP shutdown did not finish")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard",type=int,choices=(0,1),required=True)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/runtime-dflash-asd-prefix-cache.json")
    parser.add_argument("--levels",type=int,nargs="+",choices=LEVELS,default=list(LEVELS))
    parser.add_argument("--study",choices=(STUDY,"asd-shared-prefix"),default=STUDY)
    parser.add_argument("--variant",choices=("whole","shared"))
    parser.add_argument("--mode",choices=("eval","http-only"),default="eval")
    parser.add_argument("--load-states",nargs="+",choices=("off","cache"),default=["off","cache"])
    args = parser.parse_args()
    if 4096 not in args.levels or len(set(args.levels)) != len(args.levels):
        parser.error("HTTP checks require 4096 and unique input levels")
    if len(set(args.load_states)) != len(args.load_states):parser.error("Duplicate HTTP load state")
    tasks,items,data = load_data(ROOT)
    chosen = {r["id"] for i,r in enumerate(tasks) if i%2 == args.shard}
    items = [r for r in items if r["task_id"] in chosen and r["level"] in args.levels]
    tasks = {r["id"]:r for r in tasks}
    label = f"{args.variant}-{args.shard}" if args.variant else str(args.shard)
    output = ROOT/f"results/raw/{args.study}/eval-{label}"
    output.mkdir(parents=True,exist_ok=False)
    info = dict(status="started",shard=args.shard,data_files=data["files"],rows=0,
                variant=args.variant,mode=args.mode,levels=args.levels,load_states=args.load_states)
    engine = None
    all_rows = []
    try:
        started = time.perf_counter()
        engine = load_engine(RuntimeConfig.load(args.config))
        obj = engine._prefix_cache
        info.update(runtime=provenance(engine),model_setup_seconds=time.perf_counter()-started)
        def generate(item,method,limit,state):
            context = patch.object(engine,"_prefix_cache",None) if state == "off" else nullcontext()
            with context:row = asdict(engine.generate(request_for(item,limit),method))
            check_input(row,item)
            if row["peak_gpu_bytes"] > 46*2**30:raise torch.cuda.OutOfMemoryError("GPU budget exceeded")
            return row
        for level in sorted(set(args.levels)):
            item = next(r for r in items if r["level"] == level)
            obj.clear()
            for method in ("DFlash","ASD"):
                for state in ("off","cold","hit"):
                    if state == "cold":obj.clear()
                    append(output/"warmup.jsonl",dict(generate(item,method,16,state),level=level,cache_state=state))
        info["compiler_after_warmup"] = engine._regional.summary()
        old = {(r["request_id"],r["method"],r["stage"]):r for i in (0,1) for r in read_rows(
                ROOT/f"results/asd-long-context/eval-base-{i}/rows.jsonl") if r["repeat"] == 0}
        reference_ar = {r["request_id"]:r for i in (0,1) for r in read_rows(
                ROOT/f"results/asd-long-context/eval-base-{i}/references.jsonl")}
        prefix_refs = {(r["request"]["id"],m):old[r["request"]["id"],m,"prefix"] for r in items for m in ("DFlash","ASD")}
        for item in items if args.mode == "eval" else ():
            jobs = [(method,stage,limit) for method in ("DFlash","ASD") for stage,limit in (("natural",4096),("prefix",128))]
            rng = random.Random("public-prefix:"+item["request"]["id"])
            rng.shuffle(jobs)
            for method,stage,limit in jobs:
                states = ["off","cold","hit"] if rng.random() < .5 else ["cold","hit","off"]
                for state in states:
                    if state == "cold":obj.clear()
                    row = generate(item,method,limit,state)
                    row.update(task_id=item["task_id"],level=item["level"],stage=stage,cache_state=state,
                        quality=score(tasks[item["task_id"]],row["text"]) if stage == "natural" else None)
                    row["comparison_to_previous"] = compare(old[row["request_id"],method,stage],row)
                    if method == "DFlash":row["strict_token_match_ar"] = row["token_ids"] == reference_ar[row["request_id"]]["token_ids"][:limit]
                    append(output/"rows.jsonl",row)
                    all_rows.append(row)
                    if state == "off" and stage == "prefix":prefix_refs[row["request_id"],method] = row
                    info["rows"] += 1
                write_json(output/"provenance.json",info)
            print("evaluated",args.shard,item["task_id"],item["level"],info["rows"],flush=True)
        if args.mode == "eval":
            analysis = analyze(all_rows,items)
            write_json(output/"analysis.json",analysis)
            if not analysis["all_outputs_and_decisions_equal"] or any(not r["strict_token_match_ar"] for r in all_rows if r["method"] == "DFlash"):
                raise ValueError("Cache correctness gate failed; HTTP stage not started")
        # Cross-question requests may share only the registered public prefix.
        obj.clear()
        cross = []
        for item in [r for r in items if r["level"] == 4096]:
            row = generate(item,"ASD",128,"hit")
            if compare(prefix_refs[row["request_id"],"ASD"],row)["changed"]:raise ValueError("Cross-question reuse changed output")
            cross.append(row)
        write_json(output/"cross-question.json",cross)
        # Check cancellation directly before handing model ownership to the HTTP worker.
        seen = []
        try:
            engine.generate(request_for(items[-1],4096),"ASD",on_tokens=seen.extend,should_stop=lambda:bool(seen))
        except RequestCancelled:pass
        else:raise ValueError("Direct cancellation was not exercised")
        info["http"] = http_checks(engine,items,prefix_refs,output,load_states=tuple(args.load_states))
        info.update(status="complete",compiler=engine._regional.summary())
    except Exception as error:
        info.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if engine is not None:engine.close()
        write_json(output/"provenance.json",info)


if __name__ == "__main__":
    main()
