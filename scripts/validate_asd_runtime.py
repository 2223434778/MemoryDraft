"""Small real-model integration acceptance against frozen research trajectories."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import socket
import threading
import time

import httpx
import uvicorn

from memorydraft.evaluation.adaptation_followup import records
from memorydraft.evaluation.asd_suite import request_for, score
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.runtime.benchmark import benchmark, provenance, write_json
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.runtime.contracts import GenerationRequest, RequestCancelled
from memorydraft.serving.app import create_app
from memorydraft.serving.loadtest import load_test

ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = ("gsm8k:1196", "gsm8k:959", "gsm8k:450", "gsm8k:885")
ADAPTER_SHA = "42755672a816b44b727cbc5bcdbbd2b18dd2a19db42e5706f09d540d678b1f6e"


def references(variant):
    return {(r["request_id"],r["stage"],r["policy"]):r for i in (0,1) for r in read_rows(
        ROOT/f"results/asd-adaptation-followup/eval-{i}/rows.jsonl") if r["variant"] == variant and r["repeat"] == 0}


def check_result(row, reference, *, adapter_sha):
    for field in ("token_ids","text","finish_reason","prompt_sha256","snapshot_id","execution_profile"):
        if row[field] != reference[field]:
            raise ValueError("Runtime/research mismatch: "+field)
    asd = reference["policy"] == "ASD"
    if row["approximate"] != asd or row["draft_adapter_sha256"] != adapter_sha:
        raise ValueError("Runtime method/adapter identity mismatch")
    if row["counters"]["rounds"] != len(reference["rounds"]):
        raise ValueError("Runtime verification rounds changed")
    if asd:
        if (row["asd"]["gate"] != .25 or row["asd"]["cap"] != 2
                or any(row["asd"][k] != v for k,v in reference["asd"].items())
                or not 0 <= row["asd"]["spent"] <= 8):
            raise ValueError("Runtime ASD budget/trajectory changed")
    elif row["asd"] is not None:
        raise ValueError("Strict method unexpectedly used ASD")
    return True


def sse_events(response):
    response.raise_for_status()
    for block in response.text.split("\n\n"):
        lines = block.splitlines()
        if len(lines) == 2:
            yield lines[0][7:],json.loads(lines[1][6:])


def http_acceptance(engine, record, old, out, adapter_sha):
    request = request_for(record,128)
    body = {**asdict(request),"messages":list(request.messages)}
    app = create_app(lambda:engine)
    sock = socket.socket()
    sock.bind(("127.0.0.1",0))
    sock.listen(128)
    url = "http://127.0.0.1:"+str(sock.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config(app,log_level="warning",access_log=False))
    thread = threading.Thread(target=server.run,kwargs={"sockets":[sock]},daemon=True)
    thread.start()
    try:
        deadline = time.perf_counter()+30
        while not server.started:
            if not thread.is_alive() or time.perf_counter() > deadline:
                raise RuntimeError("HTTP startup failed")
            time.sleep(.02)
        with httpx.Client(trust_env=False,timeout=120) as client:
            health = client.get(url+"/healthz").json()
            if health["approximate_methods"] != ["ASD"] or health["draft_adapter_sha256"] != adapter_sha:
                raise ValueError("HTTP health metadata mismatch")
            response = client.post(url+"/v1/generate",json={**body,"method":"ASD"})
            response.raise_for_status()
            row = response.json()
            check_result(row,old[record["id"],"prefix","ASD"],adapter_sha=adapter_sha)
            write_json(out/"http-json.json",row)
            streams = {}
            for method in ("DFlash","ASD"):
                response = client.post(url+"/v1/generate",json={**body,"method":method,"stream":True})
                tokens,start,complete = [],None,None
                for event,data in sse_events(response):
                    if event == "start":start = data
                    elif event == "tokens":tokens.extend(data["token_ids"])
                    elif event == "complete":complete = data
                    elif event == "error":raise RuntimeError(str(data))
                if start is None or start["approximate"] != (method == "ASD") or complete is None or tokens != complete["token_ids"]:
                    raise ValueError("HTTP streaming contract mismatch")
                check_result(complete,old[record["id"],"prefix",method],adapter_sha=adapter_sha)
                streams[method] = complete
            write_json(out/"http-streams.json",streams)
            before = client.get(url+"/metrics").json()
            received = []
            with client.stream("POST",url+"/v1/generate",json={**body,"max_new_tokens":4096,"method":"ASD","stream":True}) as response:
                response.raise_for_status()
                event = None
                for line in response.iter_lines():
                    if line.startswith("event: "):event = line[7:]
                    elif event == "tokens" and line.startswith("data: "):
                        received.extend(json.loads(line[6:])["token_ids"])
                        if received:break
            if not received:
                raise ValueError("Disconnect test did not receive model tokens")
            deadline = time.perf_counter()+20
            while app.state.worker.metrics()["running"]:
                if time.perf_counter() > deadline:raise RuntimeError("Disconnected request did not release worker")
                time.sleep(.02)
            after = client.get(url+"/metrics").json()
            if after["cancelled"] != before["cancelled"]+1:
                raise ValueError("HTTP disconnect was not recorded as cancellation")
            recovered = client.post(url+"/v1/generate",json={**body,"method":"ASD"})
            recovered.raise_for_status()
            check_result(recovered.json(),old[record["id"],"prefix","ASD"],adapter_sha=adapter_sha)
            load = load_test(url,{**body,"method":"ASD"},out/"http-load",count=4,concurrency=2)
            if load["failed"]:raise ValueError("Queued HTTP requests failed")
            return dict(health=health,json_matches=True,streams_match=True,disconnect_cancelled=True,
                        disconnect_tokens=len(received),recovered=True,load=load,metrics=client.get(url+"/metrics").json())
    finally:
        server.should_exit = True
        thread.join(120)
        sock.close()
        if thread.is_alive():raise RuntimeError("HTTP shutdown did not finish")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config = RuntimeConfig.load(args.config)
    if (config.backend != "dflash" or config.execution_profile != "rmsnorm-inductor"
            or config.asd != dict(budget=8,gate=.25,cap=2) or config.target_revision != "b968826d9c46dd6066d109eabc6255188de91218"):
        raise ValueError("Acceptance pins the published Qwen/DFlash ASD profile")
    adapter_sha = config.draft_adapter.get("sha256")
    if adapter_sha not in (None,ADAPTER_SHA):raise ValueError("Unknown acceptance adapter")
    variant = "ordinary-position" if adapter_sha else "base"
    old = references(variant)
    tasks = {r["id"]:r for r in records(ROOT)}
    started = time.perf_counter()
    engine = load_engine(config)
    info = {**provenance(engine),"status":"started","variant":variant,"model_setup_seconds":time.perf_counter()-started,
            "task_ids":TASK_IDS,"note":"Integration acceptance only; not a new speedup study"}
    write_json(args.output/"acceptance.json",info)
    try:
        warm = GenerationRequest("cold-asd-runtime","warm",0,({"role":"user","content":"What is 3 plus 4?"},),32)
        cold = [asdict(engine.generate(warm,m)) for m in ("AR","DFlash","ASD")]
        write_json(args.output/"cold-start.json",cold)
        rows,ars = [],[]
        for task in TASK_IDS:
            record = tasks[task]
            ar = asdict(engine.generate(request_for(record,4096),"AR"))
            ar["quality"] = score(record,ar["text"])
            ars.append(ar)
            for stage,limit in (("natural",4096),("prefix",128)):
                for method in ("DFlash","ASD"):
                    commits = []
                    row = asdict(engine.generate(request_for(record,limit),method,on_tokens=commits.extend))
                    if commits != row["token_ids"]:raise ValueError("Direct stream commits changed")
                    check_result(row,old[task,stage,method],adapter_sha=adapter_sha)
                    if method == "DFlash" and row["token_ids"] != ar["token_ids"][:limit]:
                        raise ValueError("Strict output differs from AR")
                    row.update(stage=stage,quality=score(record,row["text"]) if stage == "natural" else None,research_match=True)
                    rows.append(row)
            print("accepted",variant,task,flush=True)
        write_json(args.output/"rows.json",rows)
        write_json(args.output/"ar-references.json",ars)
        request = request_for(tasks[TASK_IDS[0]],128)
        partial = []
        try:
            engine.generate(request,"ASD",on_tokens=partial.extend,should_stop=lambda:len(partial)>3)
            raise ValueError("Direct cancellation was ignored")
        except RequestCancelled:
            if not partial:raise ValueError("Cancellation did not exercise a token boundary")
        check_result(asdict(engine.generate(request,"ASD")),old[TASK_IDS[0],"prefix","ASD"],adapter_sha=adapter_sha)
        info["cancellation_recovered"] = True
        info["strict_benchmark"] = benchmark(engine,[request_for(tasks["gsm8k:959"],32)],["DFlash"],args.output/"strict-benchmark",repeats=1,warmup=False)
        try:
            benchmark(engine,[request],["ASD"],args.output/"invalid-strict-benchmark",repeats=1)
            raise RuntimeError("Approximate method entered strict benchmark")
        except ValueError as error:
            if "approximate" not in str(error):raise
        info["strict_benchmark_guard"] = True
        info["http"] = http_acceptance(engine,tasks["gsm8k:959"],old,args.output,adapter_sha)
        info.update(status="complete",rows=len(rows),ar_references=len(ars),compiler=engine._regional.summary(),
                    peak_gpu_bytes=max(r["peak_gpu_bytes"] for r in [*rows,*ars,*cold]),
                    formal_new_graphs=sum(r["counters"]["compile_graphs_after"]-r["counters"]["compile_graphs_before"] for r in rows),
                    seconds=time.perf_counter()-started)
    except Exception as error:
        info.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        engine.close()
        info["profile_restored_after_close"] = not engine._regional.active
        write_json(args.output/"acceptance.json",info)
    print(json.dumps(dict(status=info["status"],variant=variant,rows=info["rows"])),flush=True)


if __name__ == "__main__":
    main()
