"""Frozen few-shot math workloads and honest first-token timing decomposition."""
from dataclasses import replace
import hashlib
import json
import math

from memorydraft.runtime.contracts import GenerationRequest

STUDY = "asd-long-context"
LEVELS = (512, 2048, 4096)
ARMS = ("DFlash", "ASD", "position-ASD")
NATURAL_REPEATS = 2


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prompt(question, demonstrations):
    header = ('Solve the final arithmetic word problem. The worked examples illustrate the format. '
              'Show your reasoning and end with "#### <number>".\n\n')
    examples = "\n\n".join(f"Worked example {i+1}:\nQuestion: {r['question']}\nSolution: {r['answer']}"
                              for i,r in enumerate(demonstrations))
    return header+examples+f"\n\nFinal question:\n{question}\nSolution:"


def request_for(item, limit=4096):
    return replace(GenerationRequest.from_dict(item["request"]),max_new_tokens=limit)


def load_data(root):
    directory = root/f"data/workloads/{STUDY}"
    info = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    for name,digest in info["files"].items():
        if sha((directory/name).read_bytes()) != digest:raise ValueError("Changed long-context data: "+name)
    tasks = json.loads((directory/"tasks.json").read_text(encoding="utf-8"))
    items = json.loads((directory/"requests.json").read_text(encoding="utf-8"))
    expected = {(r["id"],level) for r in tasks for level in LEVELS}
    if len(tasks) != 8 or len({r["id"] for r in tasks}) != 8 or len(items) != 24 or {(r["task_id"],r["level"]) for r in items} != expected:
        raise ValueError("Missing or duplicated long-context matrix")
    for item in items:
        request_for(item)
        if not .75*item["level"] <= item["prompt_tokens"] <= 1.25*item["level"] or item["examples"] < 1:
            raise ValueError("Unexpected few-shot input size")
    return tasks,items,info


def check_input(row,item):
    if row["prompt_sha256"] != item["prompt_sha256"] or row["prompt_tokens"] != item["prompt_tokens"]:
        raise ValueError("Runtime tokenizer/input differs from frozen request")
    if row["execution_profile"] != "rmsnorm-inductor":raise ValueError("Changed numerical profile")
    if row["asd"] is not None and (row["asd"]["budget"] != 8 or not 0 <= row["asd"]["spent"] <= 8):
        raise ValueError("ASD exceeded original budget")


def timing(row):
    wall,first,n = row["latency_seconds"],row["ttft_seconds"],len(row["token_ids"])
    if (first is None or not all(math.isfinite(v) for v in (wall,first)) or not 0 <= first <= wall or n < 1):
        raise ValueError("Invalid first-token timing")
    return dict(seconds=wall,ttft_seconds=first,post_first_seconds=wall-first,
                tpot_seconds=(wall-first)/(n-1) if n > 1 else None,tokens=n,rounds=row["counters"]["rounds"])
