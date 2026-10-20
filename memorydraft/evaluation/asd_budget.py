"""Frozen new math slices and conservative final-answer scoring for the budget probe."""
from __future__ import annotations

import json
from pathlib import Path
import re

from memorydraft.evaluation.asd_bench import request_for as original_request, score as original_score
from memorydraft.runtime.contracts import GenerationRequest


def load_tasks(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = [*(f"gsm8k:{i}" for i in range(8, 24)), *(f"math500:{i}" for i in range(8))]
    if data["schema"] != 1 or [r["id"] for r in data["records"]] != expected:
        raise ValueError("Changed frozen ASD budget task selection")
    return data["records"]


def request_for(record):
    if record["benchmark"] == "gsm8k":
        return original_request(record)
    if record["benchmark"] != "math500":
        raise ValueError("Unknown budget benchmark")
    prompt = "Solve the following math problem step by step. End with your final answer in \\boxed{...}.\n\n" + record["problem"]
    return GenerationRequest(record["id"], "asd-budget-" + record["id"], 1,
                             ({"role": "user", "content": prompt},), 512)


def boxed_answer(text):
    answers = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start, depth = match.end(), 1
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                answers.append(text[start:i])
                break
    return answers[-1] if answers else None


def normalize_answer(value):
    value = re.sub(r"\s+", "", value).replace("$", "")
    for old, new in ((r"\left", ""), (r"\right", ""), (r"\dfrac", r"\frac"),
                     (r"\tfrac", r"\frac"), (r"\,", ""), (r"\!", ""),
                     (r"^{\circ}", ""), (r"^\circ", "")):
        value = value.replace(old, new)
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\frac([0-9])([0-9])", r"\\frac{\1}{\2}", value)
    return value


def score(record, text):
    if record["benchmark"] == "gsm8k":
        return original_score(record, text)
    actual = boxed_answer(text)
    expected = normalize_answer(record["answer"])
    return dict(correct=actual is not None and normalize_answer(actual) == expected,
                parsed_answer=actual, expected_answer=record["answer"],
                metric="normalized boxed exact match, not general symbolic equivalence",
                error=None if actual is not None else "missing boxed final answer")
