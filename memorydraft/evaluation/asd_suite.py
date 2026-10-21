"""Versioned cross-task evaluation, independent of model execution and selection."""
from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import string
import subprocess
import sys
import tempfile

from memorydraft.evaluation.asd_bench import extract_code, score as old_score
from memorydraft.evaluation.asd_budget import boxed_answer, normalize_answer, score as math_score
from memorydraft.runtime.contracts import GenerationRequest

BENCHMARKS = ("gsm8k", "math500", "mbpp", "humaneval", "squad", "json")
SPLITS = ("development", "validation", "reserve")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_suite(directory, split="development"):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != 1 or split not in SPLITS:
        raise ValueError("Unknown suite version or split")
    rows = []
    for benchmark in BENCHMARKS:
        path = directory / (benchmark + ".json")
        raw = path.read_bytes()
        if sha(raw) != manifest["files"][path.name]:
            raise ValueError("Changed frozen dataset: " + benchmark)
        records = json.loads(raw)
        if len(records) != 32 or any(r["benchmark"] != benchmark for r in records):
            raise ValueError("Changed benchmark selection")
        if {s: sum(r["split"] == s for r in records) for s in SPLITS} != dict(development=8, validation=8, reserve=16):
            raise ValueError("Changed split counts")
        rows.extend(r for r in records if r["split"] == split)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate suite IDs")
    return rows


def request_for(record, max_new_tokens=4096):
    b = record["benchmark"]
    if b in {"gsm8k", "math500"}:
        from memorydraft.evaluation.asd_budget import request_for as original
        request = original(record)
        return GenerationRequest(record["id"], "asd-suite-" + record["id"], 1, request.messages, max_new_tokens)
    if b == "mbpp":
        prompt = ("Write Python code for the following task. Define the function used in these tests. "
                  "Return only the implementation, without test calls or explanation.\n\n" + record["text"]
                  + "\n\nYour code should pass these tests:\n" + "\n".join(record["test_list"]))
    elif b == "humaneval":
        prompt = ("Write the complete Python implementation described below, including the function signature. "
                  "Return Python code only, without tests or explanation.\n\n" + record["prompt"])
    elif b == "squad":
        prompt = ("Read the passage and answer the question. Return only the shortest answer phrase supported by "
                  "the passage, without explanation.\n\nPassage:\n" + record["context"] + "\n\nQuestion: " + record["question"])
    elif b == "json":
        prompt = record["instruction"]
    else:
        raise ValueError("Unknown benchmark")
    return GenerationRequest(record["id"], "asd-suite-" + record["id"], 1,
                             ({"role": "user", "content": prompt},), max_new_tokens)


def normalize_qa(text):
    value = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def qa_score(text, answers):
    predicted = normalize_qa(text).split()
    exact, f1 = False, 0.
    for answer in answers:
        target = normalize_qa(answer).split()
        exact |= predicted == target
        shared = sum((Counter(predicted) & Counter(target)).values())
        if not predicted or not target:
            value = float(predicted == target)
        elif shared:
            precision, recall = shared / len(predicted), shared / len(target)
            value = 2 * precision * recall / (precision + recall)
        else:
            value = 0.
        f1 = max(f1, value)
    return dict(correct=exact, f1=f1, metric="SQuAD normalized exact match and token F1, maximum over references")


def scalar_answer(value):
    value = normalize_answer(value).strip()
    value = re.sub(r"^[A-Za-z]=", "", value)
    match = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", value)
    if match:
        try:
            return Fraction(int(match[2]), int(match[3])) * (-1 if match[1] else 1)
        except ZeroDivisionError:
            return None
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", value):
        return None
    try:
        numerator, *denominator = value.split("/")
        return Fraction(numerator) / (Fraction(denominator[0]) if denominator else 1)
    except (ValueError, ZeroDivisionError):
        return None


def validate_code(code):
    """Task-specific import/builtin restrictions; not a general Python sandbox."""
    tree = ast.parse(code)
    allowed = {"math", "cmath", "collections", "itertools", "functools", "re", "heapq", "bisect", "string", "typing",
               "statistics", "operator", "decimal", "fractions", "copy", "random"}
    forbidden = {"open", "eval", "exec", "compile", "input", "globals", "locals", "vars", "getattr", "setattr", "delattr", "breakpoint"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] not in allowed for a in node.names):
            raise ValueError("unsupported import")
        if isinstance(node, ast.ImportFrom) and (node.level or not node.module or node.module.split(".")[0] not in allowed):
            raise ValueError("unsupported import")
        if isinstance(node, ast.Name) and (node.id in forbidden or node.id.startswith("__")):
            raise ValueError("unsupported builtin")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("unsupported attribute")


def code_score(record, text):
    code = extract_code(text)
    try:
        validate_code(code)
    except (SyntaxError, ValueError) as error:
        return dict(correct=False, error=type(error).__name__ + ": " + str(error), metric="provided tests", code_sha256=sha(code.encode()))
    checks = record["test_list"] if record["benchmark"] == "mbpp" else [record["test"], "check(" + record["entry_point"] + ")"]
    limits = "import sys\nif sys.platform != 'win32':\n import resource\n resource.setrlimit(resource.RLIMIT_CPU,(2,2))\n resource.setrlimit(resource.RLIMIT_AS,(536870912,536870912))\n"
    program = limits + "\n" + record.get("test_setup_code", "") + "\n" + code + "\n" + "\n".join(checks)
    directory = Path.cwd() / "artifacts/asd-suite-code-eval"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="case-", dir=directory) as tmp:
            result = subprocess.run([sys.executable, "-I", "-c", program], cwd=tmp, env={}, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=3, check=False)
        errors = result.stderr.decode(errors="replace").strip().splitlines()
        return dict(correct=result.returncode == 0, error=errors[-1][:250] if errors and result.returncode else None,
                    metric="provided tests", code_sha256=sha(code.encode()))
    except subprocess.TimeoutExpired:
        return dict(correct=False, error="execution timeout", metric="provided tests", code_sha256=sha(code.encode()))


def exact_json(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact_json(actual[k], v) for k,v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact_json(a,b) for a,b in zip(actual,expected))
    return actual == expected


def json_score(text, expected):
    def pairs(items):
        result = {}
        for k,v in items:
            if k in result:
                raise ValueError("duplicate key")
            result[k] = v
        return result

    def invalid_constant(value):
        raise ValueError("non-finite JSON constant: " + value)

    value = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", value, re.S)
    if fenced:
        value = fenced[1]
    try:
        actual = json.loads(value, object_pairs_hook=pairs, parse_constant=invalid_constant)
        return dict(correct=exact_json(actual, expected), valid_json=True, metric="exact typed JSON contract", error=None)
    except (ValueError, TypeError) as error:
        return dict(correct=False, valid_json=False, metric="exact typed JSON contract", error=str(error)[:250])


def score(record, text):
    b = record["benchmark"]
    if b == "gsm8k":
        return old_score(record, text)
    if b == "math500":
        result = math_score(record, text)
        actual = boxed_answer(text)
        x, y = scalar_answer(actual) if actual is not None else None, scalar_answer(record["answer"])
        numeric = x == y if x is not None and y is not None else None
        result.update(scalar_equivalent=numeric, secondary_correct=result["correct"] or numeric is True,
                      needs_review=not result["correct"] and numeric is None)
        return result
    if b in {"mbpp", "humaneval"}:
        return code_score(record, text)
    if b == "squad":
        return qa_score(text, record["answers"])
    if b == "json":
        return json_score(text, record["expected"])
    raise ValueError("Unknown benchmark")


def canonical(record):
    b = record["benchmark"]
    if b == "humaneval":
        return record["prompt"] + record["canonical_solution"]
    if b == "mbpp":
        return record["code"]
    if b == "gsm8k":
        return record["answer"]
    if b == "math500":
        return "\\boxed{" + record["answer"] + "}"
    if b == "squad":
        return record["answers"][0]
    return json.dumps(record["expected"], ensure_ascii=False)
