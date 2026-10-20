"""Small public GSM8K/MBPP slice; scoring never supplies generation memory."""
from __future__ import annotations

import ast
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from memorydraft.runtime.contracts import GenerationRequest


def load_tasks(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    records = value["records"]
    expected = [*(f"gsm8k:{i}" for i in range(8)), *(f"mbpp:{i}" for i in range(11, 19))]
    if value["schema"] != 1 or [r["id"] for r in records] != expected:
        raise ValueError("Expected the frozen first-eight public slices")
    return records


def request_for(record, *, max_new_tokens=512):
    if record["benchmark"] == "gsm8k":
        prompt = "Solve the following math problem step by step. End with a line '#### <answer>'.\n\n" + record["question"]
    elif record["benchmark"] == "mbpp":
        prompt = ("Write Python code for the following task. Define the function used in these tests. "
                  "Return only the implementation, without test calls or explanation.\n\n" + record["text"]
                  + "\n\nYour code should pass these tests:\n" + "\n".join(record["test_list"]))
    else:
        raise ValueError("Unknown benchmark")
    return GenerationRequest(record["id"], "asd-bench-" + record["id"], 1,
                             ({"role": "user", "content": prompt},), max_new_tokens)


def numeric_answer(text):
    candidates = re.findall(r"####\s*(?:<answer>\s*)?(?:\*\*)?\\?\$?\s*(-?[\d,]+(?:\.\d+)?)", text, re.I)
    if not candidates:
        candidates = re.findall(r"\\boxed\{\s*(?:\\?\$)?\s*(-?[\d,]+(?:\.\d+)?)\s*\}", text)
    if not candidates:
        return None
    try:
        return Decimal(candidates[-1].replace(",", ""))
    except InvalidOperation:
        return None


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return "\n".join(blocks).strip() if blocks else text.strip().removeprefix("[BEGIN]").split("[DONE]")[0].strip()


def validate_code(code):
    """Restrict this benchmark harness to ordinary algorithmic Python."""
    tree = ast.parse(code)
    allowed = {"math", "collections", "itertools", "functools", "re", "heapq", "bisect", "string",
               "typing", "statistics", "operator", "decimal", "fractions"}
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


def score(record, text):
    if record["benchmark"] == "gsm8k":
        actual, expected = numeric_answer(text), numeric_answer(record["answer"])
        if expected is None:
            raise ValueError("Invalid frozen GSM8K answer")
        return dict(correct=actual == expected, parsed_answer=str(actual) if actual is not None else None,
                    expected_answer=str(expected), error=None if actual is not None else "missing final numeric answer")
    code = extract_code(text)
    try:
        validate_code(code)
    except (SyntaxError, ValueError) as error:
        return dict(correct=False, error=type(error).__name__ + ": " + str(error), code_sha256=hashlib.sha256(code.encode()).hexdigest())
    # Fresh process: no model state, credentials or caller environment. Bounded,
    # task-specific execution; this is not a general-purpose Python sandbox.
    limits = "import sys\nif sys.platform != 'win32':\n import resource\n resource.setrlimit(resource.RLIMIT_CPU,(2,2))\n resource.setrlimit(resource.RLIMIT_AS,(536870912,536870912))\n"
    program = limits + "\n" + record.get("test_setup_code", "") + "\n" + code + "\n" + "\n".join(record["test_list"])
    temporary_root = Path.cwd() / "artifacts/mbpp-eval"
    temporary_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="case-", dir=temporary_root) as directory:
            result = subprocess.run([sys.executable, "-I", "-c", program], cwd=directory, env={},
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    timeout=3, check=False)
        error = result.stderr.decode(errors="replace").strip().splitlines()
        return dict(correct=result.returncode == 0, error=error[-1][:250] if error else None,
                    code_sha256=hashlib.sha256(code.encode()).hexdigest())
    except subprocess.TimeoutExpired:
        return dict(correct=False, error="execution timeout", code_sha256=hashlib.sha256(code.encode()).hexdigest())
