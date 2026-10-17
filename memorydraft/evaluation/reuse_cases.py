"""Small synthetic development cases, declared before measuring reuse performance."""
from __future__ import annotations


SYSTEM = "Follow the requested artifact format exactly. Do not add explanations."


def development_projects():
    projects = []
    for i in range(3):
        name = f"reuse-code-{i}"
        lines = [f"def check_{word}(value):\n    return value >= {10 + j}\n"
                 for j, word in enumerate(("age", "score", "stock", "credit", "quota", "level"))]
        artifact = "\n".join(lines)
        past = f"Return this entire Python module exactly, without markdown fences:\n{artifact}"
        current = (f"Return the entire updated Python module, without markdown fences. Change only the threshold "
                   f"in check_score from 11 to {21 + i}. Preserve all other code and blank lines.\n{artifact}")
        projects.append(dict(id=name, scenario="P1_local_file", past=past, currents=[current],
                             prompt_contains_old_artifact=True))
    for i in range(3):
        name = f"reuse-revision-{i}"
        facts = (f"The project is delta_{i}. It has services alpha, bravo, charlie, delta, echo and foxtrot. "
                 "Each service has fields name, enabled=true, timeout_seconds=30, retries=3 and log_level=info. "
                 "Use that service order and that field order. The top-level fields are project and services. "
                 "Use two-space indented JSON, with one field per line, without markdown.")
        past = "Generate the complete deployment configuration. " + facts
        currents = ["Generate the complete deployment configuration. " + facts +
                    f" Set only bravo's timeout_seconds to {40 + i + 10 * turn}."
                    for turn in range(3)]
        projects.append(dict(id=name, scenario="P2_sequential_config", past=past, currents=currents,
                             prompt_contains_old_artifact=False))
    for i in range(3):
        name = f"reuse-template-{i}"
        sections = ("Overview", "Inputs", "Validation", "Execution", "Monitoring", "Recovery")
        artifact = "\n\n".join(f"## {title}\nOwner: operations\nReview interval: weekly\nStatus: approved"
                                for title in sections)
        past = f"Return this runbook exactly, preserving its Markdown:\n{artifact}"
        current = (f"Return the entire updated runbook, preserving all unaffected text. "
                   f"In the Monitoring section only, change Review interval to every {2 + i} days.\n{artifact}")
        projects.append(dict(id=name, scenario="P3_document_template", past=past, currents=[current],
                             prompt_contains_old_artifact=True))
    boundary = [
        ("Write a Python function returning a list in reverse order.", "Reply with exactly: ready", 32),
        ("Describe a quiet garden in three sentences.", "Explain why the sky appears blue in two sentences.", 96),
        ("Return exactly this file:\nlimit = 10\nmode = 'safe'\n",
         "Output only a unified diff changing limit=10 to limit=20 in config.py. "
         "The original file is:\nlimit = 10\nmode = 'safe'\n", 96),
    ]
    for i, (past, current, budget) in enumerate(boundary):
        projects.append(dict(id=f"reuse-boundary-{i}", scenario="boundary", past=past, currents=[current],
                             prompt_contains_old_artifact=i == 2, budget=budget))
    return projects


def messages(text):
    return [dict(role="system", content=SYSTEM), dict(role="user", content=text)]
