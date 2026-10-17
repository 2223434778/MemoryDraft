"""Small, explicitly synthetic causal tasks and response-only distillation inputs."""
import torch

from memorydraft.retrieval.pool import LexicalEncoder, retrieve


COUNTS = {"train": 24, "dev": 8, "test": 8}
FAMILIES = ("labels", "numbers", "checklist", "config")


def task_family(messages):
    text = messages[-1]["content"]
    if "JSON configuration" in text:
        return "config"
    if "release checklist" in text:
        return "checklist"
    if "Python function" in text:
        return "labels" if "normalized" in text else "numbers"
    return "unknown"


def projects(split):
    if split not in COUNTS:
        raise ValueError("Unknown split")
    result = []
    for i in range(COUNTS[split]):
        uid = f"distill-{split}-{i:02d}"
        family = FAMILIES[i % len(FAMILIES)]
        function = f"prepare_{split}_{i}"
        if family == "labels":
            memory = f"Project {uid}: {function}(items) strips labels, skips empty labels and preserves order."
            old = f"Write the complete Python function {function}(items). Return the cleaned labels normalized with lower(). Use a for loop, variables result, item, label. Return Python only, no fences."
            current = old.replace("lower()", "upper()")
        elif family == "numbers":
            limit = 3 + i % 7
            memory = f"Project {uid}: {function}(values) returns a list of numeric values, preserving order."
            old = f"Write the complete Python function {function}(values). Keep only values greater than {limit}. Use a for loop and variables result and value. Return Python only, no fences."
            current = old.replace(f"greater than {limit}", f"greater than or equal to {limit + 2}")
        elif family == "checklist":
            minutes = 5 + i % 9
            memory = f"Project {uid}: release procedure has six steps: prepare, backup, tests, canary, monitoring, rollout."
            old = f"Write the six-step release checklist for {uid}. Monitor for {minutes} minutes. Begin with a project title, use numbered steps and one short sentence per step."
            current = old.replace(f"{minutes} minutes", f"{minutes + 5} minutes")
        else:
            timeout = 10 + i % 9
            memory = f"Project {uid}: configuration keys are project, network, retries. network contains timeout_seconds and tls."
            old = f"Write the complete JSON configuration for project {uid}, timeout_seconds={timeout}, tls=true, retries=2. Use the specified key order and two-space indentation. Return JSON only, no fences."
            current = old.replace(f"timeout_seconds={timeout}", f"timeout_seconds={timeout + 5}").replace("retries=2", "retries=4")
        system = "You are a helpful assistant.\nApplication memory: " + memory
        past = dict(id=uid + ":past", user_id=uid, order=0, family=family,
                    messages=[dict(role="system", content=system), dict(role="user", content=old)])
        request = dict(id=uid, user_id=uid, order=10, scenario=family,
                       messages=[dict(role="system", content=system), dict(role="user", content=current)])
        result.append(dict(past=past, request=request))
    return result


def short_history(backend, request, snapshot):
    bundle = retrieve(request["messages"][-1]["content"], snapshot, backend.tokenizer,
                      LexicalEncoder(), top_k=1, token_budget=128)
    return "\n---\n".join(backend.tokenizer.decode(chunk.token_ids) for chunk in bundle.chunks)


def response_sample(backend, messages, answer_ids, history):
    if not answer_ids:
        raise ValueError("Empty teacher response")
    prefix = backend.prompt(messages, history)[0].cpu()
    answer = torch.tensor(answer_ids, dtype=torch.long)
    ids = torch.cat((prefix, answer))
    labels = ids.clone()
    labels[:len(prefix)] = -100
    return dict(input_ids=ids.unsqueeze(0), attention_mask=torch.ones_like(ids).unsqueeze(0),
                labels=labels.unsqueeze(0))
