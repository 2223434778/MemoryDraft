"""Fetch a pinned public baseline and create declared synthetic history controls."""
import hashlib
import json
import urllib.request
from pathlib import Path


root = Path(__file__).resolve().parents[1] / "data/research"
root.mkdir(parents=True, exist_ok=True)
commit = "fd2c1cd7d2201ef71db4c5f4e455008f017967bf"
base = f"https://raw.githubusercontent.com/hemingkx/Spec-Bench/{commit}/"


def write(name, rows):
    (root / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    raw = urllib.request.urlopen(base + "data/spec_bench/question.jsonl", timeout=30).read()
    license_text = urllib.request.urlopen(base + "LICENSE", timeout=30).read()
    (root / "SPEC_BENCH_LICENSE").write_bytes(license_text)
    original = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    categories = ["coding", "summarization", "qa", "rag", "math_reasoning", "translation"]
    for split, begin, end in [("dev", 0, 2), ("test", 2, 6)]:
        selected = []
        for category in categories:
            candidates = [r for r in original if r["category"] == category][begin:end]
            selected.extend(dict(id=f"spec-{r['question_id']}", user_id=f"spec-{r['question_id']}",
                                 order=10, scenario="specbench-" + category,
                                 messages=[dict(role="user", content=r["turns"][0])]) for r in candidates)
        write(f"specbench-{split}.jsonl", selected)
    manifest = dict(repository="hemingkx/Spec-Bench", commit=commit,
                    source=base + "data/spec_bench/question.jsonl", sha256=hashlib.sha256(raw).hexdigest(),
                    license="Upstream repository Apache-2.0; original tasks retain their source terms.",
                    selection="Upstream order within six categories; first 2 dev, next 4 test; first turn only.")
    (root / "specbench-source.json").write_text(json.dumps(manifest, indent=2) + "\n")

    history, requests = [], []
    for split in ("dev", "test"):
        for i, task in enumerate(("copy", "edit-code", "edit-doc", "copy", "edit-code", "new-task")):
            user = f"{split}-project-{i}"
            name = f"read_{split}_items_{i}"
            code = (f"def {name}(items, limit=10):\n"
                    '    """Return normalized labels from the first nonempty items."""\n'
                    "    labels = []\n    for item in items:\n        label = item.strip().lower()\n"
                    "        if not label:\n            continue\n        labels.append(label)\n"
                    "        if len(labels) >= limit:\n            break\n    return labels\n")
            doc = (f"Project {user} release checklist\n" + "\n".join([
                "1. Confirm the deployment window with the service owner.",
                "2. Export the current configuration and keep the rollback copy.",
                "3. Run unit tests and integration tests in the staging environment.",
                "4. Deploy the new package to a single instance and check health probes.",
                "5. Review error rates and request latency for ten minutes.",
                "6. Roll out the package to the remaining instances in small batches.",
                "7. Record the package version, deployment time and verification results.",
                "8. Notify the service owner that the deployment is complete.",
            ]))
            old = code if task == "edit-code" else doc
            history.append(dict(user_id=user, record_id=user + "-old", order=1, text=old, project_id=user))
            for j, distractor in enumerate([
                "The botanical garden contains orchids, moss and ferns around a quiet pond.",
                "Meeting notes: the office kitchen closes at six and the library opens at nine.",
                "A different project builds reports using a database and a web dashboard.",
            ]):
                history.append(dict(user_id=user, record_id=f"{user}-distractor-{j}", order=2, text=distractor))
            if task == "copy":
                question = "Return exactly the checklist below, preserving every line. Do not add any explanation.\n\n" + old
                memory = f"Application memory: project identifier is {user}."
            elif task == "edit-code":
                memory = (f"Application memory: project {user}; function {name}(items, limit=10) returns a list. "
                          "It strips whitespace, skips empty labels, preserves input order, and stops at limit.")
                question = (f"Write the complete Python function {name}(items, limit=10). Normalize labels with upper() "
                            "instead of lower(). Use the variable names labels, item, label, and return labels. "
                            "Return Python only, with no markdown fences.")
            elif task == "edit-doc":
                memory = f"Application memory: {user} uses an eight-step release checklist covering preparation, backup, tests, canary, monitoring, rollout, records and notification."
                question = (f"Write the eight-step deployment checklist for {user}. The monitoring period is now twenty minutes. "
                            "Keep one sentence per numbered step, and begin with the project release checklist title.")
            else:
                memory = f"Application memory: project identifier is {user}."
                question = "Explain in three sentences how rainbows form."
            requests.append(dict(id=user, user_id=user, order=10, scenario=f"synthetic-B-{task}",
                                 selected_history_ids=[user + "-old"],
                                 messages=[dict(role="system", content="You are a helpful assistant.\n" + memory),
                                           dict(role="user", content=question)]))
    write("history.jsonl", history)
    write("memory-dev.jsonl", [r for r in requests if r["id"].startswith("dev-")])
    write("memory-test.jsonl", [r for r in requests if r["id"].startswith("test-")])
    print("Prepared fixed public splits and separately labelled synthetic setting-B controls.")
