"""Create explicitly synthetic task streams; never use Target outputs as old memory."""
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1] / "data" / "synthetic"
root.mkdir(parents=True, exist_ok=True)
histories = [
    ("code", "The project used retries=3 and timeout=10. The function was: def connect(): return {'retries': 3, 'timeout': 10}"),
    ("docs", "Release note template: Summary: Improved reliability. Changes: Fixed retry handling. Verification: Unit tests passed."),
    ("project", "MemoryDraft uses R for historical token reuse, H for history-assisted drafting, and D for ordinary drafting."),
]
events = [dict(user_id=user, record_id=user + "-old", order=1, text=text, status="active", session_id="previous", source_role="assistant", project_id=user)
          for user, text in histories]
events += [dict(user_id="code", record_id="code-old", order=20, text=histories[0][1], status="superseded"),
           dict(user_id="code", record_id="code-new", order=20, text="Current project defaults are retries=5 and timeout=30.", status="active")]
requests = []
tasks = {
    "code": [("adapt", "Write def connect() returning retries=5 and timeout=30. Return Python only."),
             ("reuse", "Write def connect() returning retries=3 and timeout=10. Return Python only."),
             ("conflict", "The current retries is 7, timeout is 60. Write def connect() returning these values."),
             ("irrelevant", "What is photosynthesis? Answer in one short sentence.")],
    "docs": [("adapt", "Write a release note with sections Summary, Changes, Verification. The change adds exponential backoff; tests pass."),
             ("reuse", "Write exactly: Summary: Improved reliability. Changes: Fixed retry handling. Verification: Unit tests passed."),
             ("irrelevant", "What is 12 multiplied by 8? Answer only the number."),
             ("adapt", "Write a release note in three short sections: Summary, Changes, Verification. This release adds CSV export.")],
    "project": [("reuse", "Explain R, H and D in MemoryDraft in three short sentences."),
                ("adapt", "In MemoryDraft, explain why H needs its own KV cache in two sentences."),
                ("irrelevant", "Name two planets. Answer only their names."),
                ("adapt", "Explain why strict Target verification is needed in MemoryDraft. Use two sentences.")],
}
for user, cases in tasks.items():
    for i, (scenario, text) in enumerate(cases):
        requests.append(dict(id=f"{user}-{i}", user_id=user, order=10 + i, scenario="synthetic-" + scenario,
                             messages=[{"role": "system", "content": "You are a concise helpful assistant."}, {"role": "user", "content": text}]))
requests.append(dict(id="empty", user_id="empty", order=10, scenario="synthetic-empty",
                     messages=[{"role": "user", "content": "Write a Python function that returns 42."}]))
focused = [r for r in requests if r["id"] in {"code-0", "code-1", "code-2", "docs-2", "empty"}]
for name, rows in (("history.jsonl", events), ("requests.jsonl", requests), ("h_cases.jsonl", focused)):
    (root / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(root)
(root / "request.json").write_text(json.dumps(requests[0], indent=2), encoding="utf-8")
