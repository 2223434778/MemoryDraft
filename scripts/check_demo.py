"""Separate-process persistence and no-early-write check for the synthetic demo."""
import json
from pathlib import Path

from memorydraft.memory.store import MemoryStore


answer = json.loads(Path("results/demo/answer.json").read_text())
with MemoryStore("demo.db") as store:
    before = store.snapshot("code", 30)
    after = store.snapshot("code", 31)
    assert not any(r.record_id == "code-0:assistant" for r in before.records)
    finalized = next(r for r in after.records if r.record_id == "code-0:assistant")
    assert finalized.text == answer["text"]
    assert not any(r.record_id == "code-0:assistant" for r in store.snapshot("other-user", 31).records)
Path("results/demo/persistence.json").write_text(json.dumps({"persisted": True, "no_early_write": True,
                                                          "user_isolation": True}, indent=2) + "\n")
print("Final-answer persistence and isolation checks passed")
