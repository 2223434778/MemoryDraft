from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryEvent:
    user_id: str
    record_id: str
    order: int
    text: str
    status: str = "active"
    session_id: str = ""
    source_role: str = "assistant"
    project_id: str = ""

    def __post_init__(self):
        if not self.user_id or not self.record_id or self.order < 0:
            raise ValueError("Memory requires user/record IDs and nonnegative availability order")
        if self.status not in {"active", "superseded", "deleted"}:
            raise ValueError("Unknown memory status")
        if self.source_role not in {"assistant", "user"}:
            raise ValueError("Only finalized user/assistant text may enter memory")

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    user_id: str
    before_order: int
    records: tuple[MemoryEvent, ...]

    @property
    def snapshot_id(self) -> str:
        payload = [self.user_id, self.before_order, [asdict(r) for r in self.records]]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class MemoryStore:
    """order is a dataset's explicit availability sequence, not creation time.

    Versions and tombstones are append-only events. Corrections must append both
    the old record's supersession and the new version at their actual availability
    order. Snapshot selection never consults a record's future status.
    """

    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path))
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (
            user_id TEXT, record_id TEXT, available_order INTEGER, payload TEXT NOT NULL,
            PRIMARY KEY(user_id, record_id, available_order))""")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def append(self, events: list[MemoryEvent]) -> None:
        with self.db:
            for event in events:
                payload = json.dumps(asdict(event), sort_keys=True, ensure_ascii=False)
                old = self.db.execute(
                    "SELECT payload FROM events WHERE user_id=? AND record_id=? AND available_order=?",
                    (event.user_id, event.record_id, event.order),
                ).fetchone()
                if old and old[0] != payload:
                    raise ValueError("Cannot overwrite an immutable memory event")
                self.db.execute("INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?)",
                                (event.user_id, event.record_id, event.order, payload))

    def snapshot(self, user_id: str, before_order: int) -> Snapshot:
        rows = self.db.execute("""SELECT payload FROM events e WHERE user_id=?
            AND available_order < ? AND available_order = (
                SELECT MAX(available_order) FROM events v WHERE v.user_id=e.user_id
                AND v.record_id=e.record_id AND v.available_order < ?)
            ORDER BY record_id""", (user_id, before_order, before_order)).fetchall()
        records = tuple(MemoryEvent(**json.loads(row[0])) for row in rows)
        return Snapshot(user_id, before_order, tuple(r for r in records if r.status == "active"))

    def purge(self, user_id: str, record_id: str) -> None:
        """Physical erasure changes all future replays; there is no persistent vector cache."""
        with self.db:
            self.db.execute("DELETE FROM events WHERE user_id=? AND record_id=?", (user_id, record_id))

    def import_jsonl(self, path: str | Path) -> int:
        events = [MemoryEvent(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.append(events)
        return len(events)
