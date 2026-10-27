from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import sqlite3
import time
import uuid

from memorydraft.chat.security import token_hash


class StoreError(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message
        super().__init__(message)


class ChatStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL,
                    created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    csrf TEXT NOT NULL, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    title TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS conversations_owner ON conversations(user_id, updated);
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id)
                    ON DELETE CASCADE, turn_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
                    status TEXT NOT NULL, created REAL NOT NULL, metadata TEXT NOT NULL DEFAULT '{}');
                CREATE INDEX IF NOT EXISTS message_order ON messages(conversation_id, created);
                CREATE TABLE IF NOT EXISTS active (
                    user_id TEXT PRIMARY KEY REFERENCES users(id),
                    conversation_id TEXT UNIQUE NOT NULL, assistant_id TEXT UNIQUE NOT NULL);
            """)
            # A single worker owns streaming tasks; unfinished answers survive restarts.
            db.execute("UPDATE messages SET status='interrupted' WHERE status='generating'")
            db.execute("DELETE FROM active")
        if os.name != "nt":
            self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=5000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_user(self, username, password):
        user = {"id": uuid.uuid4().hex, "username": username}
        try:
            with self.connect() as db:
                db.execute("INSERT INTO users VALUES (?, ?, ?, ?)",
                           (user["id"], username, password, time.time()))
        except sqlite3.IntegrityError as error:
            raise StoreError(409, "用户名已被使用") from error
        return user

    def find_user(self, username):
        with self.connect() as db:
            row = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
            return dict(row) if row else None

    def new_session(self, user_id):
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))
            db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)",
                       (token_hash(token), user_id, csrf, time.time() + 7 * 86400))
        return token, csrf

    def session(self, token):
        with self.connect() as db:
            row = db.execute("SELECT u.id,u.username,s.csrf FROM sessions s JOIN users u ON u.id=s.user_id "
                             "WHERE s.token=? AND s.expires>?", (token_hash(token), time.time())).fetchone()
            return dict(row) if row else None

    def logout(self, token):
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE token=?", (token_hash(token),))

    def conversations(self, user_id):
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT id,title,created,updated FROM conversations WHERE user_id=? ORDER BY updated DESC LIMIT 200",
                (user_id,))]

    @staticmethod
    def owner(db, user_id, conversation_id):
        row = db.execute("SELECT * FROM conversations WHERE id=? AND user_id=?",
                         (conversation_id, user_id)).fetchone()
        if not row:
            raise StoreError(404, "会话不存在")
        return dict(row)

    def new_conversation(self, user_id):
        conversation_id, now = uuid.uuid4().hex, time.time()
        with self.connect() as db:
            db.execute("INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
                       (conversation_id, user_id, "新对话", now, now))
        return {"id": conversation_id, "title": "新对话"}

    def conversation(self, user_id, conversation_id):
        with self.connect() as db:
            result = self.owner(db, user_id, conversation_id)
            result["messages"] = [dict(row) for row in db.execute(
                "SELECT id,role,content,status,created,metadata FROM messages WHERE conversation_id=? "
                "ORDER BY created,rowid", (conversation_id,))]
            for message in result["messages"]:
                message["metadata"] = json.loads(message["metadata"])
            return result

    def delete_conversation(self, user_id, conversation_id):
        with self.connect() as db:
            self.owner(db, user_id, conversation_id)
            if db.execute("SELECT 1 FROM active WHERE conversation_id=?", (conversation_id,)).fetchone():
                raise StoreError(409, "请先停止当前回答")
            db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))

    def fork_conversation(self, user_id, conversation_id, message_id, content=None):
        new_id, now = uuid.uuid4().hex, time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            source = self.owner(db, user_id, conversation_id)
            if db.execute("SELECT 1 FROM active WHERE user_id=?", (user_id,)).fetchone():
                raise StoreError(409, "请先停止当前回答，再编辑或回退")
            rows = list(db.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY created,rowid",
                                   (conversation_id,)))
            index = next((i for i, row in enumerate(rows) if row["id"] == message_id), None)
            if index is None:
                raise StoreError(404, "消息不存在")
            chosen = rows[index]
            if chosen["status"] == "generating" or chosen["role"] not in {"user", "assistant"}:
                raise StoreError(409, "这条消息暂时不能编辑或回退")
            value = chosen["content"] if content is None else content
            if not isinstance(value, str) or not value.strip():
                raise StoreError(422, "消息内容不能为空")
            limit = 16384 if chosen["role"] == "user" else 262144
            if len(value) > limit:
                raise StoreError(422, "消息内容超过长度限制")
            title = source["title"].removesuffix(" · 分支")[:40] + " · 分支"
            db.execute("INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
                       (new_id, user_id, title, now, now))
            # A selected question returns to the composer; the normal send path creates its turn.
            prefix = rows[:index + (chosen["role"] == "assistant")]
            turn_ids = {}
            for row in prefix:
                turn_id = turn_ids.setdefault(row["turn_id"], uuid.uuid4().hex)
                text, status, metadata = row["content"], row["status"], row["metadata"]
                if row["id"] == message_id:
                    text, status = value, "complete"
                    if value != row["content"] or row["status"] != "complete":
                        # Edited/restored text is user-selected context, not a new model trace.
                        metadata = json.dumps({"history_edit": value != row["content"],
                                               "history_restored": row["status"] != "complete"})
                db.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (uuid.uuid4().hex, new_id, turn_id, row["role"], text, status,
                            row["created"], metadata))
        return {"id": new_id, "title": title, "prompt": value if chosen["role"] == "user" else None}

    def begin_turn(self, user_id, conversation_id, content, *, max_active=2):
        turn_id, assistant_id, now = uuid.uuid4().hex, uuid.uuid4().hex, time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.owner(db, user_id, conversation_id)
            if db.execute("SELECT 1 FROM active WHERE user_id=?", (user_id,)).fetchone():
                raise StoreError(409, "你已有一个回答正在生成，请先停止或等待完成")
            if db.execute("SELECT COUNT(*) FROM active").fetchone()[0] >= max_active:
                raise StoreError(429, "当前已有两人在使用，请稍后重试")
            history = [dict(row) for row in db.execute(
                "SELECT m.role,m.content FROM messages m WHERE m.conversation_id=? AND EXISTS "
                "(SELECT 1 FROM messages a WHERE a.conversation_id=m.conversation_id AND a.turn_id=m.turn_id "
                "AND a.role='assistant' AND a.status='complete') "
                "ORDER BY m.created,m.rowid", (conversation_id,))]
            for role, value, status, message_id in (("user", content, "complete", uuid.uuid4().hex),
                                                   ("assistant", "", "generating", assistant_id)):
                db.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, '{}')",
                           (message_id, conversation_id, turn_id, role, value, status, now))
            db.execute("INSERT INTO active VALUES (?, ?, ?)", (user_id, conversation_id, assistant_id))
            db.execute("UPDATE conversations SET title=CASE WHEN title='新对话' THEN ? ELSE title END,updated=? WHERE id=?",
                       (content.replace("\n", " ")[:40], now, conversation_id))
        return assistant_id, turn_id, history

    def save_partial(self, assistant_id, text):
        with self.connect() as db:
            db.execute("UPDATE messages SET content=? WHERE id=? AND status='generating'", (text, assistant_id))

    def finish(self, assistant_id, text, status, metadata):
        with self.connect() as db:
            db.execute("UPDATE messages SET content=?,status=?,metadata=? WHERE id=?",
                       (text, status, json.dumps(metadata, ensure_ascii=False), assistant_id))
            db.execute("DELETE FROM active WHERE assistant_id=?", (assistant_id,))
            db.execute("UPDATE conversations SET updated=? WHERE id=(SELECT conversation_id FROM messages WHERE id=?)",
                       (time.time(), assistant_id))
