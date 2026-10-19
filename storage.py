import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteStorage:
    def __init__(self, database_path: str | Path | None = None) -> None:
        configured_path = database_path or os.getenv("SQLITE_DATABASE_PATH", "data/rag_app.sqlite3")
        self.database_path = Path(configured_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_thread_id_message_id
                    ON messages(thread_id, message_id);
                CREATE TABLE IF NOT EXISTS chunk_cache (
                    cache_key TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    PRIMARY KEY (cache_key, chunk_index)
                );
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                );
                """
            )

    def create_thread(self, thread_id: str, title: str = "New conversation") -> None:
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO threads (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (thread_id, title, now, now),
            )

    def list_threads(self) -> list[dict[str, str]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT thread_id, title, created_at, updated_at FROM threads ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_thread(self, thread_id: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role}")
        now = _utc_now()
        with self._connection() as connection:
            thread = connection.execute(
                "SELECT title FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if thread is None:
                raise ValueError(f"Conversation thread does not exist: {thread_id}")
            if role == "user" and thread["title"] == "New conversation":
                title = " ".join(content.split())[:80] or "New conversation"
                connection.execute(
                    "UPDATE threads SET title = ?, updated_at = ? WHERE thread_id = ?",
                    (title, now, thread_id),
                )
            else:
                connection.execute(
                    "UPDATE threads SET updated_at = ? WHERE thread_id = ?", (now, thread_id)
                )
            connection.execute(
                """
                INSERT INTO messages (thread_id, role, content, sources_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (thread_id, role, content, json.dumps(sources or []), now),
            )

    def get_messages(self, thread_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT role, content, sources_json, created_at
                FROM messages WHERE thread_id = ? ORDER BY message_id
                """,
                (thread_id,),
            ).fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "sources": json.loads(row["sources_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_cached_chunks(self, cache_key: str) -> list[str] | None:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT content FROM chunk_cache WHERE cache_key = ? ORDER BY chunk_index",
                (cache_key,),
            ).fetchall()
        return [row["content"] for row in rows] if rows else None

    def cache_chunks(self, cache_key: str, chunks: list[str]) -> None:
        with self._connection() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO chunk_cache (cache_key, chunk_index, content) VALUES (?, ?, ?)",
                [(cache_key, index, content) for index, content in enumerate(chunks)],
            )

    @staticmethod
    def embedding_cache_key(model: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{model}:{digest}"

    def get_cached_embeddings(self, model: str, texts: list[str]) -> dict[str, list[float]]:
        keys = {text: self.embedding_cache_key(model, text) for text in texts}
        if not keys:
            return {}
        cache_keys = list(keys.values())
        cached: dict[str, list[float]] = {}
        with self._connection() as connection:
            for offset in range(0, len(cache_keys), 500):
                batch = cache_keys[offset : offset + 500]
                rows = connection.execute(
                    f"SELECT cache_key, vector_json FROM embedding_cache WHERE cache_key IN ({','.join('?' for _ in batch)})",
                    batch,
                ).fetchall()
                cached.update(
                    {row["cache_key"]: json.loads(row["vector_json"]) for row in rows}
                )
        return {text: cached[key] for text, key in keys.items() if key in cached}

    def cache_embeddings(self, model: str, embeddings: dict[str, list[float]]) -> None:
        with self._connection() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache (cache_key, model, vector_json)
                VALUES (?, ?, ?)
                """,
                [
                    (self.embedding_cache_key(model, text), model, json.dumps(vector))
                    for text, vector in embeddings.items()
                ],
            )
