"""SQLite store: the zero-setup default (``relock.db`` next to your agent)."""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from .sql import SQLStore


class SQLiteStore(SQLStore):
    placeholder = "?"
    types = {"bool": "INTEGER", "false": "0", "ts": "TEXT", "json": "TEXT", "real": "REAL"}

    def __init__(self, path: str = "relock.db", *, timeout: float = 30.0) -> None:
        super().__init__()
        self.path = path
        self.timeout = timeout
        if path not in ("", ":memory:"):
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
        self.init()

    def __repr__(self) -> str:
        return f"SQLiteStore({self.path!r})"

    def _connect(self) -> Any:
        conn = sqlite3.connect(self.path or ":memory:", timeout=self.timeout, check_same_thread=False,
                               isolation_level=None)
        conn.execute("PRAGMA foreign_keys = ON")
        if self.path not in ("", ":memory:"):
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _begin(self, cur: Any) -> None:
        cur.execute("BEGIN IMMEDIATE")
