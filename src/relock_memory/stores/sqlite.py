"""SQLite store: the zero-setup default (``relock.db`` next to your agent)."""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any

from .sql import SQLStore


class SQLiteStore(SQLStore):
    placeholder = "?"
    types = {"bool": "INTEGER", "false": "0", "ts": "TEXT", "json": "TEXT", "real": "REAL"}
    driver_errors = (sqlite3.Error,)
    integrity_errors = (sqlite3.IntegrityError,)

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
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            if self.path not in ("", ":memory:"):
                self._enable_wal(conn)
                conn.execute("PRAGMA synchronous = NORMAL")
        except BaseException:
            conn.close()
            raise
        return conn

    def _enable_wal(self, conn: Any) -> None:
        # Switching to WAL needs a moment of exclusive access and does not wait on the busy
        # timeout, so processes opening a new file together retry here. WAL then persists.
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                if str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "wal":
                    conn.execute("PRAGMA journal_mode = WAL")
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or time.monotonic() > deadline:
                    raise
                time.sleep(0.02)

    def _begin(self, cur: Any, write: bool) -> None:
        # Writes take the write lock up front, so two writers never deadlock upgrading a read
        # lock. Reads stay deferred, so they don't queue behind writers in other processes.
        cur.execute("BEGIN IMMEDIATE" if write else "BEGIN")
