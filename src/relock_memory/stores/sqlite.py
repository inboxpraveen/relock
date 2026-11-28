"""SQLite store: the zero-setup default (``relock.db`` next to your agent)."""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any

from ..types import StoreError
from .sql import SQLStore


class SQLiteStore(SQLStore):
    placeholder = "?"
    types = {"bool": "INTEGER", "false": "0", "ts": "TEXT", "json": "TEXT", "real": "REAL"}
    driver_errors = (sqlite3.Error,)
    integrity_errors = (sqlite3.IntegrityError,)

    def __init__(self, path: str | os.PathLike[str] = "relock.db", *, timeout: float = 30.0,
                 uri: bool | None = None) -> None:
        """
        Args:
            path: a file path, ``":memory:"``, or a SQLite ``file:`` URI.
            timeout: seconds to wait for another process's write lock.
            uri: open ``path`` as a URI; by default, when it starts with ``file:``.
        """
        super().__init__()
        self.path = os.fsdecode(path) if isinstance(path, (os.PathLike, bytes)) else path
        self.timeout = timeout
        self.uri = self.path.startswith("file:") if uri is None else uri
        if not self.uri and self.path not in ("", ":memory:"):
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            except OSError as e:  # a file where a directory should be, no permission...
                raise StoreError(f"{self!r}: cannot create the database's directory: "
                                 f"{type(e).__name__}: {e}") from e
        self.init()

    def __repr__(self) -> str:
        return f"SQLiteStore({self.path!r})"

    def _connect(self) -> Any:
        conn = sqlite3.connect(self.path or ":memory:", timeout=self.timeout, check_same_thread=False,
                               isolation_level=None, uri=self.uri)
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

    def _is_conflict(self, exc: BaseException) -> bool:
        # sqlite3 raises IntegrityError for every constraint; only unique keys mean a lost race.
        return "UNIQUE constraint failed" in str(exc)

    def _begin(self, cur: Any, write: bool) -> None:
        # Writes take the write lock up front, so two writers never deadlock upgrading a read
        # lock. Reads stay deferred, so they don't queue behind writers in other processes.
        cur.execute("BEGIN IMMEDIATE" if write else "BEGIN")
