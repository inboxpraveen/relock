"""Postgres store (``pip install relock-memory[postgres]``).

Uses JSONB for structured columns and TIMESTAMPTZ for times, so the ledger can
be queried directly, e.g. "which items were rewritten from verifier failures
last week?"

Connections come from a small thread-safe pool, so concurrent episodes don't queue
behind one connection. A connection that sat idle is checked before use, and one the
server dropped is discarded, so a database restart costs at most one failed call.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from ..types import StoreError
from .sql import SQLStore

_INIT_LOCK_KEY = 0x72656C6B  # "relk": serialises schema creation across processes


class PostgresStore(SQLStore):
    placeholder = "%s"
    types = {"bool": "BOOLEAN", "false": "FALSE", "ts": "TIMESTAMPTZ", "json": "JSONB", "real": "DOUBLE PRECISION"}

    def __init__(self, dsn: str, *, pool_size: int = 5, pool_timeout: float = 30.0, ping_after: float = 30.0,
                 **connect_kwargs: Any) -> None:
        """
        Args:
            dsn: a ``postgresql://`` URL or libpq connection string.
            pool_size: most connections open at once. Each in-flight store call holds one.
            pool_timeout: seconds to wait for a free connection before raising :class:`StoreError`.
            ping_after: a connection idle longer than this is checked with ``SELECT 1`` before use.
            connect_kwargs: passed to ``psycopg.connect``. ``connect_timeout`` defaults to 10 seconds
                unless the DSN sets one, so an unreachable host fails fast instead of hanging.
        """
        super().__init__()
        try:
            import psycopg
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError("PostgresStore needs psycopg: pip install 'relock-memory[postgres]'") from e
        if pool_size < 1:
            raise ValueError("pool_size must be at least 1")
        self.driver_errors = (psycopg.Error,)
        self.integrity_errors = (psycopg.errors.UniqueViolation,)
        self.dsn = dsn
        self.pool_size = pool_size
        self.pool_timeout = pool_timeout
        self.ping_after = ping_after
        if "connect_timeout" not in dsn:
            connect_kwargs.setdefault("connect_timeout", 10)
        self.connect_kwargs = connect_kwargs
        self._cv = threading.Condition()
        self._idle: list[tuple[Any, float, int]] = []  # (connection, last used, generation)
        self._gen: dict[int, int] = {}  # id(connection) -> generation it was opened in
        self._generation = 0
        self._open = 0
        self.init()

    def __repr__(self) -> str:
        return "PostgresStore(<dsn>)"

    def _connect(self) -> Any:
        import psycopg

        return psycopg.connect(self.dsn, **self.connect_kwargs)

    # ------------------------------------------------------------------ pool
    def _acquire(self) -> Any:
        deadline = time.monotonic() + self.pool_timeout
        with self._cv:
            while True:
                if self._idle:
                    conn, last, _gen = self._idle.pop()
                    break
                if self._open < self.pool_size:
                    self._open += 1
                    conn, last = None, 0.0
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StoreError(f"{self!r}: no database connection became free within {self.pool_timeout}s "
                                     f"(pool_size={self.pool_size}). Raise pool_size or reduce concurrency.")
                self._cv.wait(remaining)
        try:
            if conn is not None and not self._usable(conn, last):
                self._discard(conn)
                conn = None
            if conn is None:
                conn = self._open_connection()
                with self._cv:
                    self._gen[id(conn)] = self._generation
            return conn
        except BaseException:
            with self._cv:
                self._open -= 1
                self._cv.notify()
            raise

    def _usable(self, conn: Any, last_used: float) -> bool:
        if conn.closed or getattr(conn, "broken", False):
            return False
        if time.monotonic() - last_used < self.ping_after:
            return True
        try:
            conn.execute("SELECT 1")
            conn.rollback()
            return True
        except Exception:
            return False

    def _release(self, conn: Any, *, broken: bool) -> None:
        with self._cv:
            stale = self._gen.get(id(conn)) != self._generation
            if broken or stale or conn.closed:
                self._gen.pop(id(conn), None)
                self._discard(conn)
                self._open -= 1
            else:
                self._idle.append((conn, time.monotonic(), self._generation))
            self._cv.notify()

    def _is_disconnect(self, exc: BaseException, conn: Any) -> bool:
        return bool(conn.closed or getattr(conn, "broken", False))

    def _init_lock(self, cur: Any) -> None:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_INIT_LOCK_KEY,))

    def _table_exists(self, cur: Any, table: str) -> bool:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (table,))
        return bool(cur.fetchone()[0])

    def _has_column(self, cur: Any, table: str, column: str) -> bool:
        cur.execute("SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() "
                    "AND table_name = %s AND column_name = %s", (table, column))
        return cur.fetchone() is not None

    def close(self) -> None:
        """Close idle connections. Connections in use close when they are returned. Reusable."""
        with self._cv:
            self._generation += 1
            idle, self._idle = self._idle, []
            for conn, _last, _gen in idle:
                self._gen.pop(id(conn), None)
                self._discard(conn)
                self._open -= 1
            self._cv.notify_all()

    def pool_stats(self) -> dict[str, int]:
        with self._cv:
            return {"open": self._open, "idle": len(self._idle), "size": self.pool_size}

    def drop_all(self) -> None:
        """Remove every relock table (used by tests)."""
        with self._tx() as cur:
            for t in ("events", "errors", "expectations", "retrievals", "episodes", "versions", "items", "meta"):
                cur.execute(f"DROP TABLE IF EXISTS relock_{t} CASCADE")
