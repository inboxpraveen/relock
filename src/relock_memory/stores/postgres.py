"""Postgres store (``pip install relock-memory[postgres]``).

Uses JSONB for structured columns and TIMESTAMPTZ for times, so the ledger can
be queried directly, e.g. "which items were rewritten from verifier failures
last week?"
"""

from __future__ import annotations

from typing import Any

from .sql import SQLStore


class PostgresStore(SQLStore):
    placeholder = "%s"
    types = {"bool": "BOOLEAN", "false": "FALSE", "ts": "TIMESTAMPTZ", "json": "JSONB", "real": "DOUBLE PRECISION"}

    def __init__(self, dsn: str, **connect_kwargs: Any) -> None:
        super().__init__()
        try:
            import psycopg  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError("PostgresStore needs psycopg: pip install 'relock-memory[postgres]'") from e
        self.dsn = dsn
        self.connect_kwargs = connect_kwargs
        self.init()

    def __repr__(self) -> str:
        return "PostgresStore(<dsn>)"

    def _connect(self) -> Any:
        import psycopg

        return psycopg.connect(self.dsn, **self.connect_kwargs)

    def drop_all(self) -> None:
        """Remove every relock table (used by tests)."""
        with self._tx() as cur:
            for t in ("events", "errors", "expectations", "retrievals", "episodes", "versions", "items", "meta"):
                cur.execute(f"DROP TABLE IF EXISTS relock_{t} CASCADE")
