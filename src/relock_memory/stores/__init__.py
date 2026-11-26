"""Storage backends. ``open_store`` turns a URL or path into a store."""

from __future__ import annotations

import os
from typing import Any

from .sql import SQLStore
from .sqlite import SQLiteStore

Store = SQLStore

__all__ = ["SQLStore", "SQLiteStore", "Store", "open_store"]


def open_store(target: str | SQLStore | None = None, **kwargs: Any) -> SQLStore:
    """Open a store from a path or URL.

    * ``None``                              -> ``$RELOCK_DB`` or ``./relock.db``
    * ``":memory:"``                        -> in-process SQLite (tests, notebooks)
    * ``"agent.db"`` / ``"sqlite:///x.db"`` -> SQLite file
    * ``"postgresql://user:pw@host/db"``    -> Postgres (needs the ``postgres`` extra)
    """
    if isinstance(target, SQLStore):
        return target
    target = target or os.environ.get("RELOCK_DB") or "relock.db"
    if target.startswith(("postgres://", "postgresql://")):
        from .postgres import PostgresStore

        return PostgresStore(target, **kwargs)
    if target.startswith("sqlite://"):
        target = target[len("sqlite://"):]
        if target.startswith("/"):  # sqlite:///relative.db, sqlite:////abs/path.db
            target = target[1:]
        target = target or ":memory:"
    return SQLiteStore(target, **kwargs)
