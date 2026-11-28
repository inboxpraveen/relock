"""Storage backends. ``open_store`` turns a URL or path into a store."""

from __future__ import annotations

import os
import re
from typing import Any

from .sql import SQLStore
from .sqlite import SQLiteStore

Store = SQLStore

__all__ = ["SQLStore", "SQLiteStore", "Store", "open_store"]

_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)://")
_POSTGRES_SCHEMES = frozenset({"postgres", "postgresql", "postgresql+psycopg", "postgresql+psycopg2"})
_SUPPORTED = ("a SQLite path, sqlite:///relative.db, sqlite:////absolute.db, a file: URI, or "
              "postgresql://user:password@host/db (postgres://, postgresql+psycopg:// and "
              "postgresql+psycopg2:// work too)")


def _pool_size_from_env() -> int | None:
    raw = os.environ.get("RELOCK_PG_POOL_SIZE", "").strip()
    if not raw:
        return None
    try:
        size = int(raw)
    except ValueError:
        size = 0
    if size < 1:
        raise ValueError(f"RELOCK_PG_POOL_SIZE must be a whole number of at least 1, not {raw!r}")
    return size


def open_store(target: str | os.PathLike[str] | SQLStore | None = None, **kwargs: Any) -> SQLStore:
    """Open a store from a path or URL.

    * ``None``                              -> ``$RELOCK_DB`` or ``./relock.db``
    * ``":memory:"`` / ``"sqlite://"``      -> in-process SQLite (tests, notebooks)
    * ``"agent.db"`` / ``Path("agent.db")`` -> SQLite file
    * ``"sqlite:///rel.db"`` / ``"sqlite:////abs/x.db"`` -> SQLite file
    * ``"file:x.db?mode=rwc"``              -> SQLite, opened as a URI
    * ``"postgresql://user:pw@host/db"``    -> Postgres (needs the ``postgres`` extra). ``postgres://``,
      ``postgresql+psycopg://`` and ``postgresql+psycopg2://`` are accepted too. The pool size is
      ``pool_size=`` or ``$RELOCK_PG_POOL_SIZE`` (default 5).

    Schemes are case-insensitive and surrounding whitespace is ignored. Any other ``scheme://``
    raises :class:`ValueError` rather than becoming a file name.
    """
    if isinstance(target, SQLStore):
        return target
    if isinstance(target, (os.PathLike, bytes)):
        target = os.fsdecode(target)
    target = (target or "").strip() or os.environ.get("RELOCK_DB", "").strip() or "relock.db"
    m = _SCHEME.match(target)
    # A one-letter "scheme" is a Windows drive (C://dir/x.db), not a URL.
    scheme = m.group(1).lower() if m and len(m.group(1)) > 1 else ""
    if m and scheme in _POSTGRES_SCHEMES:
        if "pool_size" not in kwargs:
            size = _pool_size_from_env()
            if size is not None:
                kwargs["pool_size"] = size
        from .postgres import PostgresStore

        return PostgresStore("postgresql://" + target[m.end():], **kwargs)
    if m and scheme == "sqlite":
        path = target[m.end():]
        if path.startswith("/"):  # sqlite:///relative.db, sqlite:////abs/path.db
            path = path[1:]
        return SQLiteStore(path or ":memory:", **kwargs)
    if target[:5].lower() == "file:":
        return SQLiteStore("file:" + target[5:], uri=True, **kwargs)
    if scheme:  # name only the scheme: the URL may carry a password
        raise ValueError(f"unsupported database URL scheme {scheme!r}: use {_SUPPORTED}")
    return SQLiteStore(target, **kwargs)
