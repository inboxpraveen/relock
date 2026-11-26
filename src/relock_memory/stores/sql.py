"""SQL ledger shared by the SQLite and Postgres stores.

Tables (all prefixed ``relock_``):

* ``items``        - one row per memory item, pointing at its current version
* ``versions``     - every version ever written; a partial unique index keeps
                     exactly one ``active`` version per item
* ``episodes``     - reconsolidation windows (open -> closed)
* ``retrievals``   - the retrieval trace: which item versions were reactivated
* ``expectations`` - structured predictions made before acting
* ``errors``       - prediction errors with their admission status and weight
* ``events``       - an append-only audit log of every gate decision
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from .._util import dumps, loads, new_id, now, ts
from ..types import Clause, ConflictError, Item, NotFoundError

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS relock_meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS relock_items (
    id TEXT PRIMARY KEY,
    key TEXT UNIQUE,
    current_version INTEGER NOT NULL,
    archived {bool} NOT NULL DEFAULT {false},
    created_at {ts} NOT NULL,
    updated_at {ts} NOT NULL
);
CREATE TABLE IF NOT EXISTS relock_versions (
    item_id TEXT NOT NULL REFERENCES relock_items(id),
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    clauses {json} NOT NULL,
    clause_seq INTEGER NOT NULL,
    tags {json} NOT NULL,
    tools {json} NOT NULL,
    meta {json} NOT NULL,
    alpha {real} NOT NULL,
    beta {real} NOT NULL,
    confirmations INTEGER NOT NULL DEFAULT 0,
    failures INTEGER NOT NULL DEFAULT 0,
    parent_version INTEGER,
    superseded_by INTEGER,
    change_kind TEXT NOT NULL,
    delta {json},
    reason TEXT,
    episode_id TEXT,
    evidence {json} NOT NULL,
    author TEXT,
    created_at {ts} NOT NULL,
    PRIMARY KEY (item_id, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS relock_versions_one_active
    ON relock_versions (item_id) WHERE status = 'active';
CREATE TABLE IF NOT EXISTS relock_episodes (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    status TEXT NOT NULL,
    meta {json} NOT NULL,
    started_at {ts} NOT NULL,
    ended_at {ts}
);
CREATE TABLE IF NOT EXISTS relock_retrievals (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES relock_episodes(id),
    item_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    query TEXT NOT NULL,
    rank INTEGER NOT NULL,
    score {real} NOT NULL,
    at {ts} NOT NULL
);
CREATE INDEX IF NOT EXISTS relock_retrievals_episode ON relock_retrievals (episode_id);
CREATE TABLE IF NOT EXISTS relock_expectations (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES relock_episodes(id),
    item_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    statement TEXT NOT NULL,
    clause_ids {json} NOT NULL,
    tool TEXT,
    check_spec TEXT,
    auto {bool} NOT NULL DEFAULT {false},
    status TEXT NOT NULL,
    matched {bool},
    outcome {json},
    signal {json},
    created_at {ts} NOT NULL,
    resolved_at {ts}
);
CREATE INDEX IF NOT EXISTS relock_expectations_episode ON relock_expectations (episode_id);
CREATE TABLE IF NOT EXISTS relock_errors (
    id TEXT PRIMARY KEY,
    expectation_id TEXT NOT NULL REFERENCES relock_expectations(id),
    episode_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    source TEXT NOT NULL,
    error_class TEXT NOT NULL,
    weight {real} NOT NULL,
    authoritative {bool} NOT NULL DEFAULT {false},
    diff {json} NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    resolved_ref TEXT,
    created_at {ts} NOT NULL
);
CREATE INDEX IF NOT EXISTS relock_errors_item ON relock_errors (item_id, version, status);
CREATE TABLE IF NOT EXISTS relock_events (
    id TEXT PRIMARY KEY,
    at {ts} NOT NULL,
    type TEXT NOT NULL,
    episode_id TEXT,
    item_id TEXT,
    version INTEGER,
    data {json} NOT NULL
);
CREATE INDEX IF NOT EXISTS relock_events_item ON relock_events (item_id, at);
CREATE INDEX IF NOT EXISTS relock_events_episode ON relock_events (episode_id, at);
"""

_VERSION_COLS = (
    "item_id, version, status, title, kind, clauses, clause_seq, tags, tools, meta, alpha, beta, "
    "confirmations, failures, parent_version, superseded_by, change_kind, delta, reason, episode_id, "
    "evidence, author, created_at"
)


class SQLStore:
    """Dialect-neutral implementation. Subclasses provide a connection and type names."""

    placeholder = "?"
    types: dict[str, str] = {"bool": "INTEGER", "false": "0", "ts": "TEXT", "json": "TEXT", "real": "REAL"}

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._conn: Any = None

    # ------------------------------------------------------------------ plumbing
    def _connect(self) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def _sql(self, sql: str) -> str:
        return sql if self.placeholder == "?" else sql.replace("?", self.placeholder)

    @contextmanager
    def _tx(self) -> Iterator[Any]:
        with self._lock:
            if self._conn is None:
                self._conn = self._connect()
            cur = self._conn.cursor()
            try:
                self._begin(cur)
                yield cur
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def _begin(self, cur: Any) -> None:
        """Hook for dialects that need an explicit BEGIN."""

    def _exec(self, cur: Any, sql: str, params: Sequence[Any] = ()) -> Any:
        cur.execute(self._sql(sql), tuple(params))
        return cur

    def _rows(self, cur: Any) -> list[dict[str, Any]]:
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def init(self) -> None:
        ddl = _DDL.format(**self.types)
        with self._tx() as cur:
            for stmt in (s.strip() for s in ddl.split(";")):
                if stmt:
                    cur.execute(stmt)
            self._exec(cur, "SELECT v FROM relock_meta WHERE k = 'schema_version'")
            if cur.fetchone() is None:
                self._exec(cur, "INSERT INTO relock_meta (k, v) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
                self._exec(cur, "INSERT INTO relock_meta (k, v) VALUES ('revision', '0')")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _bump(self, cur: Any) -> None:
        self._exec(cur, "UPDATE relock_meta SET v = CAST(CAST(v AS INTEGER) + 1 AS TEXT) WHERE k = 'revision'")

    def revision(self) -> int:
        with self._tx() as cur:
            self._exec(cur, "SELECT v FROM relock_meta WHERE k = 'revision'")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    # ------------------------------------------------------------------ items
    def _item_from(self, row: dict[str, Any]) -> Item:
        return Item(
            id=row["item_id"],
            version=int(row["version"]),
            title=row["title"],
            clauses=tuple(Clause(c["id"], c["text"]) for c in loads(row["clauses"])),
            key=row.get("key"),
            kind=row["kind"],
            tags=tuple(loads(row["tags"])),
            tools=tuple(loads(row["tools"])),
            meta=loads(row["meta"]) or {},
            status=row["status"],
            alpha=float(row["alpha"]),
            beta=float(row["beta"]),
            confirmations=int(row["confirmations"]),
            failures=int(row["failures"]),
            parent_version=row["parent_version"],
            superseded_by=row["superseded_by"],
            change_kind=row["change_kind"],
            delta=loads(row["delta"]),
            reason=row["reason"],
            episode_id=row["episode_id"],
            evidence=tuple(loads(row["evidence"]) or ()),
            author=row["author"],
            clause_seq=int(row["clause_seq"]),
            created_at=ts(row["created_at"]) or "",
            archived=bool(row.get("archived", False)),
        )

    def _version_params(self, item: Item) -> tuple[Any, ...]:
        return (
            item.id, item.version, item.status, item.title, item.kind,
            dumps([c.to_dict() for c in item.clauses]), item.clause_seq, dumps(list(item.tags)),
            dumps(list(item.tools)), dumps(item.meta), item.alpha, item.beta, item.confirmations,
            item.failures, item.parent_version, item.superseded_by, item.change_kind,
            dumps(item.delta) if item.delta is not None else None, item.reason, item.episode_id,
            dumps(list(item.evidence)), item.author, item.created_at or now(),
        )

    def create_item(self, item: Item) -> Item:
        with self._tx() as cur:
            if item.key is not None:
                self._exec(cur, "SELECT id FROM relock_items WHERE key = ?", (item.key,))
                if cur.fetchone() is not None:
                    raise ConflictError(f"an item with key {item.key!r} already exists")
            t = item.created_at or now()
            self._exec(cur, "INSERT INTO relock_items (id, key, current_version, archived, created_at, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, ?)", (item.id, item.key, item.version, False, t, t))
            self._exec(cur, f"INSERT INTO relock_versions ({_VERSION_COLS}) VALUES ({', '.join('?' * 23)})",
                       self._version_params(item))
            self._bump(cur)
        return item

    def _select_versions(self, where: str) -> str:
        return (f"SELECT v.*, i.key AS key, i.archived AS archived FROM relock_versions v "
                f"JOIN relock_items i ON i.id = v.item_id WHERE {where}")

    def resolve(self, ref: str) -> str:
        """Map an id or key to an item id."""
        with self._tx() as cur:
            self._exec(cur, "SELECT id FROM relock_items WHERE id = ? OR key = ?", (ref, ref))
            row = cur.fetchone()
        if row is None:
            raise NotFoundError(f"no memory item {ref!r}")
        return row[0]

    def get(self, ref: str, version: int | None = None) -> Item:
        item_id = self.resolve(ref)
        with self._tx() as cur:
            if version is None:
                self._exec(cur, self._select_versions("v.item_id = ? AND v.version = i.current_version"), (item_id,))
            else:
                self._exec(cur, self._select_versions("v.item_id = ? AND v.version = ?"), (item_id, version))
            rows = self._rows(cur)
        if not rows:
            raise NotFoundError(f"{ref!r} has no version {version}")
        return self._item_from(rows[0])

    def active(self, include_archived: bool = False) -> list[Item]:
        where = "v.status = 'active'" + ("" if include_archived else " AND i.archived = ?")
        params: tuple[Any, ...] = () if include_archived else (False,)
        with self._tx() as cur:
            self._exec(cur, self._select_versions(where) + " ORDER BY v.item_id", params)
            return [self._item_from(r) for r in self._rows(cur)]

    def history(self, ref: str) -> list[Item]:
        item_id = self.resolve(ref)
        with self._tx() as cur:
            self._exec(cur, self._select_versions("v.item_id = ?") + " ORDER BY v.version", (item_id,))
            return [self._item_from(r) for r in self._rows(cur)]

    def commit_version(self, new: Item, *, expected_current: int, stale_reason: str) -> Item:
        """Atomically supersede ``expected_current`` with ``new`` (compare-and-swap).

        Pending errors against the superseded version are marked ``stale`` unless
        they were consumed as evidence for this rewrite (the caller does that).
        """
        with self._tx() as cur:
            self._exec(cur, "UPDATE relock_items SET current_version = ?, updated_at = ? "
                            "WHERE id = ? AND current_version = ?",
                       (new.version, now(), new.id, expected_current))
            if cur.rowcount != 1:
                raise ConflictError(f"{new.id} is no longer at version {expected_current}")
            self._exec(cur, "UPDATE relock_versions SET status = 'superseded', superseded_by = ? "
                            "WHERE item_id = ? AND version = ?", (new.version, new.id, expected_current))
            self._exec(cur, f"INSERT INTO relock_versions ({_VERSION_COLS}) VALUES ({', '.join('?' * 23)})",
                       self._version_params(new))
            if new.evidence:
                marks = ", ".join("?" * len(new.evidence))
                self._exec(cur, f"UPDATE relock_errors SET status = 'consumed', resolved_ref = ? "
                                f"WHERE id IN ({marks})", (f"v{new.version}", *new.evidence))
            self._exec(cur, "UPDATE relock_errors SET status = 'stale', reason = ? "
                            "WHERE item_id = ? AND version = ? AND status = 'pending'",
                       (stale_reason, new.id, expected_current))
            self._bump(cur)
        return new

    def next_version_number(self, item_id: str) -> int:
        with self._tx() as cur:
            self._exec(cur, "SELECT MAX(version) FROM relock_versions WHERE item_id = ?", (item_id,))
            row = cur.fetchone()
            return int(row[0] or 0) + 1

    def bump_confidence(self, item_id: str, version: int, *, d_alpha: float = 0.0, d_beta: float = 0.0,
                        d_confirm: int = 0, d_fail: int = 0) -> None:
        with self._tx() as cur:
            self._exec(cur, "UPDATE relock_versions SET alpha = alpha + ?, beta = beta + ?, "
                            "confirmations = confirmations + ?, failures = failures + ? "
                            "WHERE item_id = ? AND version = ?",
                       (d_alpha, d_beta, d_confirm, d_fail, item_id, version))

    def set_archived(self, ref: str, archived: bool) -> None:
        item_id = self.resolve(ref)
        with self._tx() as cur:
            self._exec(cur, "UPDATE relock_items SET archived = ?, updated_at = ? WHERE id = ?",
                       (archived, now(), item_id))
            self._bump(cur)

    # ------------------------------------------------------------------ episodes
    def create_episode(self, episode_id: str, task: str, meta: dict[str, Any]) -> None:
        with self._tx() as cur:
            self._exec(cur, "INSERT INTO relock_episodes (id, task, status, meta, started_at) VALUES (?, ?, 'open', ?, ?)",
                       (episode_id, task, dumps(meta), now()))

    def close_episode(self, episode_id: str) -> None:
        with self._tx() as cur:
            self._exec(cur, "UPDATE relock_episodes SET status = 'closed', ended_at = ? WHERE id = ?",
                       (now(), episode_id))
            self._exec(cur, "UPDATE relock_expectations SET status = 'void', resolved_at = ? "
                            "WHERE episode_id = ? AND status = 'open'", (now(), episode_id))

    def get_episode(self, episode_id: str) -> dict[str, Any]:
        with self._tx() as cur:
            self._exec(cur, "SELECT * FROM relock_episodes WHERE id = ?", (episode_id,))
            rows = self._rows(cur)
        if not rows:
            raise NotFoundError(f"no episode {episode_id!r}")
        r = rows[0]
        r["meta"] = loads(r["meta"])
        r["started_at"], r["ended_at"] = ts(r["started_at"]), ts(r["ended_at"])
        return r

    def add_retrievals(self, episode_id: str, query: str, hits: Sequence[tuple[str, int, int, float]]) -> None:
        if not hits:
            return
        t = now()
        with self._tx() as cur:
            for item_id, version, rank, score in hits:
                self._exec(cur, "INSERT INTO relock_retrievals (id, episode_id, item_id, version, query, rank, score, at) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (new_id("ret"), episode_id, item_id, version, query, rank, float(score), t))

    def retrieved(self, episode_id: str) -> dict[str, int]:
        """Item id -> most recently recalled version within the episode."""
        with self._tx() as cur:
            self._exec(cur, "SELECT item_id, MAX(version) FROM relock_retrievals WHERE episode_id = ? GROUP BY item_id",
                       (episode_id,))
            return {r[0]: int(r[1]) for r in cur.fetchall()}

    # ------------------------------------------------------------------ expectations
    def add_expectation(self, e: dict[str, Any]) -> None:
        with self._tx() as cur:
            self._exec(cur, "INSERT INTO relock_expectations (id, episode_id, item_id, version, statement, clause_ids, "
                            "tool, check_spec, auto, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)",
                       (e["id"], e["episode_id"], e["item_id"], e["version"], e["statement"],
                        dumps(list(e.get("clause_ids") or ())), e.get("tool"), e.get("check_spec"),
                        bool(e.get("auto")), now()))

    def get_expectation(self, expectation_id: str) -> dict[str, Any]:
        with self._tx() as cur:
            self._exec(cur, "SELECT * FROM relock_expectations WHERE id = ?", (expectation_id,))
            rows = self._rows(cur)
        if not rows:
            raise NotFoundError(f"no expectation {expectation_id!r}")
        r = rows[0]
        for k in ("clause_ids", "outcome", "signal"):
            r[k] = loads(r[k])
        r["auto"] = bool(r["auto"])
        r["matched"] = None if r["matched"] is None else bool(r["matched"])
        return r

    def resolve_expectation(self, expectation_id: str, *, matched: bool, outcome: dict[str, Any],
                            signal: dict[str, Any]) -> None:
        with self._tx() as cur:
            self._exec(cur, "UPDATE relock_expectations SET status = ?, matched = ?, outcome = ?, signal = ?, "
                            "resolved_at = ? WHERE id = ?",
                       ("confirmed" if matched else "violated", matched, dumps(outcome), dumps(signal), now(),
                        expectation_id))

    # ------------------------------------------------------------------ prediction errors
    def add_error(self, e: dict[str, Any]) -> None:
        with self._tx() as cur:
            self._exec(cur, "INSERT INTO relock_errors (id, expectation_id, episode_id, item_id, version, source, "
                            "error_class, weight, authoritative, diff, status, reason, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (e["id"], e["expectation_id"], e["episode_id"], e["item_id"], e["version"], e["source"],
                        e["error_class"], float(e["weight"]), bool(e.get("authoritative")), dumps(e["diff"]),
                        e["status"], e.get("reason"), now()))

    def errors(self, item_id: str, version: int | None = None, status: str | None = None) -> list[dict[str, Any]]:
        where, params = ["item_id = ?"], [item_id]
        if version is not None:
            where.append("version = ?")
            params.append(version)
        if status is not None:
            where.append("status = ?")
            params.append(status)
        with self._tx() as cur:
            self._exec(cur, f"SELECT * FROM relock_errors WHERE {' AND '.join(where)} ORDER BY created_at, id", params)
            rows = self._rows(cur)
        for r in rows:
            r["diff"] = loads(r["diff"])
            r["authoritative"] = bool(r["authoritative"])
            r["created_at"] = ts(r["created_at"])
        return rows

    def set_error_status(self, ids: Sequence[str], status: str, reason: str | None = None) -> None:
        if not ids:
            return
        marks = ", ".join("?" * len(ids))
        with self._tx() as cur:
            self._exec(cur, f"UPDATE relock_errors SET status = ?, reason = ? WHERE id IN ({marks})",
                       (status, reason, *ids))

    # ------------------------------------------------------------------ events
    def add_event(self, type_: str, *, episode_id: str | None = None, item_id: str | None = None,
                  version: int | None = None, data: dict[str, Any] | None = None) -> dict[str, Any]:
        ev = {"id": new_id("evt"), "at": now(), "type": type_, "episode_id": episode_id,
              "item_id": item_id, "version": version, "data": data or {}}
        with self._tx() as cur:
            self._exec(cur, "INSERT INTO relock_events (id, at, type, episode_id, item_id, version, data) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (ev["id"], ev["at"], type_, episode_id, item_id, version, dumps(ev["data"])))
        return ev

    def events(self, *, item_id: str | None = None, episode_id: str | None = None, type_: str | None = None,
               limit: int = 100) -> list[dict[str, Any]]:
        where, params = [], []
        if item_id:
            where.append("item_id = ?")
            params.append(item_id)
        if episode_id:
            where.append("episode_id = ?")
            params.append(episode_id)
        if type_:
            where.append("type = ?")
            params.append(type_)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._tx() as cur:
            self._exec(cur, f"SELECT * FROM relock_events {clause} ORDER BY at DESC, id DESC LIMIT ?",
                       (*params, int(limit)))
            rows = self._rows(cur)
        for r in rows:
            r["data"] = loads(r["data"])
            r["at"] = ts(r["at"])
        return rows

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        queries = {
            "items": "SELECT COUNT(*) FROM relock_items WHERE archived = ?",
            "versions": "SELECT COUNT(*) FROM relock_versions",
            "rewrites": "SELECT COUNT(*) FROM relock_versions WHERE change_kind = 'reconsolidate'",
            "rollbacks": "SELECT COUNT(*) FROM relock_versions WHERE change_kind = 'rollback'",
            "episodes": "SELECT COUNT(*) FROM relock_episodes",
            "expectations": "SELECT COUNT(*) FROM relock_expectations",
            "errors_pending": "SELECT COUNT(*) FROM relock_errors WHERE status = 'pending'",
            "errors_blocked": "SELECT COUNT(*) FROM relock_errors WHERE status = 'blocked'",
            "errors_consumed": "SELECT COUNT(*) FROM relock_errors WHERE status = 'consumed'",
        }
        with self._tx() as cur:
            for name, q in queries.items():
                self._exec(cur, q, (False,) if "archived" in q else ())
                out[name] = int(cur.fetchone()[0])
        return out

    # ------------------------------------------------------------------ export
    def iter_rewrites(self) -> list[dict[str, Any]]:
        """Every reconsolidation with its parent version and evidence (for distillation datasets)."""
        with self._tx() as cur:
            self._exec(cur, self._select_versions("v.change_kind = 'reconsolidate'") + " ORDER BY v.created_at")
            rows = [self._item_from(r) for r in self._rows(cur)]
        return [{"item": r} for r in rows]
