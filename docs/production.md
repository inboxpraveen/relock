# Running relock in production

This page covers what changes when relock runs for real: several agents, long-lived databases,
replicas and monitoring. [Configuration](configuration.md) covers the settings themselves.

## Checklist

- Wrap each task in `with mem.episode(...)`, so episodes always re-lock.
- Give each agent its own episode or its own `Toolkit`; in worker threads use `ep.run(fn, ...)`.
  See [several agents, threads and asyncio](python.md#several-agents-threads-and-asyncio).
- Mark failures whose text comes from outside your system `detail_trusted=False`.
- Seed items with `exist_ok=True`, so a restart never overwrites what was learned.
- Route `relock_memory` logs to your log pipeline, and alert on warnings.
- Schedule `relock prune --older-than 30d`.
- For the HTTP API: set `RELOCK_TOKEN`, terminate TLS in front of it, and pin each episode to one
  replica. See [HTTP deployment](http.md#deployment).

## Logs and audit

Every ledger event is logged on the `relock_memory` logger as it is stored.

| level | events |
|---|---|
| INFO | content changes (create, rewrite, manual edit, rollback, archive, prune), prediction errors, blocked or declined writes, conflicts, deferred rewrites |
| WARNING | an invalid edit proposed by the rewriter, a failing `on_event` hook, an unreachable LLM rewriter, a lost database connection |
| DEBUG | routine recalls, predictions and confirmations |

Each line names the event, episode, item and version, plus a short reason for failures: the first
300 characters of what went wrong, which can include the tool's error message. Full payloads, such
as clause text and complete tool errors, are attached to the record as `record.relock_event` only
at DEBUG, since they can carry user data.

The full trail stays queryable with `relock events` or SQL. Every version keeps its delta, reason,
author and the ids of the errors that justified it.

## Errors

- A database failure raises `relock.StoreError` naming the store, with the driver's exception as
  its cause. A lost connection is discarded and the next call reconnects.
- A write that lost a race to another process raises `relock.ConflictError`. Inside an episode it
  becomes a `conflict` decision instead, since the evidence was about a version that no longer
  exists.
- A write through a closed episode raises `relock.EpisodeClosedError`. A prediction about an item
  that wasn't recalled raises `relock.NotLabileError`. An unknown key or id raises
  `relock.NotFoundError`. All of them derive from `relock.RelockError`.
- A problem while recording a wrapped tool's outcome is logged, not raised, so your agent sees the
  tool's own result or exception.

## Growth and retention

Items and versions grow only when memory changes. The audit tables grow with use: each wrapped
tool call adds an expectation and about two events. Prune them on a schedule:

```
relock prune --older-than 30d --dry-run   # counts only
relock prune --older-than 30d
```

Pruning deletes an episode's retrievals, expectations, errors and events only when no version was
written in it and none of its errors is, or may become, evidence for a rewrite. Episodes that must
stay lose just their per-call confirmations. Items, versions and content-change events are never
pruned.

Episodes a crashed process left open are marked `abandoned`. Other processes can't be checked, so
the age must exceed the longest an episode stays open in any replica. SQLite keeps its file size
until `VACUUM`; Postgres reuses the space after autovacuum.

## Postgres

`PostgresStore(dsn, pool_size=5)` serves concurrent calls from a connection pool. It checks idle
connections before use, replaces ones the server dropped, and fails fast on an unreachable host
(`connect_timeout` 10 s unless the DSN sets one). Replicas starting together on an empty database
are safe.

The ledger is plain SQL with `JSONB` columns, so you can query it directly:

```sql
-- which items were rewritten from verifier failures this week, and what changed?
SELECT v.item_id, v.version, v.delta, v.reason
FROM relock_versions v
WHERE v.change_kind = 'reconsolidate' AND v.created_at > now() - interval '7 days'
  AND EXISTS (SELECT 1 FROM relock_errors e
              WHERE e.id IN (SELECT jsonb_array_elements_text(v.evidence)) AND e.source = 'verifier');
```

The HTTP server keeps open episodes in memory, so with several replicas route each episode's
requests to the same one.

## Upgrading

The first process of a new version upgrades the database schema in place, and older processes
keep working against it. On a large Postgres ledger, run `relock init` once at a quiet time
before the rollout, since an upgrade can build indexes over the events table. A database written
by a newer relock is refused with a clear message rather than misread. Breaking changes are listed
in the [changelog](../CHANGELOG.md).

## Scale

Measured on one laptop with SQLite:

- **20,000 items:** recall takes 16 ms, a wrapped tool call 2.3 ms, and recall right after a
  confirmation 2.7 ms.
- **100,000 items:** recall takes about 130 ms, and recall right after another process adds or
  rewrites an item about 0.2 s.

Only changed items are re-read, and long episodes stay flat per call. Loading memory costs about
3 KB of heap per short item, and the first recall in a new process reads every item: about 1 s at
20,000 items and 7 s at 100,000. Past that, plug in a [retriever](configuration.md#retrieval)
backed by a vector index.
