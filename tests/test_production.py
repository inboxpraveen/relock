"""Production behaviour: multi-agent attribution, storage robustness, scale, retention, observability."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import relock_memory as rl
from relock_memory.integrations.toolkit import Toolkit


# ---------------------------------------------------------------- multi-agent attribution
def test_two_agents_with_their_own_toolkits_keep_separate_episodes(mem, orders):
    a, b = Toolkit(mem, task="agent A"), Toolkit(mem, task="agent B")
    a.recall_memory("fetch orders by date")
    b.recall_memory("send an email")
    assert a.episode() is not b.episode()
    assert {i.key for i in a.episode().labile.values()} == {"orders.fetch"}
    assert {i.key for i in b.episode().labile.values()} == {"email.send"}


def test_a_tool_failing_in_another_thread_is_not_counted_against_an_open_episode(mem, orders, caplog):
    item, get_orders = orders
    ready, done = threading.Event(), threading.Event()

    def agent_a() -> None:
        with mem.episode("agent A") as ep:
            ep.recall("fetch orders by date")
            ready.set()
            done.wait(5)

    t = threading.Thread(target=agent_a)
    t.start()
    ready.wait(5)
    with caplog.at_level(logging.WARNING, logger="relock_memory"), pytest.raises(rl.ToolError):
        get_orders("03/14/2026")  # another agent's call, in a thread with no episode of its own
    done.set()
    t.join()
    assert mem.pending(item.key)["pending_errors"] == []
    assert "outside any episode" in caplog.text and "get_orders" in caplog.text


def test_episode_run_binds_worker_threads(mem, orders):
    item, get_orders = orders
    with mem.episode("t") as ep, ThreadPoolExecutor(2) as pool:
        ep.recall("fetch orders by date")
        with pytest.raises(rl.ToolError):
            pool.submit(ep.run, get_orders, "03/14/2026").result()

        def via_activate() -> None:
            with ep.activate():
                get_orders("03/14/2026")

        with pytest.raises(rl.ToolError):
            pool.submit(via_activate).result()
    assert mem.get(item.key).version == 2  # two trusted errors from worker threads: rewritten


def test_toolkit_episode_reaches_wrapped_tools_in_the_same_context(mem, orders):
    item, get_orders = orders
    kit = Toolkit(mem)
    kit.recall_memory("fetch orders by date")
    for _ in range(2):
        with pytest.raises(rl.ToolError):
            get_orders("03/14/2026")
    assert mem.get(item.key).version == 2
    kit.end_episode()
    assert mem.current_episode() is None


# ---------------------------------------------------------------- storage robustness
from tests.conftest import PG_URL  # noqa: E402

needs_pg = pytest.mark.skipif(not PG_URL, reason="set RELOCK_TEST_POSTGRES_URL to run Postgres tests")


def _fresh_pg() -> str:
    from relock_memory.stores.postgres import PostgresStore

    s = PostgresStore(PG_URL)
    s.drop_all()
    s.close()
    return PG_URL


@pytest.fixture(params=["sqlite-file", pytest.param("postgres", marks=needs_pg)])
def shared_url(request, tmp_path):
    """A database that several Memory instances (think: replicas) open at once."""
    return str(tmp_path / "shared.db") if request.param == "sqlite-file" else _fresh_pg()


def _in_threads(n, fn):
    errors: list[BaseException] = []

    def run(i):
        try:
            fn(i)
        except BaseException as e:  # collected for the assertion
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_replicas_starting_together_on_a_fresh_database_all_succeed(shared_url):
    def start(_):
        m = rl.Memory(shared_url)
        try:
            for i in range(60):
                m.add("Seeded procedure", ["Do the thing."], key=f"seed.{i}", exist_ok=True)
        finally:
            m.close()

    assert _in_threads(6, start) == []
    m = rl.Memory(shared_url)
    assert len(m.items()) == 60
    m.close()


def test_a_newer_schema_is_refused_with_a_clear_error(tmp_path):
    path = str(tmp_path / "x.db")
    rl.Memory(path).close()
    import sqlite3

    con = sqlite3.connect(path)
    con.execute("UPDATE relock_meta SET v = '99' WHERE k = 'schema_version'")
    con.commit()
    con.close()
    with pytest.raises(rl.StoreError, match="schema v99"):
        rl.Memory(path)


def test_a_file_that_is_not_a_database_raises_a_store_error_naming_it(tmp_path):
    path = tmp_path / "notes.db"
    path.write_bytes(b"this is not a sqlite database" * 100)
    with pytest.raises(rl.StoreError, match=r"notes.db"):
        rl.Memory(str(path))


@needs_pg
def test_postgres_recovers_after_the_server_drops_connections():
    import psycopg

    m = rl.Memory(_fresh_pg())
    m.add("Fetch orders", ["Call get_orders."], key="orders.fetch")
    with psycopg.connect(PG_URL, autocommit=True) as admin:
        admin.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                      "WHERE datname = current_database() AND pid <> pg_backend_pid()")
    outcomes = []
    for _ in range(3):
        try:
            outcomes.append(m.get("orders.fetch").version)
        except rl.StoreError as e:
            outcomes.append(str(e))
    assert outcomes[-2:] == [1, 1], outcomes  # at most the first call fails; then it reconnects
    m.close()


@needs_pg
def test_postgres_serves_threads_from_a_connection_pool():
    from relock_memory.stores.postgres import PostgresStore

    _fresh_pg()
    store = PostgresStore(PG_URL, pool_size=4)
    m = rl.Memory(store)
    for i in range(20):
        m.add(f"proc {i}", [f"step {i}"], key=f"p{i}")

    def work(_):
        with m.episode("w") as ep:
            for _ in range(20):
                ep.recall("proc step")

    assert _in_threads(8, work) == []
    assert 1 < store.pool_stats()["open"] <= 4
    m.close()


# ---------------------------------------------------------------- scale
def test_confirmations_do_not_reload_every_item(mem, orders, monkeypatch):
    _item, get_orders = orders
    loads = []
    real = mem.store.active
    monkeypatch.setattr(mem.store, "active", lambda *a, **k: loads.append(1) or real(*a, **k))
    with mem.episode("t") as ep:
        ep.recall("fetch orders by date")
        for _ in range(5):
            get_orders("2026-03-14")
            hits = ep.recall("fetch orders by date")
    assert len(loads) == 1
    assert hits[0].item.confirmations == 5  # recall still reports current confidence


def test_incremental_index_ranks_exactly_like_a_fresh_one(mem):
    import random

    rnd = random.Random(7)
    words = "orders invoice refund date currency ledger report shipping account export retry page".split()
    for i in range(60):
        mem.add(f"{rnd.choice(words)} procedure {i}", [" ".join(rnd.choices(words, k=6)) + f" step{i}"],
                key=f"k{i}", kind=rnd.choice(["procedure", "fact"]))
    queries = [" ".join(rnd.choices(words, k=3)) for _ in range(25)]
    for round_ in range(4):
        for i in rnd.sample(range(60), 5):  # new versions of some items, between queries
            mem.edit(f"k{i}", [{"op": "add", "text": " ".join(rnd.choices(words, k=4)) + f" r{round_}"}])
        mem.archive(f"k{round_}")
        items = list(mem._active_items())
        for q in queries:
            got = [(h.item.id, h.item.version, h.score) for h in mem.search(q, k=8)]  # the long-lived index
            want = [(i.id, i.version, s) for i, s in rl.BM25Retriever().rank(q, items, 8)]
            assert got == want


def test_filtered_recall_only_returns_matching_items(mem, orders):
    mem.add("Orders fact", ["The orders API paginates from 0."], key="orders.fact", kind="fact", tags=["orders"])
    with mem.episode("t") as ep:
        assert {h.item.key for h in ep.recall("orders api", k=5, kinds=["fact"])} == {"orders.fact"}
        assert {h.item.key for h in ep.recall("orders api", k=5, tags=["orders"])} == {"orders.fetch", "orders.fact"}
        assert {i.key for i in ep.labile.values()} == {"orders.fetch", "orders.fact"}


def test_a_long_episode_keeps_only_open_expectations_in_memory(mem, orders):
    _, get_orders = orders
    with mem.episode("long") as ep:
        ep.recall("fetch orders by date")
        for _ in range(300):
            get_orders("2026-03-14")
        assert len(ep._expectations) == 0 and not ep._explicit  # nothing grows per call
        assert not any(isinstance(v, (set, list, dict)) and len(v) > 5 for v in vars(ep).values())
    assert ep.summary["expectations"] == 300


def _snapshot(items):
    return {i.id: (i.version, i.archived) for i in items}


def test_content_changes_refresh_only_what_changed_across_processes(tmp_path, monkeypatch):
    path = str(tmp_path / "shared.db")
    a, b = rl.Memory(path), rl.Memory(path)  # two processes sharing one database
    for i in range(30):
        a.add(f"procedure {i}", [f"step {i}"], key=f"p{i}")
    monkeypatch.setattr("relock_memory.memory._CHANGE_WINDOW_S", 0.0)  # no re-read window for old writers
    a.search("procedure")  # warm a's cache
    full_loads, reread = [], []
    real, real_changed = a.store.active, a.store.changed_since
    monkeypatch.setattr(a.store, "active", lambda *x, **k: full_loads.append(1) or real(*x, **k))
    monkeypatch.setattr(a.store, "changed_since", lambda *x: reread.extend(real_changed(*x)) or real_changed(*x))
    b.add("new procedure", ["a new step"], key="new")
    b.edit("p3", [{"op": "add", "text": "and log it"}])
    b.archive("p4")
    b.rollback("p3", 1)
    a.edit("p5", [{"op": "add", "text": "edited by a"}])
    assert _snapshot(a._active_items()) == _snapshot(real())
    assert {h.item.key for h in a.search("new step", k=1)} == {"new"}
    assert full_loads == []
    assert {i.key for i in reread} <= {"new", "p3", "p4", "p5"}
    a.close()
    b.close()


def test_changes_from_older_relock_writers_are_still_picked_up(tmp_path):
    import sqlite3

    path = str(tmp_path / "mixed.db")
    a = rl.Memory(path)
    for i in range(5):
        a.add(f"procedure {i}", [f"step {i}"], key=f"p{i}")
    assert len(a._active_items()) == 5
    con = sqlite3.connect(path)  # a relock 0.2 replica archives p2: no revision stamp on the item
    con.execute("UPDATE relock_items SET archived = 1, updated_at = ? WHERE key = 'p2'", (rl._util.now(),))
    con.execute("UPDATE relock_meta SET v = CAST(CAST(v AS INTEGER) + 1 AS TEXT) WHERE k = 'revision'")
    con.commit()
    con.close()
    assert "p2" not in {i.key for i in a._active_items()}
    a.close()


def test_a_version_1_database_is_migrated_in_place(tmp_path):
    import sqlite3

    path = str(tmp_path / "v1.db")
    m = rl.Memory(path)
    m.add("Fetch orders", ["Call get_orders."], key="orders.fetch")
    m.close()
    con = sqlite3.connect(path)  # turn it back into a 0.2.x database
    con.execute("DROP INDEX relock_items_rev")
    con.execute("ALTER TABLE relock_items DROP COLUMN rev")
    con.execute("UPDATE relock_meta SET v = '1' WHERE k = 'schema_version'")
    con.commit()
    con.close()
    m = rl.Memory(path)
    m.edit("orders.fetch", [{"op": "add", "text": "Use ISO dates."}])
    assert m.get("orders.fetch").version == 2
    assert [h.item.version for h in m.search("orders iso dates")] == [2]
    with m.store._tx(write=False) as cur:
        cur.execute("SELECT v FROM relock_meta WHERE k = 'schema_version'")
        assert cur.fetchone()[0] == "2"
    m.close()


# ---------------------------------------------------------------- observability
def test_ledger_events_are_logged_with_ids_and_reasons(mem, orders, caplog):
    _, get_orders = orders
    caplog.set_level(logging.INFO, logger="relock_memory")
    with mem.episode("fetch") as ep:
        ep.recall("fetch orders by date")
        for _ in range(2):  # two trusted errors: enough evidence to rewrite
            with pytest.raises(rl.ToolError):
                get_orders("03/14/2026")
    lines = [r.getMessage() for r in caplog.records]
    rewrite = [m for m in lines if m.startswith("relock rewrite")]
    assert len(rewrite) == 1 and ep.id in rewrite[0] and "v1->v2" in rewrite[0]
    assert any(m.startswith("relock prediction_error") and "invalid date" in m for m in lines)
    assert not any(m.startswith("relock recall") for m in lines)  # routine events stay at DEBUG
    assert all("data" not in r.relock_event for r in caplog.records if hasattr(r, "relock_event"))


def test_debug_logging_attaches_the_full_event(mem, orders, caplog):
    caplog.set_level(logging.DEBUG, logger="relock_memory")
    with mem.episode("t", ticket="T-1") as ep:
        ep.recall("fetch orders by date")
    events = {r.relock_event["type"]: r.relock_event for r in caplog.records if hasattr(r, "relock_event")}
    assert events["recall"]["data"]["query"] == "fetch orders by date"
    assert events["episode_open"]["data"]["meta"] == {"ticket": "T-1"}


def test_a_failing_event_hook_is_reported_and_the_event_kept(store, caplog):
    def hook(ev):
        raise RuntimeError("hook bug")

    m = rl.Memory(store, on_event=hook)
    m.add("A", ["x"], key="a")
    m.add("B", ["y"], key="b")
    warnings = [r for r in caplog.records if "on_event hook raised" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].exc_info is not None
    assert len(m.events(type="create")) == 2


def test_a_rewrite_on_stale_evidence_is_recorded_as_a_conflict(mem, orders):
    _, get_orders = orders
    a, b = mem.episode("a"), mem.episode("b")
    a.recall("fetch orders by date")
    b.recall("fetch orders by date")
    for _ in range(2):
        with a.activate(), pytest.raises(rl.ToolError):
            get_orders("03/14/2026")  # a rewrites the item to v2
    with b.activate(), pytest.raises(rl.ToolError):
        get_orders("03/14/2026")  # b's evidence is about v1
    a.close()
    b.close()
    conflicts = mem.events(type="conflict")
    assert [e["episode_id"] for e in conflicts] == [b.id]
    assert "evidence is stale" in conflicts[0]["data"]["reason"]


def test_a_spent_rewrite_budget_is_recorded(mem, orders):
    _, get_orders = orders
    with mem.episode("t") as ep:
        ep.recall("fetch orders by date")
        for _ in range(4):  # 2 errors rewrite to v2; 2 more on v2 would rewrite again, past the budget
            with pytest.raises(rl.ToolError):
                get_orders("03/14/2026")
    deferred = mem.events(type="rewrite_deferred")
    assert deferred and "budget" in deferred[0]["data"]["reason"]


def test_a_failing_llm_rewriter_is_logged_and_named_in_the_decision(store, caplog):
    def down(prompt):
        raise ConnectionError("endpoint down")

    m = rl.Memory(store, rewriter=rl.LLMRewriter(down, fallback=rl.CallableRewriter(lambda r: None)))
    item = m.add("Fetch orders", ["Call get_orders with MM/DD/YYYY dates."], key="orders.fetch")
    with m.episode("t") as ep:
        ep.recall("fetch orders")
        for _ in range(2):
            obs = ep.observe(ep.expect(item, "orders come back"),
                             error=rl.ToolError("invalid date", correction="Use ISO dates."))
    assert obs.decision.verdict is rl.Verdict.DECLINED
    assert "endpoint down" in obs.decision.reason
    assert "endpoint down" in m.events(type="declined")[0]["data"]["reason"]
    assert any("LLM call failed" in r.getMessage() for r in caplog.records)


def test_huge_error_text_is_capped_in_the_ledger(mem, orders):
    import json

    item, _ = orders
    with mem.episode("t") as ep:
        ep.recall("fetch orders by date")
        ep.observe(ep.expect(item, "orders come back"), error=RuntimeError("x" * 50_000))
    stored = mem.store.errors(item.id)
    assert stored and all(len(json.dumps(e["diff"])) < 10_000 for e in stored)


# ---------------------------------------------------------------- retention
def _confirm(ep, item, n=1):
    for _ in range(n):
        ep.observe(ep.expect(item, "orders come back"), [{"id": 1}])


def _fail(ep, item, correction="Call get_orders with ISO-8601 dates."):
    return ep.observe(ep.expect(item, "orders come back", clauses=["c2"]),
                      error=rl.ToolError("invalid date", correction=correction))


def _exists(mem, episode_id):
    try:
        return mem.store.get_episode(episode_id)
    except rl.NotFoundError:
        return None


def test_prune_drops_routine_audit_and_keeps_everything_a_version_depends_on(mem, orders):
    from datetime import timedelta

    item, _ = orders
    email = mem.get("email.send")
    later_fix = "Call get_orders with ISO-8601 dates in UTC."
    with mem.episode("routine") as routine:  # nothing depends on it
        routine.recall("send email recipients")
        _confirm(routine, email, 3)
    with mem.episode("fix") as fix:  # the rewrite happened here
        fix.recall("fetch orders by date")
        fix.recall("send email recipients")
        _fail(fix, item)
        assert _fail(fix, item).verdict is rl.Verdict.REWRITTEN
        _confirm(fix, email, 2)
    with mem.episode("pending") as pending:  # live evidence against v2
        pending.recall("fetch orders by date")
        assert _fail(pending, item, later_fix).verdict is rl.Verdict.PENDING
    running = mem.episode("still running")
    running.recall("fetch orders by date")

    planned = mem.prune(timedelta(0), dry_run=True)
    assert _exists(mem, routine.id)  # a dry run changes nothing
    done = mem.prune(timedelta(0))
    assert {k: done[k] for k in planned} == planned
    assert done["episodes"] == 1 and done["episodes_abandoned"] == 0
    assert done["expectations_trimmed"] == 2  # fix's two confirmations; its failures are evidence

    assert not _exists(mem, routine.id)
    for ep in (fix, pending, running):
        assert _exists(mem, ep.id)
    v2 = mem.get(item.id)
    assert [v.version for v in mem.history(item.id)] == [1, 2]
    assert len(mem.store.errors_by_ids(list(v2.evidence))) == len(v2.evidence) == 2
    assert {e["type"] for e in mem.events(limit=500)} >= {"create", "rewrite", "prune"}
    running.close()

    with mem.episode("later") as later:  # pending evidence survived: one more error rewrites v2
        later.recall("fetch orders by date")
        assert _fail(later, item, later_fix).verdict is rl.Verdict.REWRITTEN


def test_prune_marks_episodes_of_dead_processes_abandoned(tmp_path):
    from datetime import timedelta

    path = str(tmp_path / "shared.db")
    crashed = rl.Memory(path)
    crashed.add("Fetch orders", ["Call get_orders."], key="orders.fetch")
    orphan = crashed.episode("never closed")
    orphan.recall("orders")
    crashed.store.close()  # the process died: the episode stays open in the database

    m = rl.Memory(path)
    mine = m.episode("open here")
    mine.recall("orders")
    counts = m.prune(timedelta(0))
    assert counts["episodes_abandoned"] == 1 and counts["episodes"] == 1
    assert not _exists(m, orphan.id)
    assert _exists(m, mine.id)["status"] == "open"
    mine.close()
    m.close()


def test_prune_cli_reports_a_dry_run(tmp_path, capsys):
    from relock_memory.cli import main

    db = str(tmp_path / "cli.db")
    m = rl.Memory(db)
    item = m.add("Fetch orders", ["Call get_orders."], key="orders.fetch")
    with m.episode("t") as ep:
        ep.recall("orders")
        _confirm(ep, item)
    m.close()
    assert main(["--db", db, "prune", "--older-than", "0s", "--dry-run"]) == 0
    assert "would delete 1 episode(s)" in capsys.readouterr().out
    assert main(["--db", db, "prune", "--older-than", "0s", "--json"]) == 0
    assert '"episodes": 1' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--db", db, "prune", "--older-than", "30"])  # no unit


# ---------------------------------------------------------------- integrations
def test_an_episode_policy_overrides_the_memory_policy_for_that_episode_only(mem, orders):
    item, _ = orders
    trusting = mem.policy.with_(trusted_sources=mem.policy.trusted_sources | {rl.Source.AGENT})
    report = rl.Signal.agent_report(False, "dates rejected", correction="Use ISO-8601 dates.")
    with mem.episode("trusted agent", policy=trusting) as ep:
        ep.recall("fetch orders by date")
        assert ep.observe(ep.expect(item, "rows"), signal=report).decision.verdict is not rl.Verdict.BLOCKED
    with mem.episode("other agent") as ep:
        ep.recall("fetch orders by date")
        assert ep.observe(ep.expect(item, "rows"), signal=report).decision.verdict is rl.Verdict.BLOCKED
    assert rl.Source.AGENT not in mem.policy.trusted_sources


def test_an_mcp_server_that_trusts_agents_leaves_the_shared_memory_policy_alone(mem):
    pytest.importorskip("mcp")
    from relock_memory.integrations.mcp import build_server

    before = mem.policy
    build_server(mem, trust_agent_reports=True)
    assert mem.policy is before


def test_a_missing_row_is_a_store_error_not_a_conflict(mem, orders):
    ep = mem.episode("deleted under us")  # e.g. pruned by a replica that thought it was abandoned
    with mem.store._tx() as cur:
        mem.store._exec(cur, "DELETE FROM relock_episodes WHERE id = ?", (ep.id,))
    with pytest.raises(rl.StoreError) as raised:
        ep.recall("fetch orders by date")
    assert not isinstance(raised.value, rl.ConflictError)


def test_observing_an_expectation_twice_is_refused(mem, orders):
    item, _ = orders
    with mem.episode("t") as ep:
        ep.recall("fetch orders by date")
        exp = ep.expect(item, "rows")
        ep.observe(exp, [{"id": 1}])
        with pytest.raises(rl.RelockError, match="already observed"):
            ep.observe(exp, [{"id": 1}])
        with pytest.raises(rl.NotFoundError):
            ep.observe("exp_does_not_exist", [])
