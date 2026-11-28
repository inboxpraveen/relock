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
