"""CLI and store-opening hardening: bad URLs, env vars, flag placement, clean errors, prune safety."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import pytest

import relock_memory as rl
import relock_memory.integrations.http as http_mod
import relock_memory.integrations.mcp as mcp_mod
import relock_memory.stores.postgres as pg_mod
from relock_memory import cli
from relock_memory.cli import main
from relock_memory.export import rewrite_records
from relock_memory.stores import SQLiteStore, open_store
from relock_memory.types import NotFoundError, StoreError


@pytest.fixture
def pg_calls(monkeypatch):
    """Record PostgresStore construction instead of connecting (psycopg may be installed)."""
    calls: list[tuple[str, dict]] = []

    class Recorder:
        def __init__(self, dsn, **kwargs):
            calls.append((dsn, kwargs))

    monkeypatch.setattr(pg_mod, "PostgresStore", Recorder)
    return calls


@pytest.fixture
def no_env(monkeypatch):
    for name in ("RELOCK_DB", "RELOCK_PORT", "RELOCK_LOG_LEVEL", "RELOCK_PG_POOL_SIZE"):
        monkeypatch.delenv(name, raising=False)


def _db_with_item(tmp_path, clauses=("First.", "Second.")):
    db = str(tmp_path / "cli.db")
    with rl.Memory(db) as m:
        m.add("Thing", list(clauses), key="t")
    return db


def _clauses(db, ref="t"):
    with rl.Memory(db) as m:
        return [c.text for c in m.get(ref).clauses]


def _files(directory):
    """Names in ``directory``, ignoring SQLite's WAL side files."""
    return sorted(n for n in os.listdir(directory) if not n.endswith(("-wal", "-shm", "-journal")))


# ---------------------------------------------------------------- 1. open_store schemes
@pytest.mark.parametrize("url", [
    "postgres://u:pw@h/db",
    "postgresql://u:pw@h/db",
    "postgresql+psycopg://u:pw@h/db",
    "postgresql+psycopg2://u:pw@h/db",
    "POSTGRESQL://u:pw@h/db",
    "  postgresql://u:pw@h/db\n",
])
def test_postgres_urls_open_postgres_with_a_psycopg_dsn(pg_calls, no_env, tmp_path, monkeypatch, url):
    monkeypatch.chdir(tmp_path)
    open_store(url)
    assert pg_calls == [("postgresql://u:pw@h/db", {})]
    assert os.listdir(tmp_path) == []  # never a local SQLite file named after the URL


def test_unknown_schemes_are_refused_without_echoing_the_url(no_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for url in ("mysql://u:secret@h/db", "mongodb+srv://u:secret@h/db", "http://u:secret@h/db"):
        with pytest.raises(ValueError, match="postgresql://") as e:
            open_store(url)
        assert "secret" not in str(e.value)
    assert os.listdir(tmp_path) == []


def test_sqlite_urls_and_paths(no_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = {
        "sqlite:///rel.db": "rel.db",
        "SQLITE:///upper.db": "upper.db",
        " sqlite:///spaced.db ": "spaced.db",
        "sqlite://": ":memory:",
        "sqlite:///:memory:": ":memory:",
        # sqlite:////abs/x.db on POSIX, sqlite:///C:/abs/x.db on Windows
        "sqlite:///" + (tmp_path / "abs.db").as_posix(): (tmp_path / "abs.db").as_posix(),
    }
    for target, path in cases.items():
        s = open_store(target)
        try:
            assert isinstance(s, SQLiteStore) and s.path == path, target
        finally:
            s.close()
    assert _files(tmp_path) == ["abs.db", "rel.db", "spaced.db", "upper.db"]


def test_relock_db_is_stripped_and_classified_like_an_argument(no_env, pg_calls, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RELOCK_DB", " postgresql+psycopg://u@h/db ")
    open_store()
    assert pg_calls == [("postgresql://u@h/db", {})]
    monkeypatch.setenv("RELOCK_DB", "  ")
    s = open_store()
    assert s.path == "relock.db"
    s.close()


def test_open_store_and_memory_accept_paths(no_env, tmp_path):
    s = open_store(tmp_path / "p.db")
    assert isinstance(s, SQLiteStore) and s.path == str(tmp_path / "p.db")
    s.close()
    with rl.Memory(tmp_path / "agent.db") as m:
        m.add("Thing", ["Do it."], key="t")
    assert (tmp_path / "agent.db").exists()


def test_file_uris_open_as_sqlite_uris(no_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = open_store("file:x.db?mode=rwc")
    s.close()
    assert _files(tmp_path) == ["x.db"]  # not a file named 'file' or 'file:x.db?mode=rwc'
    s = open_store("file:" + (tmp_path / "y.db").as_posix())
    s.close()
    assert (tmp_path / "y.db").exists()
    s = open_store("file:mem1?mode=memory&cache=shared")
    s.close()
    assert _files(tmp_path) == ["x.db", "y.db"]
    # relock writes on every call, so a read-only URI fails cleanly rather than half-working
    try:
        s = open_store("file:x.db?mode=ro")
    except StoreError:
        pass
    else:
        s.close()


def test_pool_size_comes_from_relock_pg_pool_size(no_env, pg_calls, monkeypatch):
    monkeypatch.setenv("RELOCK_PG_POOL_SIZE", " 32 ")
    open_store("postgresql://u@h/db")
    open_store("postgresql://u@h/db", pool_size=3)  # an explicit argument wins
    assert [kw for _, kw in pg_calls] == [{"pool_size": 32}, {"pool_size": 3}]
    for bad in ("abc", "0", "-4", "2.5"):
        monkeypatch.setenv("RELOCK_PG_POOL_SIZE", bad)
        with pytest.raises(ValueError, match="RELOCK_PG_POOL_SIZE"):
            open_store("postgresql://u@h/db")
    open_store(":memory:").close()  # irrelevant to SQLite


# ---------------------------------------------------------------- 2. bad SQLite paths
def test_a_file_in_the_way_of_the_db_directory_is_a_store_error(no_env, tmp_path, capsys):
    (tmp_path / "notadir").write_text("x")
    bad = str(tmp_path / "notadir" / "x.db")
    with pytest.raises(StoreError, match="notadir"):
        SQLiteStore(bad)
    assert main(["--db", bad, "ls"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err


# ---------------------------------------------------------------- 3. environment variables
def test_a_bad_relock_port_only_matters_to_serve(no_env, tmp_path, monkeypatch, capsys):
    db = _db_with_item(tmp_path)
    monkeypatch.setenv("RELOCK_PORT", "abc")
    assert main(["--db", db, "ls"]) == 0
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and "relock" in capsys.readouterr().out


def test_serve_falls_back_from_a_kubernetes_relock_port(no_env, tmp_path, monkeypatch, caplog):
    seen = []
    monkeypatch.setattr(http_mod, "serve", lambda db, **kw: seen.append(kw["port"]))
    db = str(tmp_path / "s.db")
    for env, port in (("tcp://10.0.0.1:8765", 8765), ("abc", 8765), ("99999", 8765), ("9000", 9000),
                      (" 9001 ", 9001), ("", 8765)):
        monkeypatch.setenv("RELOCK_PORT", env)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert main(["--db", db, "serve"]) == 0
        assert seen[-1] == port, env
        assert ("RELOCK_PORT" in caplog.text) == (env.strip() not in ("", "9000", "9001")), env
    assert main(["--db", db, "serve", "--port", "9100"]) == 0 and seen[-1] == 9100  # the flag wins
    with pytest.raises(SystemExit) as e:
        main(["--db", db, "serve", "--port", "70000"])
    assert e.value.code == 2


def test_a_bad_relock_log_level_is_a_clean_usage_error(no_env, tmp_path, monkeypatch, capsys):
    configured = []
    monkeypatch.setattr(cli.logging, "basicConfig", lambda **kw: configured.append(kw["level"]))
    db = _db_with_item(tmp_path)
    monkeypatch.setenv("RELOCK_LOG_LEVEL", "verbose")
    with pytest.raises(SystemExit) as e:
        main(["--db", db, "ls"])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "RELOCK_LOG_LEVEL" in err and "DEBUG" in err and "WARNING" in err
    assert main(["--db", db, "--log-level", "error", "ls"]) == 0  # the flag beats the env var
    monkeypatch.setenv("RELOCK_LOG_LEVEL", "info")
    assert main(["--db", db, "ls"]) == 0
    assert configured == ["ERROR", "INFO"]


# ---------------------------------------------------------------- 4. --db after the subcommand
def test_db_and_log_level_work_after_the_subcommand(no_env, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    configured = []
    monkeypatch.setattr(cli.logging, "basicConfig", lambda **kw: configured.append(kw["level"]))
    flag = str(tmp_path / "flag.db")
    monkeypatch.setenv("RELOCK_DB", str(tmp_path / "env.db"))
    assert main(["add", "Thing", "Do it.", "--key", "t", "--db", flag]) == 0
    assert main(["ls", "--db", flag, "--log-level", "debug"]) == 0
    assert "t " in capsys.readouterr().out and configured == ["DEBUG"]
    assert main(["--db", flag, "ls"]) == 0  # the global position still works
    assert not (tmp_path / "env.db").exists()
    assert main(["ls"]) == 0  # $RELOCK_DB next
    assert (tmp_path / "env.db").exists() and not (tmp_path / "relock.db").exists()
    monkeypatch.delenv("RELOCK_DB")
    assert main(["ls"]) == 0  # then ./relock.db
    assert (tmp_path / "relock.db").exists()

    seen = {}
    monkeypatch.setattr(mcp_mod, "_load_server_class", lambda: object)
    monkeypatch.setattr(mcp_mod, "serve", lambda db, **kw: seen.update(mcp=db))
    monkeypatch.setattr(http_mod, "serve", lambda db, **kw: seen.update(serve=db))
    assert main(["mcp", "--db", flag, "--transport", "stdio"]) == 0
    assert main(["serve", "--db", flag]) == 0
    assert main(["--db", "ignored.db", "serve", "--db", flag]) == 0 and seen == {"mcp": flag, "serve": flag}


# ---------------------------------------------------------------- 5. edit
def test_edit_keeps_the_order_of_several_adds(no_env, tmp_path):
    db = _db_with_item(tmp_path)
    assert main(["--db", db, "edit", "t", "--after", "c1", "--add", "A", "--add", "B", "--add", "C"]) == 0
    assert _clauses(db) == ["First.", "A", "B", "C", "Second."]
    assert main(["--db", db, "edit", "t", "--add", "D", "--add", "E"]) == 0
    assert _clauses(db)[-2:] == ["D", "E"]


def test_edit_refuses_two_ops_on_one_clause(no_env, tmp_path, capsys):
    db = _db_with_item(tmp_path)
    assert main(["--db", db, "edit", "t", "--remove", "c2", "--replace", "c2", "x"]) == 2
    assert main(["--db", db, "edit", "t", "--replace", "c1", "x", "--replace", "c1", "y"]) == 2
    assert main(["--db", db, "edit", "t", "--after", "c1"]) == 2
    assert "c2" in capsys.readouterr().err
    assert _clauses(db) == ["First.", "Second."]


# ---------------------------------------------------------------- 6. one error handler
def _bad_db(tmp_path):
    (tmp_path / "notadir").write_text("x")
    return str(tmp_path / "notadir" / "x.db")


def test_serve_and_mcp_report_a_bad_db_cleanly(no_env, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mcp_mod, "_load_server_class", lambda: object)
    bad = _bad_db(tmp_path)
    for argv in (["--db", bad, "serve"], ["--db", bad, "mcp"], ["mcp", "--db", bad]):
        assert main(argv) == 1, argv
        err = capsys.readouterr().err
        assert err.startswith("error: ") and "notadir" in err, argv


def test_mcp_without_the_mcp_extra_is_a_clean_error(no_env, tmp_path, monkeypatch, capsys):
    def missing():
        raise ImportError("the MCP server needs the 'mcp' package: pip install 'relock-memory[mcp]'")

    monkeypatch.setattr(mcp_mod, "_load_server_class", missing)
    db = tmp_path / "m.db"
    assert main(["--db", str(db), "mcp"]) == 1
    assert "pip install 'relock-memory[mcp]'" in capsys.readouterr().err
    assert not db.exists()  # checked before the database is touched


def test_mcp_verifier_output_is_trusted_unless_the_operator_says_otherwise(no_env, tmp_path, monkeypatch):
    seen: list[bool] = []
    monkeypatch.setattr(mcp_mod, "_load_server_class", lambda: object)
    monkeypatch.setattr(mcp_mod, "serve", lambda db, **kw: seen.append(kw["trust_verifier_output"]))
    db = str(tmp_path / "m.db")
    assert main(["--db", db, "mcp"]) == 0
    assert main(["--db", db, "mcp", "--untrusted-verifier-output"]) == 0
    assert seen == [True, False]


def test_a_postgres_url_without_psycopg_is_a_clean_error(no_env, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "psycopg", None)
    for argv in (["--db", "postgresql://u@h/db", "ls"], ["--db", "postgresql://u@h/db", "serve"]):
        assert main(argv) == 1, argv
        assert "psycopg" in capsys.readouterr().err


def test_export_to_a_missing_directory_is_a_clean_error(no_env, tmp_path, capsys):
    db = _db_with_item(tmp_path)
    assert main(["--db", db, "export", str(tmp_path / "nodir" / "out.jsonl")]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_an_unsupported_db_url_is_a_clean_error(no_env, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["--db", "mysql://u:secret@h/db", "ls"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "secret" not in err


# ---------------------------------------------------------------- 7. prune
def _db_with_old_episode(tmp_path):
    db = str(tmp_path / "p.db")
    with rl.Memory(db) as m:
        item = m.add("Fetch orders", ["Call get_orders."], key="orders")
        with m.episode("t") as ep:
            ep.recall("orders")
            ep.observe(ep.expect(item, "orders come back"),
                       signal=rl.Signal.agent_report(False, "dates rejected"))  # untrusted: blocked
    time.sleep(0.05)
    return db


def test_prune_refuses_short_ages_unless_forced(no_env, tmp_path, capsys):
    db = _db_with_old_episode(tmp_path)
    for age in ("0s", "59m"):
        assert main(["--db", db, "prune", "--older-than", age]) == 2
        err = capsys.readouterr().err
        assert "--force" in err and "abandoned" in err
    assert main(["--db", db, "prune", "--older-than", "0s", "--dry-run"]) == 0  # changes nothing
    assert main(["--db", db, "prune", "--older-than", "1h"]) == 0
    capsys.readouterr()
    assert main(["--db", db, "prune", "--older-than", "0s", "--force", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["episodes"] == 1


@pytest.mark.parametrize("age", ["1e12d", "100000000d", "infd", "nand", "1e400d", "36526d"])
def test_prune_rejects_absurd_ages_cleanly(no_env, tmp_path, capsys, age):
    db = _db_with_item(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(["--db", db, "prune", "--older-than", age])
    assert e.value.code == 2 and "invalid age" in capsys.readouterr().err


def test_prune_accepts_a_century(no_env, tmp_path):
    db = _db_with_item(tmp_path)
    assert main(["--db", db, "prune", "--older-than", "36500d", "--dry-run"]) == 0


def test_prune_can_keep_more_event_types(no_env, tmp_path, capsys):
    db = _db_with_old_episode(tmp_path)

    def dry(*extra):
        assert main(["--db", db, "prune", "--older-than", "0s", "--dry-run", "--json", *extra]) == 0
        return json.loads(capsys.readouterr().out)["events"]

    assert dry("--keep-events", "blocked") == dry() - 1
    assert dry("--keep-events", "blocked", "--keep-events", "recall") == dry() - 2
    assert main(["--db", db, "prune", "--older-than", "0s", "--force", "--keep-events", "blocked"]) == 0
    with rl.Memory(db) as m:
        assert [e["type"] for e in m.events(type="blocked")] == ["blocked"]
        assert m.events(type="recall") == []
        assert m.events(type="create")  # the defaults are still kept


# ---------------------------------------------------------------- 8. events --limit
def test_events_limit_must_be_positive(no_env, tmp_path, capsys):
    db = _db_with_item(tmp_path)
    for bad in ("-1", "0", "x"):
        with pytest.raises(SystemExit) as e:
            main(["--db", db, "events", "--limit", bad])
        assert e.value.code == 2
    assert main(["--db", db, "events", "--limit", "1", "--json"]) == 0


# ---------------------------------------------------------------- 9. export prompt
def _rewrite_once(tmp_path, *, author):
    seen: list[str] = []

    def llm(prompt: str) -> str:
        seen.append(prompt)
        return json.dumps({"ops": [{"op": "replace", "id": "c2", "text": "Pass the date as YYYY-MM-DD."}],
                           "rationale": "the API moved to ISO dates"})

    mem = rl.Memory(str(tmp_path / "x.db"), rewriter=rl.LLMRewriter(llm))
    mem.add("Fetch orders", ["Authenticate first.", "Pass the date as MM/DD/YYYY."], key="orders",
            tools=["get_orders"])
    kw = {"author": "agent"} if author == "agent" else {}
    with mem.episode("orders for March 14") as ep:
        ep.recall("fetch orders")
        for _ in range(2):
            e = ep.expect("orders", "returns rows", clauses=["c2"], **kw)
            ep.observe(e, error=rl.ToolError("invalid date: expected YYYY-MM-DD"))
    return mem, seen


@pytest.mark.parametrize("author", ["harness", "agent"])
def test_export_prompt_is_the_prompt_the_rewriter_saw(no_env, tmp_path, author):
    mem, seen = _rewrite_once(tmp_path, author=author)
    try:
        recs = list(rewrite_records(mem))
        assert len(recs) == 1 and len(seen) == 1
        assert "TASK CONTEXT: orders for March 14" in seen[0]
        assert recs[0]["prompt"] == seen[0]
    finally:
        mem.close()


def test_export_tolerates_a_missing_episode(no_env, tmp_path, monkeypatch):
    mem, seen = _rewrite_once(tmp_path, author="harness")

    def gone(episode_id):
        raise NotFoundError(f"no episode {episode_id!r}")

    monkeypatch.setattr(mem.store, "get_episode", gone)
    try:
        (rec,) = rewrite_records(mem)
        assert "TASK CONTEXT" not in rec["prompt"]
        assert rec["prompt"] == seen[0].replace("TASK CONTEXT: orders for March 14\n", "")
    finally:
        mem.close()


# ---------------------------------------------------------------- 10. Dockerfile healthcheck
def _healthcheck_code() -> str:
    path = Path(__file__).resolve().parents[1] / "Dockerfile"
    if not path.exists():
        pytest.skip("no Dockerfile in this checkout")
    text = path.read_text(encoding="utf-8").replace("\\\n", "")  # Docker's line continuations
    m = re.search(r'^HEALTHCHECK .*?\bCMD python -c "(.*)"[ \t]*$', text, re.M)
    assert m, "HEALTHCHECK runs python -c \"...\""
    return m.group(1)


@pytest.mark.parametrize(("argv", "port_env", "probed"), [
    (["relock", "serve"], "8765", "http://127.0.0.1:8765/v1/health"),
    (["relock", "serve"], "9000", "http://127.0.0.1:9000/v1/health"),
    (["relock", "serve"], "tcp://10.0.0.1:8765", "http://127.0.0.1:8765/v1/health"),
    (["relock", "serve"], None, "http://127.0.0.1:8765/v1/health"),
    (["relock", "--db", "/data/x.db", "serve"], "abc", "http://127.0.0.1:8765/v1/health"),
    (["relock", "mcp", "--transport", "stdio"], "8765", None),
    (None, "8765", "http://127.0.0.1:8765/v1/health"),  # no /proc: probe anyway
])
def test_docker_healthcheck(tmp_path, monkeypatch, argv, port_env, probed):
    code = _healthcheck_code()
    assert not set(code) & set('"$`\\'), "the command runs through /bin/sh inside double quotes"
    proc = tmp_path / "cmdline"
    if argv is not None:
        proc.write_bytes(b"/usr/local/bin/python\0/usr/local/bin/" + b"\0".join(a.encode() for a in argv) + b"\0")
    code = code.replace("/proc/1/cmdline", proc.as_posix())
    calls = []
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, *a, **kw: calls.append(url))
    if port_env is None:
        monkeypatch.delenv("RELOCK_PORT", raising=False)
    else:
        monkeypatch.setenv("RELOCK_PORT", port_env)
    exec(compile(code, "<healthcheck>", "exec"), {})
    assert calls == ([probed] if probed else [])
