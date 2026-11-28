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
