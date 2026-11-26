from __future__ import annotations

import os
import re

import pytest

import relock_memory as rl

PG_URL = os.environ.get("RELOCK_TEST_POSTGRES_URL")

_backends = ["sqlite-memory", "sqlite-file"]
if PG_URL:
    _backends.append("postgres")


@pytest.fixture(params=_backends)
def store(request, tmp_path):
    if request.param == "sqlite-memory":
        s = rl.open_store(":memory:")
    elif request.param == "sqlite-file":
        s = rl.open_store(str(tmp_path / "relock.db"))
    else:
        from relock_memory.stores.postgres import PostgresStore

        s = PostgresStore(PG_URL)
        s.drop_all()
        s.close()
        s = PostgresStore(PG_URL)
    yield s
    s.close()


@pytest.fixture
def mem(store):
    m = rl.Memory(store)
    yield m


ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@pytest.fixture
def orders(mem):
    """An item with a stale date-format clause, plus a wrapped tool that enforces ISO dates."""
    item = mem.add(
        "Fetch orders",
        ["Authenticate with the orders API key.", "Call get_orders with the date as MM/DD/YYYY."],
        key="orders.fetch",
        tools=["get_orders"],
        tags=["orders"],
    )
    mem.add("Send email", ["Use send_email with a comma separated list of recipients."],
            key="email.send", tools=["send_email"])

    @mem.wrap
    def get_orders(date: str) -> list[dict]:
        """Fetch orders for one day."""
        if not ISO.match(date):
            raise rl.ToolError(f"invalid date {date!r}",
                               correction="Call get_orders with the date as YYYY-MM-DD (ISO-8601).")
        return [{"id": 1, "date": date}]

    return item, get_orders
