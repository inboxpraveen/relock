from __future__ import annotations

import asyncio
import inspect

import pytest

import relock_memory as rl
from relock_memory import ErrorClass, checks, classify_exception
from relock_memory import delta as D
from relock_memory.llm import extract_json, strip_reasoning
from relock_memory.rewriters import Evidence, RewriteRequest, build_prompt
from relock_memory.types import Clause, Item

CLAUSES = (Clause("c1", "Authenticate first."), Clause("c2", "Send dates as MM/DD/YYYY."))


# ---------------------------------------------------------------- delta
def test_delta_apply_and_diff():
    ops = D.parse_ops([{"op": "replace", "id": "c2", "text": "Send dates as YYYY-MM-DD."},
                       {"op": "add", "text": "Use UTC.", "after": "c1"}])
    D.validate(CLAUSES, ops)
    new, seq = D.apply(CLAUSES, ops, 2)
    assert [c.id for c in new] == ["c1", "c3", "c2"] and seq == 3
    changes = D.diff(CLAUSES, new)
    assert {c["change"] for c in changes} == {"replaced", "added"}


@pytest.mark.parametrize("ops,msg", [
    ([], "empty"),
    ([{"op": "replace", "id": "c9", "text": "x"}], "does not exist"),
    ([{"op": "replace", "id": "c2", "text": ""}], "non-empty"),
    ([{"op": "remove", "id": "c1"}, {"op": "remove", "id": "c2"}], "no clauses"),
    ([{"op": "replace", "id": "c2", "text": "Send dates as MM/DD/YYYY."}], "does not change"),
    ([{"op": "frobnicate"}], "unknown op"),
])
def test_delta_validation(ops, msg):
    with pytest.raises(D.DeltaError, match=msg):
        D.validate(CLAUSES, D.parse_ops(ops))


def test_delta_op_aliases():
    ops = D.parse_ops([{"type": "update", "clause": "c1", "new_text": "x"}, {"action": "delete", "clause_id": "c2"}])
    assert [o.op for o in ops] == ["replace", "remove"]


# ---------------------------------------------------------------- signals
class _Status(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.status_code = code


@pytest.mark.parametrize("exc,cls", [
    (TimeoutError(), ErrorClass.TRANSIENT),
    (ConnectionRefusedError(), ErrorClass.TRANSIENT),
    (_Status(503), ErrorClass.TRANSIENT),
    (_Status(429), ErrorClass.TRANSIENT),
    (_Status(404), ErrorClass.PROGRAMMATIC),
    (ValueError("bad"), ErrorClass.PROGRAMMATIC),
    (KeyError("k"), ErrorClass.PROGRAMMATIC),
    (AssertionError(), ErrorClass.SEMANTIC),
    (type("ReadTimeout", (Exception,), {})(), ErrorClass.TRANSIENT),
    (type("ValidationError", (Exception,), {})(), ErrorClass.PROGRAMMATIC),
    (type("Weird", (Exception,), {})(), ErrorClass.UNKNOWN),
    (rl.ToolError("x", error_class="semantic"), ErrorClass.SEMANTIC),
])
def test_classify_exception(exc, cls):
    assert classify_exception(exc) is cls


def test_signal_roundtrip():
    s = rl.Signal.tool_error(rl.ToolError("bad", correction="do y"))
    assert s.correction == "do y" and s.error_class is ErrorClass.PROGRAMMATIC
    assert rl.Signal.from_dict(s.to_dict()) == s


def test_policy_admission():
    p = rl.WritePolicy()
    assert p.admit(rl.Signal.tool_error(ValueError("x"))).counts
    assert not p.admit(rl.Signal.transient()).counts
    assert not p.admit(rl.Signal.observation("x")).counts
    assert p.admit(rl.Signal.user("x", authenticated=True)).authoritative
    assert not p.admit(rl.Signal.user("x", authenticated=False)).counts
    assert not p.admit(rl.Signal.user("x")).counts  # must be vouched for explicitly
    assert rl.WritePolicy.trust_agent().admit(rl.Signal.agent_report(False, "x")).counts
    assert rl.WritePolicy.permissive().admit(rl.Signal.transient()).counts
    assert p.admit(rl.Signal.tool_error("weird")).weight == 0.5


# ---------------------------------------------------------------- checks
def test_checks():
    assert checks.succeeds()(1, None)[0]
    assert not checks.succeeds()(None, ValueError("x"))[0]
    assert checks.fails()(None, ValueError("x"))[0]
    assert checks.non_empty()([1], None)[0] and not checks.non_empty()([], None)[0]
    assert checks.has_keys("id")([{"id": 1}], None)[0]
    assert not checks.has_keys("id")([{"x": 1}], None)[0]
    assert checks.matches(r"\d{4}")("year 2026", None)[0]
    assert checks.equals(3)(3, None)[0]
    assert not checks.predicate(lambda r: 1 / 0)(1, None)[0]
    sch = checks.schema({"type": "array", "minItems": 1, "items": {"type": "object", "required": ["id"],
                         "properties": {"id": {"type": "integer"}, "date": {"type": "string", "pattern": r"^\d{4}"}}}})
    assert sch([{"id": 1, "date": "2026-01-01"}], None)[0]
    ok, why = sch([{"id": "1"}], None)
    assert not ok and "integer" in why
    assert not sch([{"id": True}], None)[0]  # bools are not integers
    assert checks.as_check(lambda r, e: (r == 2, "two"))(2, None) == (True, "two")


def test_check_resolves_observation(mem):
    item = mem.add("Count", ["count_rows returns the number of rows."], key="count")
    with mem.episode("t") as ep:
        ep.recall("count rows")
        exp = ep.expect(item, "returns 3", check=checks.equals(3))
        obs = ep.observe(exp, 4)
        assert not obs.matched and obs.verdict is rl.Verdict.PENDING  # trusted semantic mismatch


# ---------------------------------------------------------------- llm helpers
def test_extract_json_variants():
    assert extract_json('{"ops": []}') == {"ops": []}
    assert extract_json('<think>hmm {"no": 1}</think>\n```json\n{"ops": [1]}\n```') == {"ops": [1]}
    assert extract_json('Sure! Here you go: {"ops": [{"op": "add", "text": "a } b"}]} hope it helps') == {
        "ops": [{"op": "add", "text": "a } b"}]}
    assert strip_reasoning("partial reasoning</think>answer") == "answer"
    with pytest.raises(ValueError):
        extract_json("no json here")


def _request(editable=frozenset({"c2"})):
    item = Item(id="itm_x", version=1, title="Dates", clauses=CLAUSES, clause_seq=2)
    ev = Evidence("err_1", "api accepts the date", ("c2",), "get_orders", "raised ToolError", "tool",
                  "programmatic", "ToolError: invalid date", "Send dates as YYYY-MM-DD.")
    return RewriteRequest(item=item, evidence=(ev,), editable=editable)


def test_prompt_mentions_only_editable_clauses():
    p = build_prompt(_request())
    assert "Only these clauses may be replaced or removed: c2" in p
    assert "[c2] Send dates as MM/DD/YYYY." in p and "TRUSTED CORRECTION: Send dates as YYYY-MM-DD." in p


def test_llm_rewriter_retries_then_succeeds():
    replies = iter(["not json", '{"ops": [{"op": "replace", "id": "c1", "text": "x"}]}',
                    '<think>c2 is wrong</think>{"ops": [{"op": "replace", "id": "c2", "text": "Use YYYY-MM-DD."}],'
                    ' "rationale": "iso"}'])
    prompts = []

    def fake(prompt):
        prompts.append(prompt)
        return next(replies)

    rw = rl.LLMRewriter(fake, max_attempts=3, fallback=None)
    prop = rw.propose(_request())
    assert prop.ops[0].text == "Use YYYY-MM-DD." and prop.rationale == "iso"
    assert "locked" in prompts[2]  # the scope violation was fed back


def test_llm_rewriter_falls_back_to_hints():
    rw = rl.LLMRewriter(lambda p: "garbage", max_attempts=2)
    prop = rw.propose(_request())
    assert prop.ops[0].text == "Send dates as YYYY-MM-DD." and "+hint" in prop.author


def test_llm_rewriter_survives_endpoint_failure():
    def down(p):
        raise ConnectionError("refused")

    prop = rl.LLMRewriter(down).propose(_request())
    assert prop is not None and prop.ops[0].id == "c2"


def test_llm_rewriter_empty_ops_declines():
    assert rl.LLMRewriter(lambda p: '{"ops": [], "rationale": "flaky"}').propose(_request()) is None


def test_hint_rewriter_adds_when_nothing_similar():
    req = _request(editable=None)
    ev = Evidence("e", "x", (), None, "o", "tool", "programmatic", "m", "Always include a request id header.")
    req = RewriteRequest(item=req.item, evidence=(ev,), editable=None)
    assert rl.HintRewriter().propose(req).ops[0].op == "add"


# ---------------------------------------------------------------- tool wrapping
def test_wrap_preserves_signature_and_passthrough(mem):
    def search(query: str, limit: int = 5) -> list[str]:
        """Search things."""
        return [query] * limit

    w = mem.wrap(search)
    assert inspect.signature(w) == inspect.signature(search)
    assert w.__doc__ == "Search things." and w.__name__ == "search"
    assert w("a", limit=2) == ["a", "a"]  # outside an episode: plain call


def test_wrap_async_tool(mem):
    mem.add("Async fetch", ["Call afetch with an integer id."], key="afetch", tools=["afetch"])

    @mem.wrap
    async def afetch(id: int) -> dict:
        if not isinstance(id, int):
            raise TypeError("id must be int")
        return {"id": id}

    async def run():
        with mem.episode("t") as ep:
            ep.recall("async fetch id")
            await afetch(1)
            with pytest.raises(TypeError):
                await afetch("1")

    asyncio.run(run())
    it = mem.get("afetch")
    assert it.confirmations == 1 and it.failures == 1


def test_wrap_binds_explicit_expectation(mem, orders):
    item, get_orders = orders
    with mem.episode("t") as ep:
        ep.recall("fetch orders")
        exp = ep.expect(item, "returns orders for March 5", clauses=["c2"], tool="get_orders",
                        check=checks.non_empty())
        with pytest.raises(rl.ToolError):
            get_orders("03/05/2026")
    row = mem.store.get_expectation(exp.id)
    assert row["status"] == "violated"
    errs = mem.store.errors(item.id, 1)
    assert errs[0]["diff"]["clause_ids"] == ["c2"]


def test_wrap_does_not_attribute_undeclared_items(mem):
    mem.add("Generic", ["Be careful."], key="generic")

    @mem.wrap
    def tool_x():
        raise ValueError("nope")

    with mem.episode("t") as ep:
        ep.recall("generic careful")
        with pytest.raises(ValueError):
            tool_x()
    assert mem.get("generic").failures == 0


# ---------------------------------------------------------------- persistence & retrieval
def test_state_survives_reopen(tmp_path):
    path = str(tmp_path / "agent.db")
    with rl.Memory(path) as m:
        item = m.add("Fetch", ["Call get with MM/DD/YYYY."], key="f")
        for _ in range(2):
            with m.episode() as ep:
                ep.recall("fetch get")
                exp = ep.expect(item, clauses=["c1"])
                ep.observe(exp, error=rl.ToolError("bad", correction="Call get with YYYY-MM-DD."))
    with rl.Memory(f"sqlite:///{path}") as m2:
        assert m2.get("f").version == 2
        assert m2.get("f").clause("c1").text == "Call get with YYYY-MM-DD."


def test_cross_instance_cache_invalidation(tmp_path):
    path = str(tmp_path / "shared.db")
    a, b = rl.Memory(path), rl.Memory(path)
    a.add("Alpha procedure", ["Do alpha things."], key="alpha")
    assert [h.item.key for h in b.search("alpha")] == ["alpha"]
    b.add("Beta procedure", ["Do beta things."], key="beta")
    assert [h.item.key for h in a.search("beta")] == ["beta"]
    a.close()
    b.close()


def test_hybrid_retrieval_with_embeddings(mem):
    vocab = ["date", "email", "payment"]

    def embed(texts):
        return [[float(w in t.lower()) for w in vocab] for t in texts]

    m = rl.Memory(mem.store, embed=embed)
    m.add("Dates", ["Format the date properly."], key="d")
    m.add("Mail", ["Send the email."], key="e")
    assert m.search("date", k=1)[0].item.key == "d"


def test_recall_render_budget(mem, orders):
    with mem.episode("t") as ep:
        r = ep.recall("orders email", k=5)
        full = r.render()
        tight = r.render(budget_tokens=10)
        assert full.count("###") == 2 and tight.count("###") == 1


def test_archive_hides_item(mem, orders):
    mem.archive("email.send")
    assert all(h.item.key != "email.send" for h in mem.search("send email recipients"))
    mem.archive("email.send", False)
    assert any(h.item.key == "email.send" for h in mem.search("send email recipients"))


def test_events_and_stats(mem, orders):
    _, get_orders = orders
    seen = []
    m = rl.Memory(mem.store, on_event=seen.append)
    with m.episode("t") as ep:
        ep.recall("fetch orders")
        try:
            get_orders("bad")
        except rl.ToolError:
            pass
    types = [e["type"] for e in seen]
    assert types[0] == "episode_open" and "recall" in types and types[-1] == "relock"
    assert m.stats()["episodes"] >= 1


def test_ids_are_unique_and_sort_in_creation_order():
    from relock_memory._util import new_id

    ids = [new_id("x") for _ in range(20000)]  # many share a millisecond
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)


def test_hint_rewriter_adds_when_citation_points_away_from_the_fix():
    from relock_memory.rewriters import HintRewriter

    item = Item(id="i", version=1, title="Fetch", clauses=(
        Clause("c1", "Authenticate with the orders API key before calling get_orders."),
        Clause("c2", "Pass the date to get_orders as MM/DD/YYYY.")))
    ev = Evidence(error_id="e", expectation="", clause_ids=("c1",), tool="get_orders", observed="",
                  source="tool", error_class="programmatic", message="",
                  correction="Pass the date to get_orders as YYYY-MM-DD.", authoritative=False)
    suspect = HintRewriter().propose(RewriteRequest(item=item, evidence=(ev,), editable=frozenset({"c1"})))
    assert [o.op for o in suspect.ops] == ["add"]
    aimed = HintRewriter().propose(RewriteRequest(item=item, evidence=(ev,), editable=frozenset({"c2"})))
    assert [(o.op, o.id) for o in aimed.ops] == [("replace", "c2")]


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs os.fork")
def test_forked_children_do_not_continue_the_parents_id_sequence():
    import os

    from relock_memory import _util

    _util.new_id("x")
    assert _util._id_last[0] != 0
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: report the inherited sequence state, then exit without pytest teardown
        os.write(w, repr(list(_util._id_last)).encode())
        os._exit(0)
    os.close(w)
    state = os.read(r, 100).decode()
    os.waitpid(pid, 0)
    assert state == "[0, 0]"


def test_explicit_failed_signal_is_not_overridden_by_the_check():
    import relock_memory as rl

    mem = rl.Memory(":memory:")
    mem.add("Fetch", ["Pass dates as MM/DD/YYYY."], key="f", tools=["get"])
    with mem.episode("t") as ep:
        ep.recall("fetch dates")
        obs = ep.tool_result("get", signal=rl.Signal.tool_error("bad date", correction="Pass dates as YYYY-MM-DD."))
        assert [o.matched for o in obs] == [False]
        assert obs[0].verdict.value == "pending"
        exp = ep.expect("f", "get works", tool="get")
        assert ep.observe(exp, signal=rl.Signal.tool_error("bad date")).matched is False
        assert ep.tool_result("unknown_tool", error="x") == []
    with mem.episode("t2") as ep:
        ep.recall("fetch dates")
        assert ep.tool_result("get", [1, 2])[0].verdict.value == "confirmed"
    mem.close()
