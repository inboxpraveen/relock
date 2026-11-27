from __future__ import annotations

import json

import pytest

import relock_memory as rl
from relock_memory.bench.agents import ScriptedAgent
from relock_memory.bench.harness import LOADS, Config, aggregate, run_behavioral, run_structural
from relock_memory.bench.report import write_report
from relock_memory.bench.scenarios import FAMILIES
from relock_memory.cli import main
from relock_memory.export import export_jsonl, rewrite_records


def _learn(mem):
    item = mem.add("Fetch", ["Authenticate first.", "Call get with MM/DD/YYYY."], key="f", tools=["get"])
    for _ in range(2):
        with mem.episode("t") as ep:
            ep.recall("fetch get")
            exp = ep.expect(item, "get returns rows", clauses=["c2"])
            ep.observe(exp, error=rl.ToolError("bad date", correction="Call get with YYYY-MM-DD."))
    return item


# ---------------------------------------------------------------- export
def test_export_rewrite_records(tmp_path):
    mem = rl.Memory(str(tmp_path / "m.db"))
    _learn(mem)
    recs = list(rewrite_records(mem))
    assert len(recs) == 1
    r = recs[0]
    assert r["rejected"] == ["Authenticate first.", "Call get with MM/DD/YYYY."]
    assert r["chosen"] == ["Authenticate first.", "Call get with YYYY-MM-DD."]
    assert json.loads(r["completion"])["ops"][0]["id"] == "c2"
    assert "[c2] Call get with MM/DD/YYYY." in r["prompt"] and len(r["evidence"]) == 2
    assert export_jsonl(mem, str(tmp_path / "out.jsonl")) == 1
    mem.close()


# ---------------------------------------------------------------- CLI
def test_cli_roundtrip(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "add", "Fetch orders", "Call get_orders with MM/DD/YYYY.", "--key", "orders",
                 "--tool", "get_orders"]) == 0
    mem = rl.Memory(db)
    with mem.episode() as ep:
        ep.recall("fetch orders")
        ep.correct("orders", "Call get_orders with YYYY-MM-DD.", principal="cli-test", authenticated=True)
    mem.close()
    capsys.readouterr()
    assert main(["--db", db, "history", "orders"]) == 0
    out = capsys.readouterr().out
    assert "- (c1) Call get_orders with MM/DD/YYYY." in out and "+ (c1) Call get_orders with YYYY-MM-DD." in out
    assert main(["--db", db, "rollback", "orders", "1", "--reason", "test"]) == 0
    assert main(["--db", db, "ls", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert listed[0]["version"] == 3 and listed[0]["change_kind"] == "rollback"
    assert main(["--db", db, "edit", "orders", "--add", "Use UTC dates."]) == 0
    assert main(["--db", db, "pending", "orders"]) == 0
    assert main(["--db", db, "events", "--item", "orders", "--limit", "5"]) == 0
    assert main(["--db", db, "stats"]) == 0
    assert main(["--db", db, "export", str(tmp_path / "t.jsonl")]) == 0
    assert main(["--db", db, "show", "missing"]) == 1


# ---------------------------------------------------------------- RelapseBench
def test_structural_bench_mechanism():
    cfg = Config(seeds=(0,), loads=(LOADS[0], LOADS[-1]))
    agg = aggregate(run_structural(cfg))
    for load in ("L0", "L3"):
        relock = agg["relock"][load]
        assert relock["stale_exposure"]["rate"] == 0.0
        assert relock["poison_exposure"]["rate"] == 0.0
        assert relock["clean"]["rate"] == 1.0
        # each ablation fails on exactly the axis whose mechanism it lacks
        assert agg["append-gated"][load]["poison_exposure"]["rate"] == 0.0
        assert agg["append-gated"][load]["stale_exposure"]["rate"] > 0.5
        assert agg["overwrite-ungated"][load]["stale_exposure"]["rate"] == 0.0
        assert agg["overwrite-ungated"][load]["poison_exposure"]["rate"] == 1.0
    assert agg["relock"]["L0"]["learn_errors_mean"] == 2.0  # the price of the evidence threshold
    assert agg["append-notes"]["L0"]["learn_errors_mean"] == 1.0


def test_families_are_unambiguous():
    import random

    for fam in FAMILIES:
        assert fam.which(fam.old_rule) == {"old"}, fam.key
        assert fam.which(fam.new_rule) == {"new"}, fam.key
        assert fam.which(fam.poison_rule) == {"poison"}, fam.key
        assert not fam.which(" ".join(fam.stable)), fam.key
        _, variants = fam.make(random.Random(0))
        assert len(set(variants.values())) == 3, fam.key
        for k, v in variants.items():
            assert fam.classify(v, variants) == k


def test_behavioral_pipeline_and_report(tmp_path):
    cfg = Config(seeds=(0,), loads=(LOADS[0],), probes=1, systems=("append-notes", "relock"))
    recs = run_behavioral(cfg, ScriptedAgent())
    assert {r.outcome for r in recs} <= {"new", "old", "poison", "other"}
    paths = write_report(str(tmp_path), recs, cfg, "behavioral", "scripted", pipeline_test=True)
    from pathlib import Path

    assert "Pipeline test only" in Path(paths["md"]).read_text(encoding="utf-8")
    svg = Path(paths["svg"]).read_text(encoding="utf-8")
    assert svg.startswith("<svg") and "prefers-color-scheme: dark" in svg
    assert json.loads(Path(paths["json"]).read_text(encoding="utf-8"))["pipeline_test"] is True


def test_llm_agent_parsing():
    from relock_memory.bench.agents import LLMAgent

    fam = FAMILIES[0]
    agent = LLMAgent(lambda p: '<think>iso</think>{"tool": "get_orders", "args": {"date": "2026-03-14"}}')
    assert agent.act(fam, "t", "", "", {"old": "x", "new": "y", "poison": "z"}) == "2026-03-14"
    broken = LLMAgent(lambda p: "I refuse")
    assert broken.act(fam, "t", "", "", {"old": "x", "new": "y", "poison": "z"}) == ""
    assert broken.failures == 1


# ---------------------------------------------------------------- mcp cli
def test_cli_mcp_passes_verifiers_and_cwd(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    import relock_memory.integrations.mcp as mcp_mod

    seen = {}
    monkeypatch.setattr(mcp_mod, "serve", lambda db, **kw: seen.update(db=db, **kw))
    db = str(tmp_path / "m.db")
    assert main(["--db", db, "mcp", "--verifier", "tests=pytest -q", "--cwd", str(tmp_path)]) == 0
    assert seen["db"] == db
    assert seen["verifiers"] == {"tests": "pytest -q"}
    assert seen["verifier_cwd"] == str(tmp_path)
    assert main(["--db", db, "mcp", "--cwd", str(tmp_path / "missing")]) == 2
    assert main(["--db", db, "mcp", "--verifier", "no-command"]) == 2
