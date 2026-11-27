"""``relock`` command line: inspect, audit, correct and serve agent memory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from . import __version__
from .delta import diff as clause_diff
from .memory import Memory
from .types import Item, RelockError


def _out(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _item_line(i: Item) -> str:
    tools = f" tools={','.join(i.tools)}" if i.tools else ""
    return f"{i.ref:<28} v{i.version:<3} conf={i.confidence:.2f} ok={i.confirmations:<4} fail={i.failures:<4}{tools}  {i.title}"


def _llm_rewriter(args: argparse.Namespace) -> Any:
    if not getattr(args, "llm_base_url", None):
        return None
    from .llm import OpenAICompatible
    from .rewriters import LLMRewriter

    return LLMRewriter(OpenAICompatible(args.llm_base_url, args.llm_model, api_key=args.llm_api_key))


def cmd_init(args: argparse.Namespace, mem: Memory) -> int:
    print(f"relock ledger ready: {mem.store!r}")
    return 0


def cmd_add(args: argparse.Namespace, mem: Memory) -> int:
    item = mem.add(args.title, args.clause, key=args.key, kind=args.kind, tags=args.tag or (),
                   tools=args.tool or (), exist_ok=args.exist_ok)
    _out(item.to_dict()) if args.json else print(item.render())
    return 0


def cmd_ls(args: argparse.Namespace, mem: Memory) -> int:
    items = mem.items(include_archived=args.all)
    if args.json:
        _out([i.to_dict() for i in items])
        return 0
    if not items:
        print("(no memory items)")
    for i in items:
        print(_item_line(i) + ("  [archived]" if i.archived else ""))
    return 0


def cmd_show(args: argparse.Namespace, mem: Memory) -> int:
    item = mem.get(args.ref, args.version)
    if args.json:
        _out(item.to_dict())
        return 0
    print(item.render())
    print(f"\nstatus={item.status} change={item.change_kind} author={item.author} created={item.created_at}")
    if item.reason:
        print(f"reason: {item.reason}")
    return 0


def cmd_history(args: argparse.Namespace, mem: Memory) -> int:
    versions = mem.history(args.ref)
    if args.json:
        _out([v.to_dict() for v in versions])
        return 0
    prev = None
    for v in versions:
        head = f"v{v.version} [{v.status}] {v.change_kind} by {v.author or '?'} at {v.created_at}"
        if v.episode_id:
            head += f" (episode {v.episode_id})"
        print(head)
        if v.reason:
            print(f"    reason: {v.reason}")
        if prev is None:
            for c in v.clauses:
                print(f"    ({c.id}) {c.text}")
        else:
            for ch in clause_diff(prev.clauses, v.clauses):
                if ch["change"] == "replaced":
                    print(f"  - ({ch['id']}) {ch['old']}\n  + ({ch['id']}) {ch['new']}")
                elif ch["change"] == "added":
                    print(f"  + ({ch['id']}) {ch['new']}")
                else:
                    print(f"  - ({ch['id']}) {ch['old']}")
        prev = v
    return 0


def cmd_pending(args: argparse.Namespace, mem: Memory) -> int:
    _out(mem.pending(args.ref))
    return 0


def cmd_rollback(args: argparse.Namespace, mem: Memory) -> int:
    item = mem.rollback(args.ref, args.version, reason=args.reason or "", actor=args.actor)
    print(f"{item.ref}: restored v{args.version} as v{item.version}")
    return 0


def cmd_edit(args: argparse.Namespace, mem: Memory) -> int:
    ops: list[dict[str, Any]] = []
    for cid, text in args.replace or ():
        ops.append({"op": "replace", "id": cid, "text": text})
    for text in args.add or ():
        ops.append({"op": "add", "text": text, "after": args.after})
    for cid in args.remove or ():
        ops.append({"op": "remove", "id": cid})
    if not ops:
        print("nothing to do: pass --replace, --add or --remove", file=sys.stderr)
        return 2
    item = mem.edit(args.ref, ops, reason=args.reason or "", actor=args.actor)
    print(item.render())
    return 0


def cmd_archive(args: argparse.Namespace, mem: Memory) -> int:
    mem.archive(args.ref, not args.undo)
    print(f"{args.ref}: {'unarchived' if args.undo else 'archived'}")
    return 0


def cmd_events(args: argparse.Namespace, mem: Memory) -> int:
    events = mem.events(args.item, episode_id=args.episode, type=args.type, limit=args.limit)
    if args.json:
        _out(events)
        return 0
    for e in reversed(events):
        bits = [e["at"][:19], f"{e['type']:<17}"]
        if e.get("item_id"):
            bits.append(f"{e['item_id']}@v{e['version']}")
        data = {k: v for k, v in (e.get("data") or {}).items() if k not in ("hits", "clauses", "changes")}
        bits.append(json.dumps(data, ensure_ascii=False, default=str)[:160])
        print("  ".join(bits))
    return 0


def cmd_stats(args: argparse.Namespace, mem: Memory) -> int:
    _out(mem.stats())
    return 0


def cmd_export(args: argparse.Namespace, mem: Memory) -> int:
    from .export import export_jsonl

    n = export_jsonl(mem, args.out)
    print(f"wrote {n} rewrite trace(s) to {args.out}")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    try:
        from .integrations.mcp import serve
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    verifiers: dict[str, str] = {}
    for spec in args.verifier or ():
        name, sep, command = spec.partition("=")
        if not sep or not name.strip() or not command.strip():
            print(f"bad --verifier {spec!r}; use NAME=COMMAND", file=sys.stderr)
            return 2
        verifiers[name.strip()] = command.strip()
    if args.cwd is not None and not os.path.isdir(args.cwd):
        print(f"error: --cwd {args.cwd!r} is not a directory", file=sys.stderr)
        return 2
    serve(args.db, transport=args.transport, rewriter=_llm_rewriter(args), verifiers=verifiers,
          verifier_timeout=args.verifier_timeout, verifier_cwd=args.cwd,
          trust_agent_reports=args.trust_agent_reports,
          allow_teach=args.allow_teach, idle_timeout=args.idle_timeout or None, k=args.k)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="relock", description="Agent memory that unlocks on recall, rewrites on "
                                                           "surprise, and re-locks.")
    p.add_argument("--version", action="version", version=f"relock {__version__}")
    p.add_argument("--db", default=os.environ.get("RELOCK_DB", "relock.db"),
                   help="SQLite path or postgresql:// URL (default: $RELOCK_DB or ./relock.db)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn: Any, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.set_defaults(fn=fn)
        return sp

    add("init", cmd_init, "Create the ledger tables.")
    sp = add("add", cmd_add, "Add a memory item.")
    sp.add_argument("title")
    sp.add_argument("clause", nargs="+", help="one argument per clause")
    sp.add_argument("--key")
    sp.add_argument("--kind", default="procedure")
    sp.add_argument("--tag", action="append")
    sp.add_argument("--tool", action="append", help="tool name this item governs (repeatable)")
    sp.add_argument("--exist-ok", action="store_true")
    sp.add_argument("--json", action="store_true")

    sp = add("ls", cmd_ls, "List active items.")
    sp.add_argument("--all", action="store_true", help="include archived items")
    sp.add_argument("--json", action="store_true")

    sp = add("show", cmd_show, "Show one item (current or a given version).")
    sp.add_argument("ref")
    sp.add_argument("--version", type=int)
    sp.add_argument("--json", action="store_true")

    sp = add("history", cmd_history, "Show every version of an item as clause diffs.")
    sp.add_argument("ref")
    sp.add_argument("--json", action="store_true")

    sp = add("pending", cmd_pending, "Explain pending evidence vs. the rewrite threshold.")
    sp.add_argument("ref")

    sp = add("rollback", cmd_rollback, "Restore an earlier version (as a new version).")
    sp.add_argument("ref")
    sp.add_argument("version", type=int)
    sp.add_argument("--reason")
    sp.add_argument("--actor", default=os.environ.get("USER") or os.environ.get("USERNAME") or "admin")

    sp = add("edit", cmd_edit, "Manually edit clauses (versioned and audited).")
    sp.add_argument("ref")
    sp.add_argument("--replace", nargs=2, action="append", metavar=("CLAUSE_ID", "TEXT"))
    sp.add_argument("--add", action="append", metavar="TEXT")
    sp.add_argument("--after", metavar="CLAUSE_ID")
    sp.add_argument("--remove", action="append", metavar="CLAUSE_ID")
    sp.add_argument("--reason")
    sp.add_argument("--actor", default=os.environ.get("USER") or os.environ.get("USERNAME") or "admin")

    sp = add("archive", cmd_archive, "Hide an item from retrieval (or --undo).")
    sp.add_argument("ref")
    sp.add_argument("--undo", action="store_true")

    sp = add("events", cmd_events, "Audit log of recalls, predictions, gate decisions and rewrites.")
    sp.add_argument("--item")
    sp.add_argument("--episode")
    sp.add_argument("--type")
    sp.add_argument("--limit", type=int, default=50)
    sp.add_argument("--json", action="store_true")

    add("stats", cmd_stats, "Ledger counters.")
    sp = add("export", cmd_export, "Export rewrite traces as JSONL (SFT / preference pairs).")
    sp.add_argument("out")

    sp = sub.add_parser("mcp", help="Run the MCP server.", description="Run the relock MCP server.")
    sp.set_defaults(fn=None, mcp=True)
    sp.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    sp.add_argument("--verifier", action="append", metavar="NAME=COMMAND",
                    help="trusted check the agent may run by name, e.g. tests='pytest -q' (repeatable)")
    sp.add_argument("--verifier-timeout", type=float, default=600.0)
    sp.add_argument("--cwd", metavar="DIR",
                    help="directory the verifier commands run in (default: the server's working directory)")
    sp.add_argument("--trust-agent-reports", action="store_true",
                    help="let the agent's own outcome reports rewrite memory (single-user sandboxes only)")
    sp.add_argument("--allow-teach", action="store_true",
                    help="expose relock_teach so the agent can add new items (agent-written content)")
    sp.add_argument("--idle-timeout", type=float, default=1800.0,
                    help="re-lock an episode after this many idle seconds (0 disables; default 1800)")
    sp.add_argument("-k", type=int, default=3, help="items per recall")
    for target in (sp,):
        target.add_argument("--llm-base-url", default=os.environ.get("RELOCK_LLM_BASE_URL"),
                            help="OpenAI-compatible endpoint for the LLM rewriter (default: hint rewriter)")
        target.add_argument("--llm-model", default=os.environ.get("RELOCK_LLM_MODEL"))
        target.add_argument("--llm-api-key", default=os.environ.get("RELOCK_LLM_API_KEY"))

    sp = sub.add_parser("bench", help="Run RelapseBench.", add_help=False)
    sp.set_defaults(fn=None, bench=True)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)
    if getattr(args, "bench", False):
        from .bench.cli import main as bench_main

        return bench_main(rest)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    if getattr(args, "mcp", False):
        return cmd_mcp(args)
    mem = Memory(args.db)
    try:
        return int(args.fn(args, mem) or 0)
    except (RelockError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        mem.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
