"""``relock`` command line: inspect, audit, correct and serve agent memory."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import timedelta
from typing import Any

from . import __version__
from .delta import diff as clause_diff
from .memory import KEEP_EVENTS, Memory
from .stores import SQLiteStore
from .types import Item, RelockError

log = logging.getLogger("relock_memory.cli")

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
DEFAULT_PORT = 8765
MIN_PRUNE_AGE = timedelta(hours=1)
MAX_AGE = timedelta(days=36525)  # 100 years; beyond this, "now - age" leaves the calendar


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
    targets = [cid for cid, _ in args.replace or ()] + list(args.remove or ())
    twice = sorted({cid for cid in targets if targets.count(cid) > 1})
    if twice:
        print(f"error: clause {', '.join(twice)} is named by more than one --replace/--remove; "
              "give each clause one operation", file=sys.stderr)
        return 2
    if args.after is not None and not args.add:
        print("error: --after places --add clauses; pass --add TEXT", file=sys.stderr)
        return 2
    ops: list[dict[str, Any]] = []
    for cid, text in args.replace or ():
        ops.append({"op": "replace", "id": cid, "text": text})
    adds = list(args.add or ())
    if args.after is not None:
        adds.reverse()  # each lands right after --after, so insert the last first to keep the given order
    for text in adds:
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


def _positive_int(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"invalid count {text!r}: use a whole number of at least 1")
    return n


def _parse_port(text: str) -> int | None:
    text = text.strip()
    return int(text) if text.isascii() and text.isdigit() and int(text) <= 65535 else None


def _port_arg(text: str) -> int:
    port = _parse_port(text)
    if port is None:
        raise argparse.ArgumentTypeError(f"invalid port {text!r}: use a number from 0 to 65535")
    return port


def _env_port() -> int:
    raw = os.environ.get("RELOCK_PORT", "")
    if not raw.strip():
        return DEFAULT_PORT
    port = _parse_port(raw)
    if port is None:  # e.g. Kubernetes sets RELOCK_PORT=tcp://<ip>:<port> for a Service named "relock"
        log.warning("ignoring RELOCK_PORT=%r: not a port number; serving on %d (use --port to choose)",
                    raw, DEFAULT_PORT)
        return DEFAULT_PORT
    return port


def _age_arg(text: str) -> str:
    from ._util import parse_age

    number = text.strip()[:-1]
    try:
        finite = math.isfinite(float(number))
    except ValueError:
        finite = True  # not a number at all: parse_age explains the format
    if not finite:
        raise argparse.ArgumentTypeError(f"invalid age {text!r}: not a finite number")
    try:
        age = parse_age(text)
    except OverflowError:
        age = MAX_AGE + MAX_AGE
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    if age > MAX_AGE:
        raise argparse.ArgumentTypeError(f"invalid age {text!r}: at most 100 years (36525d)")
    return text


def cmd_prune(args: argparse.Namespace, mem: Memory) -> int:
    from ._util import parse_age

    if parse_age(args.older_than) < MIN_PRUNE_AGE and not (args.force or args.dry_run):
        print(f"error: refusing --older-than {args.older_than} (under 1h) without --force. Prune can't see "
              "episodes open in other processes (agents, 'relock serve', 'relock mcp'): one idle past the "
              "cutoff is marked abandoned and its detail deleted, so its next writes fail and its evidence "
              "is lost. Use an age longer than any episode stays open (e.g. 30d), or pass --force if nothing "
              "else uses this database.", file=sys.stderr)
        return 2
    keep = tuple(dict.fromkeys((*KEEP_EVENTS, *(args.keep_events or ()))))
    counts = mem.prune(args.older_than, dry_run=args.dry_run, keep_events=keep)
    if args.json:
        _out(counts)
        return 0
    verb = "would delete" if args.dry_run else "deleted"
    print(f"{verb} {counts['episodes']} episode(s) older than {args.older_than}: {counts['expectations']} "
          f"expectation(s), {counts['errors']} error(s), {counts['retrievals']} retrieval(s), "
          f"{counts['events']} event(s)")
    print(f"{verb} per-call detail of kept episodes: {counts['expectations_trimmed']} expectation(s), "
          f"{counts['events_trimmed']} event(s)")
    if counts["episodes_abandoned"]:
        print(f"{'would mark' if args.dry_run else 'marked'} {counts['episodes_abandoned']} idle open episode(s) "
              "as abandoned")
    if not args.dry_run and isinstance(mem.store, SQLiteStore):
        print("SQLite keeps the file size until you run: sqlite3 <db> VACUUM")
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
    from .integrations.mcp import _load_server_class, serve

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
    _load_server_class()  # without the mcp extra, fail before touching the database
    serve(args.db, transport=args.transport, rewriter=_llm_rewriter(args), verifiers=verifiers,
          verifier_timeout=args.verifier_timeout, verifier_cwd=args.cwd,
          trust_agent_reports=args.trust_agent_reports,
          trust_verifier_output=not args.untrusted_verifier_output,
          allow_teach=args.allow_teach, idle_timeout=args.idle_timeout or None, k=args.k)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .integrations.http import serve

    token = os.environ.get("RELOCK_TOKEN") or None
    port = _env_port() if args.port is None else args.port
    serve(args.db, host=args.host, port=port, token=token, rewriter=_llm_rewriter(args),
          idle_timeout=args.idle_timeout or None, k=args.k)
    return 0


def _add_global_options(p: argparse.ArgumentParser, default: Any) -> None:
    # Environment variables are read after parsing, so a bad one only affects the command that uses it.
    p.add_argument("--log-level", default=default, choices=LOG_LEVELS, type=str.upper,
                   help="log relock events to stderr at this level (default: $RELOCK_LOG_LEVEL; "
                        "'serve' logs INFO when unset)")
    p.add_argument("--db", default=default,
                   help="SQLite path, sqlite:/// or file: URI, or postgresql:// URL (default: $RELOCK_DB or "
                        "./relock.db). $RELOCK_PG_POOL_SIZE sizes the Postgres pool (default 5)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="relock", description="Agent memory that unlocks on recall, rewrites on "
                                                           "surprise, and re-locks.")
    p.add_argument("--version", action="version", version=f"relock {__version__}")
    _add_global_options(p, None)
    sub = p.add_subparsers(dest="cmd", required=True)

    def parser(name: str, **kw: Any) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, **kw)
        _add_global_options(sp, argparse.SUPPRESS)  # also accepted after the subcommand
        return sp

    def add(name: str, fn: Any, help_: str) -> argparse.ArgumentParser:
        sp = parser(name, help=help_, description=help_)
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
    sp.add_argument("--limit", type=_positive_int, default=50, help="most recent events to show (default 50)")
    sp.add_argument("--json", action="store_true")

    sp = add("prune", cmd_prune, "Delete old audit detail (items and versions are always kept).")
    sp.add_argument("--older-than", required=True, metavar="AGE", type=_age_arg,
                    help="e.g. 30d, 12h, 2w; must exceed the longest an episode stays open in any process")
    sp.add_argument("--dry-run", action="store_true", help="only count what would be deleted")
    sp.add_argument("--force", action="store_true",
                    help="allow --older-than under 1h (only when no other process is using the database)")
    sp.add_argument("--keep-events", action="append", metavar="TYPE",
                    help="also keep events of this type, e.g. blocked (repeatable; added to the kept "
                         f"defaults: {', '.join(KEEP_EVENTS)})")
    sp.add_argument("--json", action="store_true")

    add("stats", cmd_stats, "Ledger counters.")
    sp = add("export", cmd_export, "Export rewrite traces as JSONL (SFT / preference pairs).")
    sp.add_argument("out")

    sp = parser("mcp", help="Run the MCP server.", description="Run the relock MCP server.")
    sp.set_defaults(fn=None, mcp=True)
    sp.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    sp.add_argument("--verifier", action="append", metavar="NAME=COMMAND",
                    help="trusted check the agent may run by name, e.g. tests='pytest -q' (repeatable)")
    sp.add_argument("--verifier-timeout", type=float, default=600.0)
    sp.add_argument("--cwd", metavar="DIR",
                    help="directory the verifier commands run in (default: the server's working directory)")
    sp.add_argument("--trust-agent-reports", action="store_true",
                    help="let the agent's own outcome reports rewrite memory (single-user sandboxes only)")
    sp.add_argument("--untrusted-verifier-output", action="store_true",
                    help="keep verifier output away from the rewriter; pass/fail still counts (use when the "
                         "agent can edit what a verifier prints, such as its tests)")
    sp.add_argument("--allow-teach", action="store_true",
                    help="expose relock_teach so the agent can add new items (agent-written content)")
    sp.add_argument("--idle-timeout", type=float, default=1800.0,
                    help="re-lock an episode after this many idle seconds (0 disables; default 1800)")
    sp.add_argument("-k", type=int, default=3, help="items per recall")
    serve_p = parser(
        "serve", help="Run the HTTP/JSON API (any language).",
        description="Run relock's HTTP/JSON API for harnesses in any language. Set RELOCK_TOKEN to require "
                    "'Authorization: Bearer <token>'; without it the server only listens on loopback.")
    serve_p.set_defaults(fn=None, serve=True)
    serve_p.add_argument("--host", default=os.environ.get("RELOCK_HOST", "127.0.0.1"))
    serve_p.add_argument("--port", type=_port_arg, default=None,
                         help=f"default: $RELOCK_PORT if it is a port number, else {DEFAULT_PORT}")
    serve_p.add_argument("--idle-timeout", type=float, default=1800.0,
                         help="re-lock an episode after this many idle seconds (0 disables; default 1800)")
    serve_p.add_argument("-k", type=int, default=3, help="default items per recall")
    for target in (sp, serve_p):
        target.add_argument("--llm-base-url", default=os.environ.get("RELOCK_LLM_BASE_URL"),
                            help="OpenAI-compatible endpoint for the LLM rewriter (default: hint rewriter)")
        target.add_argument("--llm-model", default=os.environ.get("RELOCK_LLM_MODEL"))
        target.add_argument("--llm-api-key", default=os.environ.get("RELOCK_LLM_API_KEY"))

    sp = sub.add_parser("bench", help="Run RelapseBench.", add_help=False)
    sp.set_defaults(fn=None, bench=True)
    return p


def _log_level(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str | None:
    if args.log_level:  # the flag, already checked against LOG_LEVELS
        return str(args.log_level)
    raw = os.environ.get("RELOCK_LOG_LEVEL", "").strip()
    if raw and raw.upper() not in LOG_LEVELS:
        parser.error(f"RELOCK_LOG_LEVEL={raw!r} is not a log level; use one of {', '.join(LOG_LEVELS)}")
    return raw.upper() or None


def _run(args: argparse.Namespace) -> int:
    if getattr(args, "mcp", False):
        return cmd_mcp(args)
    if getattr(args, "serve", False):
        return cmd_serve(args)
    mem = Memory(args.db)
    try:
        return int(args.fn(args, mem) or 0)
    finally:
        mem.close()


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
    level = _log_level(parser, args)
    if level:  # stderr only: stdout carries command output and the MCP stdio protocol
        logging.basicConfig(level=level, stream=sys.stderr,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Usage errors exit 2 (argparse); everything that goes wrong running a command exits 1.
    try:
        return _run(args)
    except (RelockError, ImportError, OSError, ValueError, KeyError) as e:
        log.debug("relock %s failed", args.cmd, exc_info=True)
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
