"""``relock bench`` - run RelapseBench.

    relock bench                                   # structural mode, no model needed
    relock bench --mode behavioral --base-url http://localhost:8000/v1 \\
                 --model Qwen/Qwen3-30B-A3B-Instruct-2507
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from .harness import LOADS, SYSTEM_NAMES, Config, run_behavioral, run_structural
from .report import write_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="relock bench", description="RelapseBench: relapse to corrected behaviour "
                                                                 "under load, poisoning and noise.")
    p.add_argument("--mode", choices=["structural", "behavioral", "both"], default="structural")
    p.add_argument("--systems", nargs="+", default=list(SYSTEM_NAMES), choices=list(SYSTEM_NAMES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--loads", nargs="+", default=[ld.name for ld in LOADS], choices=[ld.name for ld in LOADS])
    p.add_argument("--probes", type=int, default=3, help="probes per family and load (behavioral)")
    p.add_argument("--max-learn", type=int, default=5)
    p.add_argument("--out", default="bench-results")
    p.add_argument("--base-url", default=os.environ.get("RELOCK_LLM_BASE_URL"),
                   help="OpenAI-compatible endpoint (vLLM, SGLang, Ollama, llama.cpp, LM Studio...)")
    p.add_argument("--model", default=os.environ.get("RELOCK_LLM_MODEL"))
    p.add_argument("--api-key", default=os.environ.get("RELOCK_LLM_API_KEY"))
    p.add_argument("--rewriter", choices=["llm", "hint"], default="llm",
                   help="behavioral mode: let the same model do relock's rewrites (llm) or use hints")
    p.add_argument("--agent", choices=["llm", "scripted"], default="llm",
                   help="'scripted' is a deterministic stand-in for testing the pipeline only")
    args = p.parse_args(argv)

    cfg = Config(systems=args.systems, loads=[ld for ld in LOADS if ld.name in args.loads], seeds=args.seeds,
                 probes=args.probes, max_learn=args.max_learn)

    def progress(msg: str) -> None:
        print(f"  {msg}", file=sys.stderr, flush=True)

    if args.mode in ("structural", "both"):
        t = time.time()
        recs = run_structural(cfg)
        paths = write_report(args.out, recs, cfg, "structural")
        print(f"structural: {len(recs)} probes in {time.time() - t:.1f}s -> {paths['md']}, {paths['svg']}")

    if args.mode in ("behavioral", "both"):
        rewriter = None
        if args.agent == "scripted":
            from .agents import ScriptedAgent

            agent = ScriptedAgent()
            print("WARNING: scripted agent - pipeline test only, not a result.", file=sys.stderr)
        else:
            if not args.base_url or not args.model:
                p.error("behavioral mode needs --base-url and --model (or RELOCK_LLM_BASE_URL / RELOCK_LLM_MODEL)")
            from ..llm import OpenAICompatible
            from .agents import LLMAgent

            llm = OpenAICompatible(args.base_url, args.model, api_key=args.api_key, max_tokens=512)
            agent = LLMAgent(llm)
            if args.rewriter == "llm":
                from ..rewriters import LLMRewriter

                rewriter = LLMRewriter(llm)
        t = time.time()
        recs = run_behavioral(cfg, agent, rewriter=rewriter, progress=progress)
        paths = write_report(args.out, recs, cfg, "behavioral", agent.name,
                             pipeline_test=args.agent == "scripted")
        print(f"behavioral: {len(recs)} probes in {time.time() - t:.1f}s -> {paths['md']}, {paths['svg']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
