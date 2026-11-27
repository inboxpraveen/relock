"""RelapseBench behavioural study across hosted models (via OpenRouter).

The question: after an agent has been corrected once, how often does it go back to the old rule
(or follow a poisoned one) when the memory layer is the usual "append the lesson" kind, compared
with relock?

    python scripts/model_study.py pilot  --models qwen/qwen3.8-flash ...
    python scripts/model_study.py run    --models ... --budget 4.5
    python scripts/model_study.py report

The API key is read from $OPENROUTER_API_KEY or from API_KEY / OPENROUTER_API_KEY in ./.env. It is
never printed, logged or cached. Responses are cached in bench-results/models/cache.sqlite and
finished units are kept in bench-results/models/units/, so an interrupted run resumes without
paying twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from relock_memory._util import approx_tokens
from relock_memory.bench.agents import SYSTEM, LLMAgent, build_prompt
from relock_memory.bench.harness import LOADS, Config, run_behavioral
from relock_memory.bench.scenarios import FAMILIES, padding
from relock_memory.rewriters import LLMRewriter

API = "https://openrouter.ai/api/v1"
ROOT = Path("bench-results/models")
UNITS = ROOT / "units"
DEFAULT_MODELS = [
    "qwen/qwen3.8-27b", "qwen/qwen3.8-flash", "deepseek/deepseek-v4.1-flash", "moonshotai/kimi-k3",
    "anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "thinkingmachines/inkling-2-mini",
    "radixark/daisyrose-27b",
]
DEFAULT_SYSTEMS = ["append-notes", "append-clause", "append-gated", "relock"]
LOAD_BY_NAME = {ld.name: ld for ld in LOADS}


# ------------------------------------------------------------------------------------ plumbing
class Fatal(BaseException):
    """Stops the run. A BaseException, so the agent's and rewriter's ``except Exception`` can't
    swallow it and silently turn "out of budget" into a wrong answer."""


def api_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() in ("OPENROUTER_API_KEY", "API_KEY"):
                return value.strip().strip("'\"")
    raise SystemExit("no API key: set OPENROUTER_API_KEY or put API_KEY=... in .env")


def key_usage(key: str) -> float:
    req = urllib.request.Request(f"{API}/key", headers={"Authorization": f"Bearer {key}"})
    return float(json.load(urllib.request.urlopen(req, timeout=30))["data"]["usage"] or 0.0)


class Cache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS r (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self.lock = threading.Lock()

    def get(self, k: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT v FROM r WHERE k = ?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v: dict) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO r VALUES (?, ?)", (k, json.dumps(v)))
            self.db.commit()


class Meter:
    """Spend actually incurred in this process (cache hits cost nothing)."""

    def __init__(self, budget: float) -> None:
        self.budget = budget
        self.spent = 0.0
        self.calls = 0
        self.lock = threading.Lock()
        self.stopped: str | None = None

    def add(self, cost: float) -> None:
        with self.lock:
            self.spent += cost
            self.calls += 1
            if self.spent > self.budget and not self.stopped:
                self.stopped = f"budget ${self.budget:.2f} reached"

    def check(self) -> None:
        if self.stopped:
            raise Fatal(self.stopped)


_tl = threading.local()


def _log() -> list[dict]:
    if not hasattr(_tl, "calls"):
        _tl.calls = []
    return _tl.calls


class Client:
    """``Callable[[str], str]`` over OpenRouter chat completions, with usage accounting."""

    def __init__(self, key: str, model: str, *, tag: str, cache: Cache, meter: Meter,
                 reasoning: dict | None, max_tokens: int, cheapest: bool = True) -> None:
        self._key = key
        self.model = model
        self.tag = tag
        self.cache = cache
        self.meter = meter
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.cheapest = cheapest

    def body(self, prompt: str) -> dict:
        b: dict[str, Any] = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                             "max_tokens": self.max_tokens, "temperature": 0, "usage": {"include": True}}
        if self.reasoning is not None:
            b["reasoning"] = self.reasoning
        if self.cheapest:
            b["provider"] = {"sort": "price"}  # always route to the model's cheapest provider
        return b

    def __call__(self, prompt: str) -> str:
        self.meter.check()
        body = self.body(prompt)
        # Routing preferences are not part of the cache key: the prompt and parameters are.
        keyed = {k: v for k, v in body.items() if k != "provider"}
        k = hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()
        hit = self.cache.get(k)
        entry: dict[str, Any] = {"tag": self.tag, "cached": hit is not None}
        if hit is None:
            hit = self._post(body)
            if "error" not in hit:
                self.cache.put(k, hit)
            self.meter.add(hit.get("cost", 0.0))
        entry.update(hit)
        _log().append(entry)
        if "error" in hit:
            raise RuntimeError(f"api error: {hit['error']}")
        return hit["content"]

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        delay = 2.0
        for attempt in range(6):
            req = urllib.request.Request(f"{API}/chat/completions", data=data, method="POST", headers={
                "Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                "X-Title": "RelapseBench"})
            t = time.time()
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    d = json.load(r)
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code in (401, 402, 403):
                    self.meter.stopped = f"HTTP {e.code} from OpenRouter: {detail}"
                    raise Fatal(self.meter.stopped) from None
                if e.code in (408, 429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(delay + random.random())
                    delay *= 2
                    continue
                return {"error": f"HTTP {e.code}: {detail}"}
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                if attempt < 5:
                    time.sleep(delay + random.random())
                    delay *= 2
                    continue
                return {"error": f"{type(e).__name__}: {e}"}
            if "error" in d and not d.get("choices"):
                err = d["error"]
                code = err.get("code") if isinstance(err, dict) else None
                if code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(delay + random.random())
                    delay *= 2
                    continue
                return {"error": json.dumps(err)[:300]}
            u = d.get("usage") or {}
            msg = (d.get("choices") or [{}])[0].get("message") or {}
            details = u.get("completion_tokens_details") or {}
            return {"content": msg.get("content") or "", "provider": d.get("provider"),
                    "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                    "reasoning_tokens": details.get("reasoning_tokens", 0) or 0,
                    "cost": float(u.get("cost") or 0.0), "latency": round(time.time() - t, 3),
                    "finish": (d.get("choices") or [{}])[0].get("finish_reason")}
        return {"error": "retries exhausted"}


class RecordingAgent:
    """Wraps :class:`LLMAgent` and keeps, per call, the value, parse status and usage."""

    def __init__(self, llm: Client) -> None:
        self.inner = LLMAgent(llm, name=llm.model)
        self.name = self.inner.name
        self.log: list[dict] = []

    def act(self, family, task, memory, padding_text, variants):
        before_fail, before_calls = self.inner.failures, len(_log())
        value = self.inner.act(family, task, memory, padding_text, variants)
        calls = _log()[before_calls:]
        call = calls[-1] if calls else {}
        self.log.append({
            "family": family.key, "rule": family.classify(value, variants),
            "value": value, "parsed": self.inner.failures == before_fail, "api_error": "error" in call,
            "mem_tokens": approx_tokens(memory), "prompt_tokens": call.get("prompt_tokens", 0),
            "completion_tokens": call.get("completion_tokens", 0), "reasoning_tokens": call.get("reasoning_tokens", 0),
            "cost": call.get("cost", 0.0), "provider": call.get("provider")})
        return value


class CountingRewriter:
    def __init__(self, inner: LLMRewriter) -> None:
        self.inner = inner
        self.name = inner.name
        self.proposals = 0
        self.fallbacks = 0

    def propose(self, request):
        p = self.inner.propose(request)
        if p is not None:
            self.proposals += 1
            if "+" in (p.author or ""):
                self.fallbacks += 1
        return p


# ------------------------------------------------------------------------------------ pilot
def cmd_pilot(args: argparse.Namespace) -> int:
    key = api_key()
    cache, meter = Cache(ROOT / "cache.sqlite"), Meter(args.budget)
    fam = FAMILIES[1]
    rng = random.Random(7)
    task, variants = fam.make(rng)
    mem = f"## Procedures from memory\n### {fam.title}\n- " + "\n- ".join([*fam.stable, fam.new_rule])
    prompts = {"L0": build_prompt(fam, task, mem, ""),
               "L3": build_prompt(fam, task, mem, padding(8000, fam, rng, stale=True, poison=True, example=variants))}
    start = key_usage(key)
    variants_r = [("off", {"enabled": False}), ("default", None)]
    for model in args.models:
        for rname, reasoning in variants_r:
            for lname, prompt in prompts.items():
                c = Client(key, model, tag="pilot", cache=cache, meter=meter, reasoning=reasoning, max_tokens=512)
                try:
                    out = c(f"{SYSTEM}\n\n{prompt}")
                except RuntimeError:
                    out = None
                e = _log()[-1]
                if "error" in e:
                    print(f"{model:34} reasoning={rname:7} {lname}: ERROR {e['error'][:160]}")
                    continue
                print(f"{model:34} reasoning={rname:7} {lname}: in={e['prompt_tokens']:5} out={e['completion_tokens']:4} "
                      f"reason={e['reasoning_tokens']:4} ${e['cost']:.5f} {e['latency']:5.1f}s "
                      f"{e['provider']!s:12} finish={e['finish']} -> {(out or '')[:70]!r}")
            if rname == "off":
                ok = [x for x in _log()[-2:] if "error" not in x and x["reasoning_tokens"] == 0]
                if len(ok) == 2:
                    break  # reasoning switched off cleanly; no need to try the default
    time.sleep(8)
    print(f"\npilot spend (metered): ${meter.spent:.4f}; key usage delta: ${key_usage(key) - start:.4f}")
    return 0


# ------------------------------------------------------------------------------------ run
def unit_path(model: str, seed: int, load: str, system: str) -> Path:
    return UNITS / f"{model.replace('/', '__')}__s{seed}__{load}__{system}.json"


def run_unit(key: str, model: str, seed: int, load: str, system: str, args, cache: Cache, meter: Meter,
             reasoning: dict | None) -> dict:
    _tl.calls = []
    cheapest = args.providers == "cheapest"
    agent_llm = Client(key, model, tag="agent", cache=cache, meter=meter, reasoning=reasoning,
                       max_tokens=args.max_tokens, cheapest=cheapest)
    rewrite_llm = Client(key, model, tag="rewrite", cache=cache, meter=meter, reasoning=reasoning,
                         max_tokens=1024, cheapest=cheapest)
    agent = RecordingAgent(agent_llm)
    rewriter = CountingRewriter(LLMRewriter(rewrite_llm))
    cfg = Config(systems=[system], loads=[LOAD_BY_NAME[load]], seeds=[seed], probes=args.probes,
                 max_learn=args.max_learn)
    recs = run_behavioral(cfg, agent, rewriter=rewriter)
    probe_log = agent.log[len(agent.log) - len(recs):]
    calls = _log()
    probes = []
    for r, c in zip(recs, probe_log, strict=True):
        assert c["family"] == r.family and c["rule"] == r.outcome, "probe log out of step with records"
        probes.append({"probe": r.probe, "outcome": r.outcome, "learned": r.learned,
                       "learn_errors": r.learn_errors, "seen": r.seen, **c})
    learn_log = agent.log[: len(agent.log) - len(recs)]
    out = {
        "model": model, "seed": seed, "load": load, "system": system,
        "probes": probes,
        "learn": [{"family": c["family"], "rule": c["rule"], "value": c["value"]} for c in learn_log],
        "learn_calls": len(learn_log),
        "learn_cost": sum(c["cost"] for c in learn_log),
        "rewrite_calls": sum(1 for c in calls if c["tag"] == "rewrite"),
        "rewrite_cost": sum(c.get("cost", 0.0) for c in calls if c["tag"] == "rewrite"),
        "rewrite_proposals": rewriter.proposals, "rewrite_fallbacks": rewriter.fallbacks,
        "api_errors": sum(1 for c in calls if "error" in c),
        "cached_calls": sum(1 for c in calls if c.get("cached")),
        "providers": sorted({str(c.get("provider")) for c in calls if c.get("provider")}),
    }
    return out


def cmd_run(args: argparse.Namespace) -> int:
    key = api_key()
    UNITS.mkdir(parents=True, exist_ok=True)
    cache, meter = Cache(ROOT / "cache.sqlite"), Meter(args.budget)
    reasoning = None if args.reasoning == "default" else {"enabled": False}
    todo = [(m, s, ld, sy) for s in args.seeds for ld in args.loads for sy in args.systems for m in args.models
            if not unit_path(m, s, ld, sy).exists()]
    total = len(args.models) * len(args.seeds) * len(args.loads) * len(args.systems)
    print(f"{total - len(todo)}/{total} units already done; running {len(todo)} with {args.workers} workers; "
          f"budget ${args.budget:.2f}", flush=True)
    start_usage = key_usage(key)
    done = 0
    stop = threading.Event()

    def job(m, s, ld, sy):
        if stop.is_set():
            return None
        try:
            res = run_unit(key, m, s, ld, sy, args, cache, meter, reasoning)
        except Fatal as e:
            stop.set()
            return ("fatal", str(e))
        unit_path(m, s, ld, sy).write_text(json.dumps(res), encoding="utf-8")
        return res

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(job, *u): u for u in todo}
        for f in as_completed(futs):
            res = f.result()
            if res is None:
                continue
            if isinstance(res, tuple):
                print(f"STOPPED: {res[1]}", flush=True)
                continue
            done += 1
            correct = sum(p["outcome"] == "new" for p in res["probes"])
            print(f"[{done}/{len(todo)}] ${meter.spent:.3f} {res['model']:32} s{res['seed']} {res['load']} "
                  f"{res['system']:14} correct {correct}/{len(res['probes'])} errors={res['api_errors']}", flush=True)
    time.sleep(5)
    print(f"metered spend ${meter.spent:.3f}; key usage delta ${key_usage(key) - start_usage:.3f}", flush=True)
    return 1 if meter.stopped else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="model_study")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("pilot", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
        sp.add_argument("--budget", type=float, default=0.25 if name == "pilot" else 4.5)
    sp = sub.choices["run"]
    sp.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS)
    sp.add_argument("--loads", nargs="+", default=["L0", "L1", "L3"], choices=sorted(LOAD_BY_NAME))
    sp.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    sp.add_argument("--probes", type=int, default=3)
    sp.add_argument("--max-learn", type=int, default=5)
    sp.add_argument("--max-tokens", type=int, default=512)
    sp.add_argument("--workers", type=int, default=24)
    sp.add_argument("--reasoning", choices=["off", "default"], default="off")
    sp.add_argument("--providers", choices=["cheapest", "default"], default="cheapest",
                    help="cheapest: route every call to the model's lowest-priced provider; "
                         "default: OpenRouter's load balancing")
    args = p.parse_args(argv)
    return {"pilot": cmd_pilot, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
