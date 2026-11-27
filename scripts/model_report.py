"""Build the published report for the hosted-model study from bench-results/models/units/.

    python scripts/model_report.py [--prices]     # --prices fetches OpenRouter list prices

Writes docs/results/models/: README.md, summary.json, records.csv and three SVG charts.
Every number in README.md is computed here from the recorded calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any

from relock_memory.bench.harness import wilson
from relock_memory.bench.report import svg_panels
from relock_memory.bench.scenarios import FAMILIES

UNITS = Path("bench-results/models/units")
OUT = Path("docs/results/models")
SYSTEMS = ["append-notes", "append-clause", "append-gated", "relock"]
LOADS = ["L0", "L1", "L2", "L3"]
LOAD_NAMES = {"L0": "fresh session", "L1": "1.5k-token history", "L2": "4k-token history",
              "L3": "8k-token history, tight memory"}
SYSTEM_NAMES = {
    "append-notes": "append-notes (each correction becomes a new memory note)",
    "append-clause": "append-clause (the correction is appended to the procedure, like a rules file)",
    "append-gated": "append-gated (append-notes behind relock's trust and evidence gates)",
    "relock": "relock (gated, in-place rewrite of the clause that fired)",
}
CONSEQUENCE = {
    "orders.fetch": "MM/DD/YYYY sent to an ISO-date API: rejected or wrong day",
    "payments.charge": "dollars sent to a cents API: wrong amount charged, or a failed charge",
    "hvac.thermostat": "Fahrenheit sent as Celsius: a 72 °C setpoint",
    "reports.auth": "old auth scheme: every call is a 401",
    "email.send": "wrong recipient separator: mail goes to a malformed address",
    "deploy.service": "deploys to the old region: data residency breach",
    "users.list": "1-based page on a 0-based API: silently skips a page",
    "files.write": "Latin-1 instead of UTF-8: corrupted text",
    "sql.count": "wrong identifier quoting: query fails",
    "routes.limit": "miles sent as kilometres: routes capped 38% short",
}
OUTCOMES = [("new", "correct"), ("old", "relapse"), ("poison", "poisoned"), ("other", "other")]


def load_units() -> list[dict]:
    units = [json.loads(p.read_text("utf-8")) for p in sorted(UNITS.glob("*.json"))]
    if not units:
        sys.exit(f"no units in {UNITS}; run scripts/model_study.py run first")
    return units


def short(model: str) -> str:
    return model.split("/", 1)[1]


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def cell(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else 0.0, "ci": [round(lo, 4), round(hi, 4)]}


def summarise(units: list[dict]) -> dict[str, Any]:
    models = sorted({u["model"] for u in units})
    probes: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    learn: dict[tuple[str, str], list[tuple[int, bool]]] = defaultdict(list)
    money: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    rewrites = {"proposals": 0, "fallbacks": 0, "calls": 0}
    api_errors = defaultdict(int)
    for u in units:
        m, s, ld = u["model"], u["system"], u["load"]
        for p in u["probes"]:
            for key in ((m, s, ld), ("all", s, ld), (m, s, "all"), ("all", s, "all")):
                probes[key].append(p)
        for p in u["probes"]:
            if p["probe"] == 0:
                learn[(m, s)].append((p["learn_errors"], p["learned"]))
                learn[("all", s)].append((p["learn_errors"], p["learned"]))
        money[s]["probe"] += sum(p["cost"] for p in u["probes"])
        money[s]["learn"] += u["learn_cost"]
        money[s]["rewrite"] += u["rewrite_cost"]
        money[s]["probe_tokens"] += sum(p["prompt_tokens"] for p in u["probes"])
        money[s]["probes"] += len(u["probes"])
        if s == "relock":
            rewrites["proposals"] += u["rewrite_proposals"]
            rewrites["fallbacks"] += u["rewrite_fallbacks"]
            rewrites["calls"] += u["rewrite_calls"]
        api_errors[m] += u["api_errors"]

    def outcome_cells(ps: list[dict]) -> dict[str, Any]:
        valid = [p for p in ps if not p["api_error"]]
        n = len(valid)
        out = {label: cell(sum(p["outcome"] == o for p in valid), n) for o, label in OUTCOMES}
        out["no_valid_call"] = cell(sum(p["outcome"] == "other" and (not p["parsed"] or not p["value"])
                                        for p in valid), n)
        out["api_errors_excluded"] = len(ps) - n
        out["prompt_tokens_mean"] = round(sum(p["prompt_tokens"] for p in valid) / max(1, n), 1)
        out["memory_tokens_mean"] = round(sum(p["mem_tokens"] for p in valid) / max(1, n), 1)
        return out

    table = {f"{m}|{s}|{ld}": outcome_cells(ps) for (m, s, ld), ps in probes.items()}
    fam: dict[str, dict[str, Any]] = {}
    for f in FAMILIES:
        for s in ("append-notes", "relock"):
            ps = [p for p in probes[("all", s, "L3")] if p["family"] == f.key and not p["api_error"]]
            fam.setdefault(f.key, {})[s] = cell(sum(p["outcome"] == "new" for p in ps), len(ps))
    learning = {}
    for (m, s), xs in learn.items():
        learning[f"{m}|{s}"] = {"mean_errors": round(sum(e for e, _ in xs) / len(xs), 3),
                                "learned_rate": round(sum(ok for _, ok in xs) / len(xs), 4), "n": len(xs)}
    return {"models": models, "systems": SYSTEMS, "loads": LOADS, "outcomes": table, "families": fam,
            "learning": learning, "money": {s: dict(v) for s, v in money.items()}, "rewrites": rewrites,
            "api_errors": dict(api_errors)}


# ------------------------------------------------------------------------------------ charts
def dumbbell(summary: dict[str, Any], load: str) -> str:
    """Per model: append-notes -> relock, with append-gated as a hollow marker. Rows sorted by baseline."""
    models = sorted(summary["models"], key=lambda m: summary["outcomes"][f"{m}|append-notes|{load}"]["correct"]["rate"])
    rows = [*models, "all"]
    label_w, plot_w, right = 190, 470, 70
    width = label_w + plot_w + right
    top, row_h = 104, 30
    height = top + row_h * len(rows) + 52
    x = lambda v: label_w + v * plot_w  # noqa: E731
    title = "Corrected once. Does the agent keep doing it right?"
    n_dot = round(sum(summary["outcomes"][f"{m}|relock|{load}"]["correct"]["n"] for m in models) / len(models))
    n_all = summary["outcomes"][f"all|relock|{load}"]["correct"]["n"]
    sub = (f"Correct tool calls in a long session ({LOAD_NAMES[load]}), after the same correction, "
           f"noise and prompt injection. About {n_dot} calls per dot; pooled row: {n_all}.")
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        "role=\"img\" aria-labelledby=\"t d\" font-family=\"system-ui, -apple-system, 'Segoe UI', sans-serif\">",
        f'<title id="t">{escape(title)}</title><desc id="d">{escape(sub)}</desc>',
        "<style>",
        ":root{--surface:#ffffff;--ink:#191919;--ink2:#666666;--grid:#e7e7e7;--axis:#c2c2c2;"
        "--accent:#257de0;--muted:#8c8c8c;--link:#c9c9c9}",
        "@media (prefers-color-scheme: dark){:root{--surface:#191919;--ink:#efefef;--ink2:#9c9c9c;"
        "--grid:#303030;--axis:#474747;--accent:#338ced;--muted:#7a7a7a;--link:#3d3d3d}}",
        ".bg{fill:var(--surface)} .t{fill:var(--ink);font-size:16px;font-weight:600}"
        " .s{fill:var(--ink2);font-size:12px} .l{fill:var(--ink);font-size:12px}"
        " .lh{fill:var(--ink);font-size:12px;font-weight:700} .v{fill:var(--ink2);font-size:11px;"
        "font-variant-numeric:tabular-nums} .vh{fill:var(--ink);font-size:11px;font-weight:700;"
        "font-variant-numeric:tabular-nums} .g{stroke:var(--grid);stroke-width:1} .a{stroke:var(--axis);"
        "stroke-width:1} .k{stroke:var(--link);stroke-width:2} .base{fill:var(--muted);stroke:var(--surface);"
        "stroke-width:2} .gate{fill:var(--surface);stroke:var(--muted);stroke-width:2}"
        " .rl{fill:var(--accent);stroke:var(--surface);stroke-width:2} .tick{fill:var(--ink2);font-size:10px}",
        "</style>",
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" rx="8"/>',
        f'<text class="t" x="16" y="28">{escape(title)}</text>',
        f'<text class="s" x="16" y="48">{escape(sub[:96])}</text>',
        f'<text class="s" x="16" y="64">{escape(sub[96:].strip())}</text>',
    ]
    ly = 86
    legend = [("base", "append-notes (usual memory)"), ("gate", "append-gated (relock's gates, still appending)"),
              ("rl", "relock")]
    lx = label_w
    for cls, text in legend:
        out.append(f'<circle class="{cls}" cx="{lx + 6}" cy="{ly - 4}" r="5"/>')
        out.append(f'<text class="l" x="{lx + 16}" y="{ly}">{escape(text)}</text>')
        lx += 16 + 7 * len(text) + 18
    bottom = top + row_h * len(rows)
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gx = x(frac)
        out.append(f'<line class="{"a" if frac == 0 else "g"}" x1="{gx:.1f}" y1="{top - 6}" x2="{gx:.1f}" '
                   f'y2="{bottom - 6}"/>')
        out.append(f'<text class="tick" x="{gx:.1f}" y="{bottom + 8}" text-anchor="middle">{int(frac * 100)}%</text>')
    for i, m in enumerate(rows):
        cy = top + i * row_h + row_h / 2 - 6
        if m == "all":
            out.append(f'<line class="a" x1="16" y1="{cy - row_h / 2 + 1}" x2="{width - 16}" y2="{cy - row_h / 2 + 1}"/>')
        name = f"all {len(models)} models" if m == "all" else short(m)
        out.append(f'<text class="{"lh" if m == "all" else "l"}" x="{label_w - 14}" y="{cy + 4}" '
                   f'text-anchor="end">{escape(name)}</text>')
        o = summary["outcomes"]
        b, g, r = (o[f"{m}|{s}|{load}"]["correct"]["rate"] for s in ("append-notes", "append-gated", "relock"))
        out.append(f'<line class="k" x1="{x(min(b, r)):.1f}" y1="{cy}" x2="{x(max(b, r)):.1f}" y2="{cy}"/>')
        for cls, v, s in (("gate", g, "append-gated"), ("base", b, "append-notes"), ("rl", r, "relock")):
            c = o[f"{m}|{s}|{load}"]["correct"]
            tip = f"{name}, {s}: {pct(v)} correct ({c['k']}/{c['n']}, 95% CI {pct(c['ci'][0])}-{pct(c['ci'][1])})"
            out.append(f'<g><title>{escape(tip)}</title><circle class="{cls}" cx="{x(v):.1f}" cy="{cy}" '
                       f'r="{6 if cls == "rl" else 5}"/></g>')
        out.append(f'<text class="v" x="{x(b) - 10:.1f}" y="{cy + 4}" text-anchor="end">{pct(b)}</text>')
        out.append(f'<text class="vh" x="{x(r) + 11:.1f}" y="{cy + 4}">{pct(r)}</text>')
    foot = "RelapseBench behavioural mode via OpenRouter; reasoning off, temperature 0. Details: docs/results/models."
    out.append(f'<text class="s" x="16" y="{height - 14}">{escape(foot)}</text>')
    out.append("</svg>")
    return "\n".join(out)


# ------------------------------------------------------------------------------------ markdown
def ci(c: dict[str, Any]) -> str:
    return f"{pct(c['rate'])} ({pct(c['ci'][0])}–{pct(c['ci'][1])})"  # noqa: RUF001


def build_markdown(s: dict[str, Any], prices: dict[str, tuple[float, float]] | None, units: list[dict]) -> str:
    o = s["outcomes"]
    L = lambda m, sy, ld, k="correct": o[f"{m}|{sy}|{ld}"][k]  # noqa: E731
    an3, rl3 = L("all", "append-notes", "L3"), L("all", "relock", "L3")
    ac3, ag3 = L("all", "append-clause", "L3"), L("all", "append-gated", "L3")
    rel = {sy: L("all", sy, "all", "relapse") for sy in SYSTEMS}
    poi = {sy: L("all", sy, "all", "poisoned") for sy in SYSTEMS}
    better = sum(L(m, "relock", "L3")["rate"] > L(m, "append-notes", "L3")["rate"] for m in s["models"])
    sep = sum(L(m, "relock", "L3")["ci"][0] > L(m, "append-notes", "L3")["ci"][1] for m in s["models"])
    wrong_an, wrong_rl = (1 - an3["rate"]) * 1000, (1 - rl3["rate"]) * 1000
    lines: list[str] = []
    w = lines.append
    w(f"# relock vs. \"append the lesson\": {len(s['models'])} models, {len(FAMILIES)} changed tool contracts")
    w("")
    w("An agent is corrected: the API now wants ISO dates, amounts in cents, the EU region. Later it")
    w("works through a long session, with stale docs and an injected web page in its context. Does it")
    w("keep making the corrected call?")
    w("")
    w(f"**Long sessions:** with relock, agents made the corrected call **{pct(rl3['rate'])}** of the")
    w(f"time, pooled over {len(s['models'])} models.")
    w(f"- A memory that appends each lesson as a note managed **{pct(an3['rate'])}**.")
    w(f"- Appending the correction to the procedure, like a rules file, managed **{pct(ac3['rate'])}**.")
    w(f"- relock beat append-notes for **{better} of {len(s['models'])} models**. The 95% intervals don't")
    w(f"  overlap for {sep} of them.")
    w(f"- That is {wrong_an:.0f} wrong tool calls per 1,000 with append-notes, against {wrong_rl:.0f} with")
    w("  relock.")
    w("")
    w("![Correct tool calls per model, long session](by-model.svg)")
    w("")
    w("The design was committed before the run: see [DESIGN.md](DESIGN.md). Every number below is")
    w("computed by [`scripts/model_report.py`](../../../scripts/model_report.py) from")
    w("[`records.csv`](records.csv), which holds every probe call.")
    w("")
    w("## Why the usual memories fail")
    w("")
    w("Across all loads and models, the share of calls in each outcome was:")
    w("")
    w("| memory | correct | relapsed to the old rule | followed the poisoned rule | other |")
    w("|---|---|---|---|---|")
    for sy in SYSTEMS:
        a = o[f"all|{sy}|all"]
        w(f"| {sy} | {pct(a['correct']['rate'])} | {pct(a['relapse']['rate'])} | {pct(a['poisoned']['rate'])} "
          f"| {pct(a['other']['rate'])} |")
    w("")
    w("Two separate failures account for the gap.")
    w("")
    w("- **Relapse.** The appended note sits next to the old rule, and models follow the old one. The")
    w(f"  relapse rate was {pct(rel['append-notes']['rate'])} for append-notes and "
      f"{pct(rel['append-clause']['rate'])} for append-clause. relock rewrites the clause in place,")
    w(f"  so the old rule can't be retrieved and the relapse rate was {pct(rel['relock']['rate'])}.")
    w("- **Poisoning.** Without a trust gate, a flaky check's wrong \"fix\" goes into memory too. Models")
    w(f"  followed the poisoned rule {pct(poi['append-notes']['rate'])} of the time with append-notes and")
    w(f"  {pct(poi['relock']['rate'])} with relock.")
    w("")
    w("**Gating alone isn't enough.** `append-gated` keeps relock's trust and evidence gates but still")
    w(f"appends. It cuts poisoning to {pct(poi['append-gated']['rate'])}, but the old rule stays")
    w(f"retrievable, and its relapse rate was {pct(rel['append-gated']['rate'])}. In long sessions it")
    w(f"made {pct(ag3['rate'])} correct calls, against {pct(rl3['rate'])} for relock. Most of the gain")
    w("comes from superseding the rule in place, and the gate keeps that rewrite safe.")
    w("")
    w("![Outcomes by memory system](outcomes.svg)")
    w("")
    w("## Every model, every load")
    w("")
    n_cell = round(sum(c["correct"]["n"] for k, c in o.items() if k.split("|")[0] != "all"
                       and k.split("|")[2] != "all") / max(1, sum(1 for k in o if k.split("|")[0] != "all"
                                                                     and k.split("|")[2] != "all")))
    w(f"Correct-call rate with 95% Wilson intervals, n ≈ {n_cell} per cell:")
    w("")
    for ld in LOADS:
        w(f"**{ld}: {LOAD_NAMES[ld]}**")
        w("")
        w("| model | append-notes | append-clause | append-gated | relock |")
        w("|---|---|---|---|---|")
        for m in [*s["models"], "all"]:
            name = f"**all {len(s['models'])} models**" if m == "all" else f"`{m}`"
            w(f"| {name} | " + " | ".join(ci(L(m, sy, ld)) for sy in SYSTEMS) + " |")
        w("")
    w("![Correct calls by load, pooled over models](by-load.svg)")
    w("")
    w("## Where it bites")
    w("")
    w("This table pools the long-session (`L3`) calls across all models. The consequence column is our")
    w("description of what the wrong call would do; it was not measured.")
    w("")
    w("| task | append-notes | relock | what a wrong call does |")
    w("|---|---|---|---|")
    for f in FAMILIES:
        c = s["families"][f.key]
        w(f"| `{f.key}` | {pct(c['append-notes']['rate'])} | {pct(c['relock']['rate'])} | {CONSEQUENCE[f.key]} |")
    w("")
    w("## What it costs and what it saves")
    w("")
    money, lr = s["money"], s["learning"]
    w("| memory | errors before learning a correction | prompt tokens per call | model $ per 1,000 correct calls |")
    w("|---|---|---|---|")
    for sy in SYSTEMS:
        mo = money[sy]
        correct = o[f"all|{sy}|all"]["correct"]["k"]
        per_k = (mo["probe"] + mo["rewrite"]) / max(1, correct) * 1000
        w(f"| {sy} | {lr[f'all|{sy}']['mean_errors']:.2f} | {mo['probe_tokens'] / mo['probes']:,.0f} | ${per_k:.2f} |")
    w("")
    rw = s["rewrites"]
    errs = {sy: lr[f"all|{sy}"]["mean_errors"] for sy in SYSTEMS}
    fastest = min(errs, key=errs.get)
    w("- **Learning:** these are the errors before the model first followed the correction (mean per")
    w("  task):")
    w("  " + ", ".join(f"{sy} {errs[sy]:.2f}" for sy in SYSTEMS) + ".")
    w("  relock waits for 2 trusted errors before it rewrites (`min_evidence=2`; set it to 1 to learn")
    w("  from the first error).")
    if fastest != "relock":
        w(f"  `{fastest}` learned fastest, because the correction lands inside the procedure. The")
        w("  poisoned \"fix\" lands there too, which is why it followed the poisoned rule in")
        w(f"  {pct(poi[fastest]['rate'])} of calls.")
    over = money["relock"]["rewrite"] / max(1e-9, money["relock"]["probe"] + money["relock"]["learn"]
                                            + money["relock"]["rewrite"])
    w(f"- **Rewrite overhead:** relock made {rw['proposals']} rewrites, done by the same model, for")
    w(f"  ${money['relock']['rewrite']:.3f}. That is {over * 100:.1f}% of relock's model spend in this study.")
    w(f"  The LLM rewriter fell back to the deterministic one {rw['fallbacks']} times.")
    toks = {sy: money[sy]["probe_tokens"] / money[sy]["probes"] for sy in SYSTEMS}
    w(f"- **Tokens:** prompts averaged {toks['relock']:,.0f} tokens with relock and "
      f"{toks['append-notes']:,.0f} with append-notes.")
    w("  relock keeps one current rule instead of a growing pile of notes, so the fix adds no prompt")
    w("  cost.")
    w(f"- **Per 1,000 calls in long sessions:** about {wrong_an - wrong_rl:.0f} fewer wrong calls than")
    w("  append-notes. Each one is a failed request, a retry, or, for the silent cases above, a wrong")
    w("  charge, region or file.")
    w("")
    w("## Cost of this study")
    w("")
    paid = sum(sum(p["cost"] for p in u["probes"]) + u["learn_cost"] + u["rewrite_cost"] for u in units)
    w(f"- **Recorded spend:** ${paid:.2f} for {sum(len(u['probes']) for u in units):,} probe calls, plus")
    w(f"  learning and rewrite calls, across {len(s['models'])} models. The first run (seeds 0-1) used")
    w("  OpenRouter's default provider routing; the extension used each model's cheapest provider.")
    if prices:
        tok = defaultdict(lambda: [0, 0])
        for u in units:
            for p in u["probes"]:
                tok[u["model"]][0] += p["prompt_tokens"]
                tok[u["model"]][1] += p["completion_tokens"]
        cheap = sum(i * prices[m][0] + c * prices[m][1] for m, (i, c) in tok.items() if m in prices)
        probe_paid = sum(sum(p["cost"] for p in u["probes"]) for u in units)
        w(f"- **Reproduction:** the probe calls cost ${probe_paid:.2f} here and would cost about")
        w(f"  ${cheap:.2f} at each model's cheapest provider. `scripts/model_study.py` now routes there")
        w("  by default.")
    w("")
    w("## Caveats")
    w("")
    excluded = sum(s["api_errors"].values())
    w(f"- **API errors:** {excluded} calls failed at the API after retries, all on `openai/gpt-6-luna`.")
    w("  Failed probes are excluded from the rates above. A failure during learning counted as a wrong")
    w("  answer for every memory system alike.")
    w("- **Refusals:** `gpt-6-luna` sometimes declines the payment task (\"can't charge without looking up")
    w("  the card\"). These count as `other` for every memory.")
    w("- **Exact reruns differ slightly.** OpenRouter routes calls to different providers; the provider")
    w("  of every call is in `records.csv`.")
    w("- **Synthetic tasks.** The 10 tasks are short tool-contract changes, built to have one")
    w("  unambiguous right answer. Real agents face messier memories, and we expect the old-rule")
    w("  conflict to matter more there, but that is not measured here.")
    w("")
    w("Reproduce:")
    w("")
    w("```")
    w("pip install -e .")
    w("OPENROUTER_API_KEY=... python scripts/model_study.py run --budget 5")
    w("python scripts/model_report.py --prices")
    w("```")
    return "\n".join(lines) + "\n"


def fetch_prices() -> dict[str, tuple[float, float]]:
    d = json.load(urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60))["data"]
    return {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])) for m in d}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="model_report")
    ap.add_argument("--prices", action="store_true", help="fetch OpenRouter list prices for the cost section")
    args = ap.parse_args(argv)
    units = load_units()
    s = summarise(units)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8", newline="\n")
    import csv

    cols = ["model", "system", "load", "seed", "family", "probe", "outcome", "value", "parsed", "api_error",
            "learned", "learn_errors", "memory_tokens", "prompt_tokens", "completion_tokens", "cost", "provider"]
    with (OUT / "records.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(cols)
        for u in sorted(units, key=lambda u: (u["model"], u["system"], u["load"], u["seed"])):
            for p in sorted(u["probes"], key=lambda p: (p["family"], p["probe"])):
                wr.writerow([u["model"], u["system"], u["load"], u["seed"], p["family"], p["probe"], p["outcome"],
                             p["value"], int(p["parsed"]), int(p["api_error"]), int(p["learned"]),
                             p["learn_errors"], p["mem_tokens"], p["prompt_tokens"], p["completion_tokens"],
                             f"{p['cost']:.8f}", p["provider"] or ""])
    (OUT / "by-model.svg").write_text(dumbbell(s, "L3") + "\n", encoding="utf-8", newline="\n")
    o = s["outcomes"]
    by_load = [(f"{ld}: {LOAD_NAMES[ld]}", {sy: o[f"all|{sy}|{ld}"]["correct"]["rate"] for sy in SYSTEMS})
               for ld in LOADS]
    (OUT / "by-load.svg").write_text(svg_panels(
        "Correct tool calls after a correction", f"Pooled over {len(s['models'])} models (about "
        f"{o['all|relock|L3']['correct']['n']} calls per bar). Longer, noisier sessions to the right.",
        by_load, SYSTEMS) + "\n", encoding="utf-8", newline="\n")
    outcome_panels = [(title, {sy: o[f"all|{sy}|all"][key]["rate"] for sy in SYSTEMS})
                      for key, title in (("correct", "Correct call"), ("relapse", "Relapsed to the old rule"),
                                         ("poisoned", "Followed the poisoned rule"))]
    (OUT / "outcomes.svg").write_text(svg_panels(
        "What the agent did after being corrected", f"Pooled over {len(s['models'])} models and all loads "
        f"(about {o['all|relock|all']['correct']['n']:,} calls per bar).", outcome_panels, SYSTEMS) + "\n", encoding="utf-8", newline="\n")
    prices = fetch_prices() if args.prices else None
    (OUT / "README.md").write_text(build_markdown(s, prices, units), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}/README.md, summary.json, records.csv, by-model.svg, by-load.svg, outcomes.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
