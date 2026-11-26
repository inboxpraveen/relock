"""RelapseBench reports: JSON (raw + aggregates), Markdown tables, and an SVG chart.

The chart is an *emphasis* small-multiple: one panel per metric (or per load
level), one horizontal bar per memory system, relock in the accent hue and the
baselines in a de-emphasis gray. Every bar carries its system name and value,
so colour never carries meaning alone; the Markdown tables are the table view.
The SVG adapts to light/dark through ``prefers-color-scheme``.
"""

from __future__ import annotations

import json
import os
import platform
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
from typing import Any

from .. import __version__
from .harness import Config, Record, aggregate

STRUCTURAL_LABELS = {
    "stale_exposure": "Superseded rule in context",
    "poison_exposure": "Poisoned rule in context",
    "conflict": "Conflicting rules in context",
    "correction_present": "Correction in context",
    "clean": "Only the correction in context",
}
BEHAVIORAL_LABELS = {
    "relapse": "Relapse (followed the old rule)",
    "poisoned": "Followed the poisoned rule",
    "correct": "Followed the correction",
    "other": "Other / invalid call",
}


def _pooled(records: Sequence[Record], metric_fn: Any) -> dict[str, float]:
    by: dict[str, list[bool]] = {}
    for r in records:
        by.setdefault(r.system, []).append(bool(metric_fn(r)))
    return {s: sum(v) / len(v) for s, v in by.items()}


def _wrap(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def svg_panels(
    title: str,
    subtitle: str,
    panels: Sequence[tuple[str, dict[str, float]]],
    systems: Sequence[str],
    *,
    highlight: str = "relock",
    footnote: str = "",
) -> str:
    """Small multiples of horizontal 0-100% bars, one row per system, one panel per metric."""
    label_w, panel_w, gap = 132, 168, 22
    bar_h, row_gap = 14, 10
    width = label_w + len(panels) * panel_w + (len(panels) - 1) * gap + 24
    sub_lines = _wrap(subtitle, max_chars=int((width - 32) / 6.4))
    top = 92 + 16 * (len(sub_lines) - 1)
    rows_h = len(systems) * (bar_h + row_gap)
    height = top + rows_h + 40 + (18 if footnote else 0)
    plot_w = panel_w - 40  # leave room for the value label after the bar
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-labelledby="t d" font-family="system-ui, -apple-system, '
        f"'Segoe UI', sans-serif\">",
        f"<title id=\"t\">{escape(title)}</title><desc id=\"d\">{escape(subtitle)}</desc>",
        "<style>",
        ":root{--surface:#ffffff;--ink:#191919;--ink2:#666666;--grid:#e7e7e7;--axis:#c2c2c2;"
        "--accent:#257de0;--muted:#8c8c8c}",
        "@media (prefers-color-scheme: dark){:root{--surface:#191919;--ink:#efefef;--ink2:#9c9c9c;"
        "--grid:#303030;--axis:#474747;--accent:#338ced;--muted:#7a7a7a}}",
        ".bg{fill:var(--surface)} .t{fill:var(--ink);font-size:15px;font-weight:600}"
        " .s{fill:var(--ink2);font-size:12px} .p{fill:var(--ink);font-size:12px;font-weight:600}"
        " .l{fill:var(--ink);font-size:12px} .lh{fill:var(--ink);font-size:12px;font-weight:700}"
        " .v{fill:var(--ink2);font-size:11px;font-variant-numeric:tabular-nums}"
        " .vh{fill:var(--ink);font-size:11px;font-weight:700;font-variant-numeric:tabular-nums}"
        " .g{stroke:var(--grid);stroke-width:1} .a{stroke:var(--axis);stroke-width:1}"
        " .bm{fill:var(--muted)} .bh{fill:var(--accent)} .tick{fill:var(--ink2);font-size:10px}",
        "</style>",
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" rx="8"/>',
        f'<text class="t" x="16" y="28">{escape(title)}</text>',
    ]
    out += [f'<text class="s" x="16" y="{48 + 16 * i}">{escape(line)}</text>' for i, line in enumerate(sub_lines)]
    for i, s in enumerate(systems):
        y = top + i * (bar_h + row_gap) + bar_h - 3
        cls = "lh" if s == highlight else "l"
        out.append(f'<text class="{cls}" x="{label_w - 10}" y="{y}" text-anchor="end">{escape(s)}</text>')
    for p, (ptitle, values) in enumerate(panels):
        x0 = label_w + p * (panel_w + gap)
        out.append(f'<text class="p" x="{x0}" y="{top - 22}">{escape(ptitle)}</text>')
        for frac in (0.0, 0.5, 1.0):
            gx = x0 + frac * plot_w
            out.append(f'<line class="{"a" if frac == 0 else "g"}" x1="{gx:.1f}" y1="{top - 8}" x2="{gx:.1f}" '
                       f'y2="{top + rows_h - row_gap + 4}"/>')
            out.append(f'<text class="tick" x="{gx:.1f}" y="{top + rows_h + 10}" text-anchor="middle">'
                       f"{int(frac * 100)}%</text>")
        for i, s in enumerate(systems):
            if s not in values:
                continue
            v = max(0.0, min(1.0, values[s]))
            y = top + i * (bar_h + row_gap)
            w = v * plot_w
            cls = "bh" if s == highlight else "bm"
            label = f"{v * 100:.0f}%"
            tip = f"{s} - {ptitle}: {label}"
            out.append(f"<g><title>{escape(tip)}</title>")
            if w >= 1:
                # 4px rounded data end, square at the baseline
                r = min(4.0, w / 2, bar_h / 2)
                out.append(f'<path class="{cls}" d="M{x0},{y} h{w - r:.2f} a{r},{r} 0 0 1 {r},{r} '
                           f'v{bar_h - 2 * r} a{r},{r} 0 0 1 {-r},{r} h{-(w - r):.2f} z"/>')
            vcls = "vh" if s == highlight else "v"
            out.append(f'<text class="{vcls}" x="{x0 + w + 5:.1f}" y="{y + bar_h - 3}">{label}</text></g>')
    if footnote:
        out.append(f'<text class="s" x="16" y="{height - 12}">{escape(footnote)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def markdown(records: Sequence[Record], cfg: Config, mode: str, agent_name: str | None) -> str:
    agg = aggregate(records)
    labels = BEHAVIORAL_LABELS if mode == "behavioral" else STRUCTURAL_LABELS
    loads = [ld.name for ld in cfg.loads]
    systems = [s for s in cfg.systems if s in agg]
    lines = [f"# RelapseBench - {mode} mode", ""]
    if mode == "structural":
        lines += ["Model-free and deterministic: every system receives the identical scripted feedback stream, "
                  "and we measure which rule variants end up in the memory context an agent would see. "
                  "This verifies the mechanism; it is **not** a measurement of model behaviour.", "",
                  "Cells are counts over families x seeds. Seeds only re-sample the distractor items, so the "
                  "counts are not independent samples and no confidence intervals are given.", ""]
    else:
        lines += [f"Agent: `{agent_name}`. Closed loop: the agent learns from its own failures, then is probed "
                  "under load. Rates are over probes; brackets are 95% Wilson intervals, which treat probes as "
                  "independent and so understate uncertainty (probes cluster by family).", ""]
    lines += ["| load | k | memory budget (tokens) | distractor items | padding tokens | stale + injected text in history |",
              "|---|---|---|---|---|---|"]
    for ld in cfg.loads:
        lines.append(f"| {ld.name} | {ld.k} | {ld.budget_tokens} | {ld.distractors} | {ld.padding_tokens} | "
                     f"{'yes' if ld.stale_in_context else 'no'} |")
    lines.append("")
    for metric, label in labels.items():
        lines += [f"### {label}", "", "| system | " + " | ".join(loads) + " |",
                  "|---|" + "---|" * len(loads)]
        for s in systems:
            cells = []
            for ld in loads:
                m = agg[s].get(ld, {}).get(metric)
                if m is None:
                    cells.append("-")
                elif mode == "structural":  # deterministic counts, not samples: no intervals
                    cells.append(f"{m['rate'] * 100:.0f}% ({m['k']}/{m['n']})")
                else:
                    lo, hi = m["ci"]
                    cells.append(f"{m['rate'] * 100:.0f}% [{lo * 100:.0f}-{hi * 100:.0f}] (n={m['n']})")
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["### Learning cost (trusted errors before the correction took effect)", "",
              "| system | " + " | ".join(loads) + " |", "|---|" + "---|" * len(loads)]
    for s in systems:
        cells = [f"{agg[s][ld]['learn_errors_mean']:.2f} ({agg[s][ld]['learned_rate'] * 100:.0f}% learned)"
                 if ld in agg[s] else "-" for ld in loads]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_report(out_dir: str, records: Sequence[Record], cfg: Config, mode: str,
                 agent_name: str | None = None, pipeline_test: bool = False) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    systems = [s for s in cfg.systems if any(r.system == s for r in records)]
    paths = {"json": os.path.join(out_dir, f"{mode}.json"), "md": os.path.join(out_dir, f"{mode}.md"),
             "svg": os.path.join(out_dir, f"{mode}.svg")}
    payload = {
        "benchmark": "RelapseBench",
        "mode": mode,
        "relock_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "agent": agent_name,
        "pipeline_test": pipeline_test,
        "config": cfg.to_dict(),
        "aggregate": aggregate(records),
        "records": [asdict(r) for r in records],
    }
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    md = markdown(records, cfg, mode, agent_name)
    if pipeline_test:
        md = ("> **Pipeline test only.** These numbers come from a scripted stand-in agent, not a model. "
              "They show that the harness runs; they are not results.\n\n" + md)
    with open(paths["md"], "w", encoding="utf-8") as f:
        f.write(md)

    n_probes = len({(r.seed, r.load, r.family, r.probe) for r in records})
    if mode == "structural":
        from .harness import STRUCTURAL_METRICS as M

        panels = [(STRUCTURAL_LABELS[k], _pooled(records, M[k]))
                  for k in ("stale_exposure", "poison_exposure", "clean")]
        title = "RelapseBench (structural): what reaches the agent's memory context"
        sub = (f"Same feedback stream for every system: drift correction, flaky timeouts, a prompt injection, "
               f"a noisy failure. {n_probes} probes per system, all load levels pooled.")
        foot = "Model-free measurement of memory contents, not of model behaviour. Tables: structural.md"
    else:
        from .harness import BEHAVIORAL_METRICS as M

        panels = [(f"Relapse at {ld.name}", _pooled([r for r in records if r.load == ld.name], M["relapse"]))
                  for ld in cfg.loads]
        title = f"RelapseBench (behavioural): relapse to the old rule under load - {agent_name}"
        sub = f"Fraction of probes where the agent followed the superseded rule. {n_probes} probes per system."
        foot = "Pipeline test with a scripted agent - not a result." if pipeline_test else "Tables: behavioral.md"
    with open(paths["svg"], "w", encoding="utf-8") as f:
        f.write(svg_panels(title, sub, panels, systems, footnote=foot))
    return paths
