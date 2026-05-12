#!/usr/bin/env python3
"""Export Experiment 1.2 sessions to Markdown.

One .md file per model. Each file contains a table per session:
rows = rounds, columns = P1..P4, mean, pot, payoff_each.

Usage:
    python scripts/sessions_to_md.py
    python scripts/sessions_to_md.py --input outputs/exp1_2_pilot --output outputs/exp1_2_md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_slope(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    return 0.0 if denom == 0 else sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / denom


def _fmt(x: Any, decimals: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x:.{decimals}f}"


def _session_to_md(session: Dict[str, Any]) -> str:
    model     = session["model_requested"]
    sess_idx  = session["session"]
    temp      = session["temperature"]
    transp    = session["transparency"]
    endow     = session["endowment"]
    mult      = session["multiplier"]
    n         = session["num_players"]

    lines: List[str] = []
    lines.append(f"### Session {sess_idx}")
    lines.append(f"T={temp} · transparency={transp} · endowment={endow} · multiplier={mult}")
    lines.append("")

    # header
    header = "| Round |" + "".join(f" P{p} |" for p in range(1, n + 1)) + " mean | pot | payoff |"
    sep    = "|" + "------:|" * (1 + n) + "------:|-----:|-------:|"
    lines.append(header)
    lines.append(sep)

    round_means: List[float] = []

    for rnd in session.get("rounds", []):
        if rnd.get("incomplete"):
            continue
        round_num = rnd["round"]
        actions   = rnd.get("actions") or []
        payoffs   = rnd.get("payoffs") or []
        acts_pad  = list(actions) + [None] * (n - len(actions))

        parsed = [a for a in actions if isinstance(a, (int, float))]
        mean_val   = sum(parsed) / len(parsed) if parsed else None
        pot_val    = sum(parsed) * mult if parsed else None
        payoff_avg = sum(payoffs) / len(payoffs) if payoffs else None
        if mean_val is not None:
            round_means.append(mean_val)

        row = f"| {round_num} |"
        for a in acts_pad:
            row += f" {_fmt(a, 1)} |"
        row += f" {_fmt(mean_val)} | {_fmt(pot_val, 1)} | {_fmt(payoff_avg)} |"
        lines.append(row)

    lines.append("")

    # footer stats
    if round_means:
        xs = list(range(1, len(round_means) + 1))
        slope = _safe_slope(xs, round_means)
        avg   = sum(round_means) / len(round_means)
        r1    = round_means[0]
        rl    = round_means[-1]
        delta = r1 - rl
        sign  = "+" if delta >= 0 else ""
        lines.append(
            f"**avg** {_fmt(avg)} · **R1** {_fmt(r1)} · **Rlast** {_fmt(rl)} · "
            f"**ΔR1-Rlast** {sign}{_fmt(delta)} · **slope** {_fmt(slope, 4)}"
        )
        lines.append("")

    return "\n".join(lines)


def _model_to_md(model: str, sessions: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"# {model}")
    lines.append("")
    for sess in sorted(sessions, key=lambda s: s["session"]):
        lines.append(_session_to_md(sess))
    return "\n".join(lines) + "\n"


def _iter_sessions(root: Path) -> List[Dict[str, Any]]:
    result = []
    for sp in sorted(root.rglob("session_summary.json")):
        if "failed_attempt" in str(sp):
            continue
        data = json.loads(sp.read_text())
        if not data.get("incomplete"):
            result.append(data)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Exp 1.2 sessions to Markdown.")
    parser.add_argument("--input",  type=Path, default=Path("outputs/exp1_2_pilot"))
    parser.add_argument("--output", type=Path, default=Path("outputs/exp1_2_md"))
    args = parser.parse_args()

    sessions = _iter_sessions(args.input)
    if not sessions:
        print(f"No complete sessions found under {args.input}", file=sys.stderr)
        sys.exit(1)

    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in sessions:
        by_model[s["model_requested"]].append(s)

    args.output.mkdir(parents=True, exist_ok=True)

    for model, model_sessions in sorted(by_model.items()):
        safe = model.replace("/", "_").replace(":", "_")
        path = args.output / f"{safe}.md"
        path.write_text(_model_to_md(model, model_sessions), encoding="utf-8")
        print(f"  {model:<35} {len(model_sessions)} sessions  →  {path}")

    print(f"\nSaved {len(by_model)} files to {args.output}/")


if __name__ == "__main__":
    main()
