"""Print per-round contributions table for Experiment 1.2 sessions.

Reads session_summary.json files (either directly, or by scanning a directory)
and prints a compact table with one row per round and one column per player.
Also prints per-round mean, total pot, and per-player payoff.

Usage:
    venv/bin/python -m src.services.print_exp1_2_table \\
        --path outputs/exp1_2_pilot

    # or a specific session
    venv/bin/python -m src.services.print_exp1_2_table \\
        --path outputs/exp1_2_pilot/anthropic_claude-opus-4.7/session_0/session_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _fmt_action(x: Any) -> str:
    if x is None:
        return "  —  "
    try:
        return f"{float(x):5.1f}"
    except (TypeError, ValueError):
        return f"{str(x):>5s}"


def _fmt_float(x: Any, spec: str = "6.2f") -> str:
    if x is None:
        return "  —   "
    return f"{x:{spec}}"


def print_session(session: Dict[str, Any]) -> None:
    model = session.get("model_requested", "?")
    sess_id = session.get("session", "?")
    temp = session.get("temperature", "?")
    transp = session.get("transparency", "?")
    n_players = session.get("num_players", 4)
    incomplete = session.get("incomplete", False)
    multiplier = session.get("multiplier")

    header_meta = (
        f"Model: {model}  Session: {sess_id}  T={temp}  transparency={transp}  "
        f"players={n_players}"
        + ("  [INCOMPLETE]" if incomplete else "")
    )
    print(header_meta)

    cols = ["Round"] + [f"P{p + 1}" for p in range(n_players)] + ["mean", "pot", "payoff_each"]
    widths = [5] + [6] * n_players + [6, 6, 11]
    line = " ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    print(line)
    print("-" * len(line))

    rounds = session.get("rounds", [])
    means: List[float] = []
    for r in rounds:
        round_num = r.get("round", "?")
        actions = r.get("actions") or []
        payoffs = r.get("payoffs") or []
        # Pad to num_players width if a session aborted mid-round
        acts_padded = list(actions) + [None] * (n_players - len(actions))

        cells = [f"{round_num:>5}"]
        for a in acts_padded:
            cells.append(f"{_fmt_action(a):>6}")

        parsed = [a for a in actions if isinstance(a, (int, float))]
        if parsed:
            m = sum(parsed) / len(parsed)
            means.append(m)
            cells.append(_fmt_float(m, "6.2f"))
            pot_total = sum(parsed) * (multiplier or 1.6) if multiplier else None
            cells.append(_fmt_float(pot_total, "6.1f"))
        else:
            cells.append("  —   ")
            cells.append("  —   ")

        if payoffs:
            # Each round in PGG all players receive the same public-pot share, but
            # individual payoff differs by own contribution. Show the average to
            # keep it one column; per-player payoff is recoverable from actions.
            avg_payoff = sum(payoffs) / len(payoffs)
            cells.append(_fmt_float(avg_payoff, "10.2f") + " ")
        else:
            cells.append("    —    ")

        print(" ".join(cells))

    if means:
        avg_overall = sum(means) / len(means)
        first = means[0]
        last = means[-1]
        print(f"   {'session avg':<{sum(widths[1:n_players + 1]) + n_players}} avg={avg_overall:5.2f}  R1={first:5.2f}  Rlast={last:5.2f}  ΔR1-Rlast={first - last:+.2f}")
    if incomplete:
        reason = session.get("abort_reason") or "(no reason recorded)"
        print(f"   ABORT_REASON: {reason}")


def _iter_sessions(path: Path) -> List[Dict[str, Any]]:
    if path.is_file():
        data = json.loads(path.read_text())
        if "sessions" in data:
            return list(data["sessions"])
        return [data]

    sessions: List[Dict[str, Any]] = []
    for sp in sorted(path.rglob("session_summary.json")):
        sessions.append(json.loads(sp.read_text()))
    if not sessions:
        # fallback: aggregated run_summary.json
        for sp in sorted(path.rglob("run_summary.json")):
            data = json.loads(sp.read_text())
            sessions.extend(data.get("sessions", []))
    return sessions


def main() -> int:
    parser = argparse.ArgumentParser(description="Print Exp 1.2 contributions table.")
    parser.add_argument(
        "--path", type=Path, default=Path("outputs/exp1_2_pilot"),
        help="Directory to scan, or a single session_summary.json / run_summary.json file.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Filter by model (substring match against model_requested).",
    )
    parser.add_argument(
        "--session", type=int, default=None,
        help="Filter by session index.",
    )
    args = parser.parse_args()

    sessions = _iter_sessions(args.path)
    if args.model:
        sessions = [s for s in sessions if args.model in s.get("model_requested", "")]
    if args.session is not None:
        sessions = [s for s in sessions if s.get("session") == args.session]

    if not sessions:
        print(f"No sessions found under {args.path}")
        return 1

    for i, s in enumerate(sessions):
        if i:
            print()
        print_session(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
