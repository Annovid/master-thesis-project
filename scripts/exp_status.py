#!/usr/bin/env python3
"""Show live status of experiment sessions.

Usage:
    python scripts/exp_status.py                        # scan outputs/exp1_2_pilot
    python scripts/exp_status.py outputs/exp1_2_pilot
    python scripts/exp_status.py --watch                # refresh every 15s
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── ANSI colours (stripped when not a tty) ──────────────────────────────────
_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
RED    = lambda t: _c("31", t)
CYAN   = lambda t: _c("36", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)


# ── helpers ──────────────────────────────────────────────────────────────────

def _last_player_txt(session_dir: Path) -> Optional[Path]:
    """Return the most recently written player_*.txt, or None."""
    files = sorted(session_dir.rglob("player_*.txt"))
    return files[-1] if files else None


def _count_complete_rounds(session_dir: Path, num_players: int = 4) -> int:
    """Count rounds where all player files exist."""
    count = 0
    for r in range(1, 100):
        rd = session_dir / f"round_{r}"
        if not rd.exists():
            break
        if all((rd / f"player_{p}.txt").exists() for p in range(1, num_players + 1)):
            count += 1
        else:
            break
    return count


def _in_progress_rounds(session_dir: Path, num_players: int = 4) -> tuple[int, int]:
    """Return (last_complete_round, players_done_in_current_round)."""
    complete = _count_complete_rounds(session_dir, num_players)
    current_rd = session_dir / f"round_{complete + 1}"
    if not current_rd.exists():
        return complete, 0
    players_done = sum(
        1 for p in range(1, num_players + 1)
        if (current_rd / f"player_{p}.txt").exists()
    )
    return complete, players_done


def _session_status(model_dir: Path, sess_idx: int, n_rounds: int = 10) -> Dict[str, Any]:
    sess_dir = model_dir / f"session_{sess_idx}"
    summary_path = sess_dir / "session_summary.json"

    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        rounds_recorded = len(data.get("rounds", []))
        return {
            "session": sess_idx,
            "state": "done" if not data.get("incomplete") else "incomplete",
            "rounds_done": rounds_recorded,
            "abort_reason": data.get("abort_reason"),
        }

    if sess_dir.exists():
        complete, players_in_round = _in_progress_rounds(sess_dir)
        return {
            "session": sess_idx,
            "state": "running",
            "rounds_done": complete,
            "players_in_round": players_in_round,
        }

    return {"session": sess_idx, "state": "missing", "rounds_done": 0}


def _render_bar(done: int, total: int, width: int = 10) -> str:
    filled = int(done / total * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {done}/{total}"


def _format_model(model_id: str) -> str:
    # shorten long model names
    replacements = [
        ("anthropic/", ""),
        ("google/", ""),
        ("meta-llama/", ""),
        ("openai/", ""),
        ("mistralai/", ""),
    ]
    s = model_id
    for prefix, sub in replacements:
        s = s.replace(prefix, sub)
    return s[:32]


# ── main status function ─────────────────────────────────────────────────────

def print_status(root: Path, n_rounds: int = 10, n_sessions: int = 2) -> None:
    if _TTY:
        print("\033[2J\033[H", end="")  # clear screen

    print(BOLD(f"Experiment status  —  {root}"))
    print(DIM(f"{'─' * 72}"))

    model_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and "failed" not in d.name
        and d.name not in ("plots",)
    )

    if not model_dirs:
        print(RED("  No model directories found."))
        return

    all_done = True
    for md in model_dirs:
        model_name = md.name.replace("_", "/", 1)
        print(f"\n  {CYAN(_format_model(model_name))}")

        for s in range(n_sessions):
            info = _session_status(md, s, n_rounds)
            state = info["state"]
            rd = info["rounds_done"]

            if state == "done":
                bar = GREEN(_render_bar(rd, n_rounds))
                label = GREEN("✓ done")
            elif state == "running":
                pp = info.get("players_in_round", 0)
                bar = YELLOW(_render_bar(rd, n_rounds))
                label = YELLOW(f"↻ running  R{rd + 1} P{pp + 1}/{n_sessions * 2}")
                all_done = False
            elif state == "incomplete":
                bar = RED(_render_bar(rd, n_rounds))
                reason = (info.get("abort_reason") or "")[:50]
                label = RED(f"✗ incomplete  [{reason}]")
                all_done = False
            else:  # missing
                bar = DIM(_render_bar(0, n_rounds))
                label = DIM("– not started")
                all_done = False

            print(f"    session {s}  {bar}  {label}")

    # check for failed_attempt dirs
    failed = [d for d in root.rglob("*failed_attempt*") if d.is_dir() and "round_" not in str(d)]
    if failed:
        print(f"\n  {DIM('archived failed attempts:')}")
        for f in sorted(failed):
            print(f"    {DIM(str(f.relative_to(root)))}")

    print(f"\n{DIM('─' * 72)}")
    if all_done:
        print(GREEN("  All sessions complete."))
    else:
        print(YELLOW("  Experiment in progress."))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Show experiment session status.")
    parser.add_argument(
        "path", nargs="?", default="outputs/exp1_2_pilot",
        help="Root output directory (default: outputs/exp1_2_pilot)",
    )
    parser.add_argument(
        "--n-sessions", type=int, default=2,
        help="Expected sessions per model (default: 2)",
    )
    parser.add_argument(
        "--n-rounds", type=int, default=10,
        help="Expected rounds per session (default: 10)",
    )
    parser.add_argument(
        "--watch", "-w", action="store_true",
        help="Refresh every 15 seconds (Ctrl-C to exit).",
    )
    parser.add_argument(
        "--interval", type=int, default=15,
        help="Watch interval in seconds (default: 15).",
    )
    args = parser.parse_args()
    root = Path(args.path)

    if not root.exists():
        print(f"Directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    if args.watch:
        try:
            while True:
                print_status(root, n_rounds=args.n_rounds, n_sessions=args.n_sessions)
                print(DIM(f"  [refreshing every {args.interval}s — Ctrl-C to exit]"))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    else:
        print_status(root, n_rounds=args.n_rounds, n_sessions=args.n_sessions)


if __name__ == "__main__":
    main()
