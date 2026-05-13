#!/usr/bin/env python3
"""Status overview for Experiment 2.1 (one-shot prompt-condition sweep).

Usage:
    python scripts/exp2_1_status.py
    python scripts/exp2_1_status.py --input outputs/exp2_1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROMPT_ORDER = [
    "neutral",
    "persona",
    "inequality_aversion",
    "conditional_coop",
    "expectations",
    "social_norms",
    "emotional",
]

HUMAN_LOW, HUMAN_HIGH = 8.0, 12.0
ENDOWMENT = 20.0


def _load_summary(root: Path) -> list[dict]:
    sessions = []
    for path in sorted(root.rglob("session_summary.json")):
        try:
            sessions.append(json.loads(path.read_text()))
        except Exception:
            pass
    if sessions:
        return sessions

    path = root / "run_summary.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("sessions", [])
    except Exception:
        return []


def _contribution(session: dict) -> float | None:
    rounds = session.get("rounds", [])
    if not rounds:
        return None
    actions = rounds[0].get("actions", [])
    return float(actions[0]) if actions else None


def _is_running(pid_hint: int | None = None) -> str:
    try:
        result = subprocess.run(
            ["pgrep", "-fa", "experiment2_1"],
            capture_output=True, text=True,
        )
        lines = [l for l in result.stdout.strip().splitlines() if "experiment2_1" in l]
        if lines:
            return f"RUNNING  ({lines[0].split()[0]})"
    except Exception:
        pass
    return "stopped"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/exp2_1"))
    args = parser.parse_args()

    root = args.input
    sessions = _load_summary(root)

    # Index by (model_short, prompt_label)
    data: dict[tuple[str, str], dict] = {}
    models_seen: list[str] = []
    for s in sessions:
        m = s["model_requested"]
        lbl = s.get("prompt_label", "neutral")
        data[(m, lbl)] = s
        if m not in models_seen:
            models_seen.append(m)

    labels = [l for l in PROMPT_ORDER if any((m, l) in data for m in models_seen)]
    labels += [l for l in {s.get("prompt_label") for s in sessions} if l not in labels]
    total_cells = len(models_seen) * len(PROMPT_ORDER)

    complete = sum(1 for s in sessions if not s.get("incomplete") and _contribution(s) is not None)
    incomplete = sum(1 for s in sessions if s.get("incomplete"))
    pending = total_cells - len(sessions)

    print(f"\nExperiment 2.1 status  —  {root}")
    print(f"Process: {_is_running()}")
    print(f"Cells: {complete} complete  |  {incomplete} incomplete  |  {pending} pending  |  {total_cells} total")
    print()

    # ── contribution table ───────────────────────────────────────────────────
    col_w = 16
    model_w = 28
    header = f"{'Model':<{model_w}}" + "".join(f"{l[:col_w-1]:>{col_w}}" for l in labels)
    print(header)
    print("─" * len(header))

    for m in models_seen:
        short = m.split("/")[-1][:model_w - 2]
        row = f"{short:<{model_w}}"
        for lbl in labels:
            s = data.get((m, lbl))
            if s is None:
                cell = "·"
            elif s.get("incomplete"):
                cell = "ERR"
            else:
                c = _contribution(s)
                if c is None:
                    cell = "?"
                else:
                    marker = "✓" if HUMAN_LOW <= c <= HUMAN_HIGH else " "
                    cell = f"{c:.0f}({c/ENDOWMENT*100:.0f}%){marker}"
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"\n  ✓ = in human range [{HUMAN_LOW:.0f}–{HUMAN_HIGH:.0f}] (40–60% of {ENDOWMENT:.0f})")

    # ── incomplete details ───────────────────────────────────────────────────
    inc_sessions = [s for s in sessions if s.get("incomplete")]
    if inc_sessions:
        print(f"\nIncomplete sessions ({len(inc_sessions)}):")
        for s in inc_sessions:
            reason = s.get("abort_reason", "unknown")[:90]
            print(f"  {s['model_requested'].split('/')[-1]:<25}  {s.get('prompt_label','?'):<22}  {reason}")

    print()


if __name__ == "__main__":
    main()
