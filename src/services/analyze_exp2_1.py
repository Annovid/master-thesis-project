"""Analyze Experiment 2.1 results — prompt-condition sweep.

Reads all session_summary.json files under the exp2_1 output directory and
prints a contribution table (models × prompt conditions) with distance from
the human baseline [8, 12] (40-60% of endowment=20).

Usage:
    python -m src.services.analyze_exp2_1
    python -m src.services.analyze_exp2_1 --input outputs/exp2_1 --endowment 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


HUMAN_LOW = 8.0
HUMAN_HIGH = 12.0
HUMAN_MID = (HUMAN_LOW + HUMAN_HIGH) / 2.0

PROMPT_ORDER = [
    "neutral",
    "persona",
    "inequality_aversion",
    "conditional_coop",
    "expectations",
    "social_norms",
    "emotional",
]


def _load_sessions(root: Path) -> List[Dict[str, Any]]:
    sessions = []
    for p in sorted(root.rglob("session_summary.json")):
        try:
            sessions.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"WARN: could not read {p}: {e}")
    return sessions


def _extract_contribution(session: Dict[str, Any]) -> Optional[float]:
    rounds = session.get("rounds", [])
    if not rounds:
        return None
    actions = rounds[0].get("actions", [])
    return actions[0] if actions else None


def _human_distance(val: float) -> str:
    if HUMAN_LOW <= val <= HUMAN_HIGH:
        return "IN_RANGE"
    dist = min(abs(val - HUMAN_LOW), abs(val - HUMAN_HIGH))
    return f"{dist:+.1f}"


def analyze(input_dir: Path, endowment: float) -> None:
    sessions = _load_sessions(input_dir)
    if not sessions:
        print(f"No session_summary.json found under {input_dir}")
        return

    models: List[str] = sorted({s["model_requested"] for s in sessions})
    labels_found: List[str] = sorted({s.get("prompt_label", "neutral") for s in sessions})
    labels = [l for l in PROMPT_ORDER if l in labels_found] + [l for l in labels_found if l not in PROMPT_ORDER]

    contrib: Dict[Tuple[str, str], Optional[float]] = {}
    for s in sessions:
        m = s["model_requested"]
        lbl = s.get("prompt_label", "neutral")
        contrib[(m, lbl)] = _extract_contribution(s)

    col_w = 20
    model_w = 30

    print(f"\n{'Experiment 2.1 — Contribution table':^{model_w + col_w * len(labels)}}")
    print(f"Endowment={endowment}  Human baseline=[{HUMAN_LOW:.0f}, {HUMAN_HIGH:.0f}] "
          f"({HUMAN_LOW/endowment*100:.0f}-{HUMAN_HIGH/endowment*100:.0f}%)")
    print()

    header = f"{'Model':<{model_w}}" + "".join(f"{lbl[:col_w-1]:>{col_w}}" for lbl in labels)
    print(header)
    print("-" * len(header))

    for m in models:
        short = m.split("/")[-1][:model_w - 2]
        row = f"{short:<{model_w}}"
        for lbl in labels:
            val = contrib.get((m, lbl))
            if val is None:
                cell = "-"
            else:
                pct = val / endowment * 100
                cell = f"{val:.0f} ({pct:.0f}%)"
            row += f"{cell:>{col_w}}"
        print(row)

    print()
    print("Distance from human range [8, 12]:")
    print(f"{'Model':<{model_w}}" + "".join(f"{lbl[:col_w-1]:>{col_w}}" for lbl in labels))
    print("-" * len(header))

    for m in models:
        short = m.split("/")[-1][:model_w - 2]
        row = f"{short:<{model_w}}"
        for lbl in labels:
            val = contrib.get((m, lbl))
            cell = _human_distance(val) if val is not None else "-"
            row += f"{cell:>{col_w}}"
        print(row)

    print()
    print("Best prompt per model (closest to human midpoint):")
    for m in models:
        best_lbl, best_dist = None, float("inf")
        for lbl in labels:
            val = contrib.get((m, lbl))
            if val is not None:
                dist = abs(val - HUMAN_MID)
                if dist < best_dist:
                    best_dist, best_lbl = dist, lbl
        val = contrib.get((m, best_lbl)) if best_lbl else None
        short = m.split("/")[-1][:model_w - 2]
        print(f"  {short:<{model_w - 2}}  {best_lbl}  →  {val:.0f} ({val/endowment*100:.0f}%)" if val is not None else f"  {short}: ?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 2.1 results.")
    parser.add_argument("--input", type=Path, default=Path("outputs/exp2_1"))
    parser.add_argument("--endowment", type=float, default=20.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(args.input, args.endowment)
