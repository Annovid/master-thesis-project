"""Analyze Experiment 2.1 results — prompt-condition sweep.

Reads either the new one-shot run_summary.json (`runs`) or the legacy
session_summary.json files and prints a contribution table (models × prompt
conditions) with distance from the human baseline [8, 12]
(40-60% of endowment=20).

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
    "self_interest",
    "group_welfare",
    "inequality_aversion",
    "warm_glow",
    "expectations",
    "conditional_coop",
    "social_norms",
    "emotional_perspective",
    # Legacy label kept for old outputs/exp2_1.
    "emotional",
]


def _load_records(root: Path) -> List[Dict[str, Any]]:
    summary_path = root if root.is_file() else root / "run_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            if "runs" in summary:
                return [
                    {
                        "model": r.get("model_requested") or r.get("model"),
                        "prompt_label": r.get("prompt_label", "neutral"),
                        "contribution": r.get("parsed_action"),
                    }
                    for r in summary["runs"]
                ]
        except Exception as e:
            print(f"WARN: could not read {summary_path}: {e}")

    records = []
    for p in sorted(root.rglob("session_summary.json")):
        try:
            session = json.loads(p.read_text())
            records.append(
                {
                    "model": session["model_requested"],
                    "prompt_label": session.get("prompt_label", "neutral"),
                    "contribution": _extract_contribution(session),
                }
            )
        except Exception as e:
            print(f"WARN: could not read {p}: {e}")
    return records


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
    records = _load_records(input_dir)
    if not records:
        print(f"No Exp. 2.1 records found under {input_dir}")
        return

    models: List[str] = sorted({r["model"] for r in records})
    labels_found: List[str] = sorted({r.get("prompt_label", "neutral") for r in records})
    labels = [l for l in PROMPT_ORDER if l in labels_found] + [l for l in labels_found if l not in PROMPT_ORDER]

    contrib: Dict[Tuple[str, str], List[float]] = {}
    for r in records:
        val = r.get("contribution")
        if val is None:
            continue
        m = r["model"]
        lbl = r.get("prompt_label", "neutral")
        contrib.setdefault((m, lbl), []).append(float(val))

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
            vals = contrib.get((m, lbl), [])
            if not vals:
                cell = "-"
            else:
                val = sum(vals) / len(vals)
                pct = val / endowment * 100
                cell = f"{val:.1f} ({pct:.0f}%)"
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
            vals = contrib.get((m, lbl), [])
            val = sum(vals) / len(vals) if vals else None
            cell = _human_distance(val) if val is not None else "-"
            row += f"{cell:>{col_w}}"
        print(row)

    print()
    print("Best prompt per model (closest to human midpoint):")
    for m in models:
        best_lbl, best_dist = None, float("inf")
        for lbl in labels:
            vals = contrib.get((m, lbl), [])
            if vals:
                val = sum(vals) / len(vals)
                dist = abs(val - HUMAN_MID)
                if dist < best_dist:
                    best_dist, best_lbl = dist, lbl
        best_vals = contrib.get((m, best_lbl), []) if best_lbl else []
        val = sum(best_vals) / len(best_vals) if best_vals else None
        short = m.split("/")[-1][:model_w - 2]
        print(f"  {short:<{model_w - 2}}  {best_lbl}  ->  {val:.1f} ({val/endowment*100:.0f}%)" if val is not None else f"  {short}: ?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 2.1 results.")
    parser.add_argument("--input", type=Path, default=Path("outputs/exp2_1"))
    parser.add_argument("--endowment", type=float, default=20.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(args.input, args.endowment)
