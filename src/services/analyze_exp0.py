"""Analyze Experiment 0 outputs.

Reads run_summary.json produced by src/services/experiment0.py and computes
per (model, horizon, temperature) cell:
  - n_unique_actions / n        (numeric determinism)
  - n_unique_sha256 / n         (textual determinism)
  - action distribution (counts + share)
  - mean / max absolute deviation of action from the mode
  - count of parse errors

Writes analysis.json next to run_summary.json and prints a short table.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _cell_key(run: Dict[str, Any]) -> tuple:
    return (run["model"], run["horizon"], run["temperature"])


def analyze_cell(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(runs)
    actions = [r["parsed_action"] for r in runs if r["parsed_action"] is not None]
    parse_errors = sum(1 for r in runs if r["parsed_action"] is None)
    sha_set = {r["response_sha256"] for r in runs}

    action_counts = Counter(actions)
    n_unique_actions = len(action_counts)

    if actions:
        mode_action, _ = action_counts.most_common(1)[0]
        deviations = [abs(a - mode_action) for a in actions]
        mean_dev = statistics.mean(deviations)
        max_dev = max(deviations)
        mean_action = statistics.mean(actions)
        std_action = statistics.pstdev(actions) if len(actions) > 1 else 0.0
    else:
        mode_action = None
        mean_dev = max_dev = mean_action = std_action = None

    distribution = [
        {"action": a, "count": c, "share": c / n}
        for a, c in sorted(action_counts.items())
    ]

    return {
        "n": n,
        "n_parsed": len(actions),
        "n_parse_errors": parse_errors,
        "n_unique_actions": n_unique_actions,
        "share_unique_actions": n_unique_actions / max(len(actions), 1),
        "n_unique_sha256": len(sha_set),
        "share_unique_sha256": len(sha_set) / n,
        "mode_action": mode_action,
        "mean_action": mean_action,
        "std_action": std_action,
        "mean_abs_dev_from_mode": mean_dev,
        "max_abs_dev_from_mode": max_dev,
        "distribution": distribution,
    }


def analyze(summary_path: Path) -> Dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    runs = summary["runs"]

    cells: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in runs:
        cells.setdefault(_cell_key(r), []).append(r)

    results = []
    for (model, horizon, temperature), cell_runs in sorted(cells.items()):
        results.append(
            {
                "model": model,
                "horizon": horizon,
                "temperature": temperature,
                **analyze_cell(cell_runs),
            }
        )

    out_path = summary_path.parent / "analysis.json"
    out_path.write_text(
        json.dumps({"cells": results}, indent=2, ensure_ascii=False)
    )

    _print_table(results)
    print(f"\nSaved: {out_path}")
    return {"cells": results}


def _print_table(rows: List[Dict[str, Any]]) -> None:
    header = (
        f"{'model':<28} {'h':>3} {'T':>5} {'n':>3} "
        f"{'uniq_a':>7} {'uniq_sha':>9} {'mode':>6} "
        f"{'mean':>6} {'std':>5} {'max_dev':>7} {'errs':>4}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        mode = "—" if r["mode_action"] is None else f"{r['mode_action']:.2f}"
        mean = "—" if r["mean_action"] is None else f"{r['mean_action']:.2f}"
        std = "—" if r["std_action"] is None else f"{r['std_action']:.2f}"
        mxd = "—" if r["max_abs_dev_from_mode"] is None else f"{r['max_abs_dev_from_mode']:.2f}"
        print(
            f"{r['model']:<28} {r['horizon']:>3} {r['temperature']:>5.2f} {r['n']:>3} "
            f"{r['n_unique_actions']:>7} {r['n_unique_sha256']:>9} {mode:>6} "
            f"{mean:>6} {std:>5} {mxd:>7} {r['n_parse_errors']:>4}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 0 outputs.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/exp0/run_summary.json"),
        help="Path to run_summary.json produced by experiment0.py",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(args.summary)
