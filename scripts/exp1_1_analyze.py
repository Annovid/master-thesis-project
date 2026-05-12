"""Aggregate Experiment 1.1 results.jsonl into (model, temperature) metrics.

Run from project root:
    python scripts/exp1_1_analyze.py artifacts/exp1_1/<dir>/results.jsonl
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def entropy(values):
    if not values:
        return 0.0
    counts = defaultdict(int)
    for v in values:
        counts[v] += 1
    total = len(values)
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: exp1_1_analyze.py <results.jsonl>")
        return 2
    path = Path(sys.argv[1])
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    cells = defaultdict(list)
    invalid = defaultdict(list)
    for r in rows:
        key = (r["model"], r["temperature"])
        if r.get("parsed_action") is None or r.get("error"):
            invalid[key].append(r)
            continue
        cells[key].append(r["parsed_action"])

    all_keys = sorted(set(list(cells.keys()) + list(invalid.keys())))

    header = ("model", "T", "n", "mean", "median", "std", "min", "max",
              "p_zero", "p_full", "p_corner", "entropy", "invalid")
    widths = (40, 5, 3, 6, 6, 5, 4, 4, 6, 6, 7, 7, 7)
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))

    summary = []
    for key in all_keys:
        model, T = key
        actions = cells[key]
        n = len(actions)
        inv = len(invalid.get(key, []))
        if n == 0:
            row = (model, f"{T}", 0, "-", "-", "-", "-", "-", "-", "-", "-", "-", inv)
        else:
            mean = statistics.mean(actions)
            med = statistics.median(actions)
            std = statistics.stdev(actions) if n > 1 else 0.0
            mn = min(actions)
            mx = max(actions)
            p_zero = sum(1 for a in actions if a == 0) / n
            p_full = sum(1 for a in actions if a == 20) / n
            p_corner = p_zero + p_full
            h = entropy(actions)
            row = (model, f"{T}", n, f"{mean:.2f}", f"{med:.2f}", f"{std:.2f}",
                   f"{mn:g}", f"{mx:g}", f"{p_zero:.2f}", f"{p_full:.2f}",
                   f"{p_corner:.2f}", f"{h:.2f}", inv)
        summary.append(row)
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    print()
    print("Per-model summary (all temperatures pooled):")
    per_model = defaultdict(list)
    for (model, _T), actions in cells.items():
        per_model[model].extend(actions)
    for model, actions in sorted(per_model.items()):
        if not actions:
            continue
        n = len(actions)
        mean = statistics.mean(actions)
        std = statistics.stdev(actions) if n > 1 else 0.0
        mn, mx = min(actions), max(actions)
        print(f"  {model}: n={n} mean={mean:.2f} std={std:.2f} range=[{mn:g},{mx:g}]")

    out_path = path.parent / "summary.json"
    summary_records = []
    for (model, T), actions in cells.items():
        if not actions:
            continue
        n = len(actions)
        mean = statistics.mean(actions)
        med = statistics.median(actions)
        std = statistics.stdev(actions) if n > 1 else 0.0
        mn = min(actions); mx = max(actions)
        p_zero = sum(1 for a in actions if a == 0) / n
        p_full = sum(1 for a in actions if a == 20) / n
        summary_records.append({
            "model": model, "temperature": T, "n": n,
            "mean": mean, "median": med, "std": std,
            "min": mn, "max": mx,
            "p_zero": p_zero, "p_full": p_full,
            "p_corner": p_zero + p_full,
            "entropy": entropy(actions),
            "invalid": len(invalid.get((model, T), [])),
            "actions": actions,
        })
    out_path.write_text(json.dumps(summary_records, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
