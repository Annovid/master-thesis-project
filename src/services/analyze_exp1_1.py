"""Analyze Experiment 1.1 outputs: contribution distribution by temperature.

Reads run_summary.json from experiment0.py runs and, per (model, horizon, T)
cell, computes:
  - n_total / n_parsed / n_parse_errors
  - mean, median, std
  - p_zero  = share of contributions == 0
  - p_endow = share of contributions == endowment
  - entropy of the binned distribution (Shannon, log base e)
  - mode and unique action count

Outputs (next to the input run_summary.json):
  - exp1_1_analysis.json   — all cell metrics
  - exp1_1_report.md       — Markdown table + plot references
  - plots/
      hist_<model_safe>_h<h>.png        — 4-panel histogram (one per T)
      summary_<model_safe>_h<h>.png     — mean±std / corners / entropy vs T

Usage:
    venv/bin/python -m src.services.analyze_exp1_1 \
        --summary outputs/exp1_1_gpt4o_probe/run_summary.json \
        --endowment 20.0
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _binned_entropy(actions: List[float], endowment: float, n_bins: int = 10) -> float:
    if not actions:
        return 0.0
    bins = np.linspace(0.0, endowment, n_bins + 1)
    counts, _ = np.histogram(actions, bins=bins)
    p = counts / counts.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def _cell_metrics(actions: List[float], endowment: float) -> Dict[str, Any]:
    if not actions:
        return {
            "n_parsed": 0, "mean": None, "median": None, "std": None,
            "p_zero": None, "p_endowment": None, "entropy": None,
            "mode_action": None, "n_unique_actions": 0,
        }
    counts = Counter(actions)
    mode_action, _ = counts.most_common(1)[0]
    return {
        "n_parsed": len(actions),
        "mean": statistics.mean(actions),
        "median": statistics.median(actions),
        "std": statistics.pstdev(actions) if len(actions) > 1 else 0.0,
        "p_zero": sum(1 for a in actions if a == 0) / len(actions),
        "p_endowment": sum(1 for a in actions if a == endowment) / len(actions),
        "entropy": _binned_entropy(actions, endowment),
        "mode_action": mode_action,
        "n_unique_actions": len(counts),
    }


def _group_runs(runs: Iterable[Dict[str, Any]]):
    cells = defaultdict(list)
    for r in runs:
        cells[(r["model"], r["horizon"], r["temperature"])].append(r)
    return cells


def analyze(summary_path: Path, endowment: float) -> Dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    runs = summary["runs"]
    cells = _group_runs(runs)

    cell_metrics: List[Dict[str, Any]] = []
    for (model, horizon, t), cell_runs in sorted(cells.items()):
        actions = [r["parsed_action"] for r in cell_runs if r["parsed_action"] is not None]
        errors = sum(1 for r in cell_runs if r["parsed_action"] is None)
        cell_metrics.append({
            "model": model,
            "horizon": horizon,
            "temperature": t,
            "n_total": len(cell_runs),
            "n_parse_errors": errors,
            **_cell_metrics(actions, endowment),
        })

    out_dir = summary_path.parent
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_paths = _make_plots(cells, cell_metrics, endowment, plots_dir)

    analysis_path = out_dir / "exp1_1_analysis.json"
    analysis_path.write_text(
        json.dumps({"endowment": endowment, "cells": cell_metrics}, indent=2, ensure_ascii=False)
    )

    report_path = out_dir / "exp1_1_report.md"
    report_path.write_text(_build_report(cell_metrics, plot_paths, endowment))

    _print_table(cell_metrics)
    print(f"\nSaved: {analysis_path}")
    print(f"Saved: {report_path}")
    for p in plot_paths:
        print(f"Saved: {p}")
    return {"cells": cell_metrics, "plots": [str(p) for p in plot_paths]}


def _make_plots(cells, cell_metrics, endowment, plots_dir: Path) -> List[Path]:
    paths: List[Path] = []
    by_mh: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for c in cell_metrics:
        by_mh[(c["model"], c["horizon"])].append(c)

    for (model, horizon), rows in sorted(by_mh.items()):
        rows = sorted(rows, key=lambda r: r["temperature"])
        ts = [r["temperature"] for r in rows]
        model_safe = _safe_name(model)

        # 1) Histogram panel — one subplot per T
        n_t = len(rows)
        fig, axes = plt.subplots(1, n_t, figsize=(3.5 * n_t, 3.2), sharey=True)
        if n_t == 1:
            axes = [axes]
        bins = np.linspace(0, endowment, 11)
        for ax, r in zip(axes, rows):
            actions = [
                run["parsed_action"]
                for run in cells[(r["model"], r["horizon"], r["temperature"])]
                if run["parsed_action"] is not None
            ]
            ax.hist(actions, bins=bins, edgecolor="black", color="#4c72b0")
            ax.set_xlim(0, endowment)
            ax.set_xticks(np.linspace(0, endowment, 6))
            ax.set_title(f"T={r['temperature']}  n={r['n_parsed']}")
            ax.set_xlabel("contribution")
        axes[0].set_ylabel("count")
        fig.suptitle(f"{model}  h={horizon}  (endowment={endowment})")
        fig.tight_layout()
        p = plots_dir / f"hist_{model_safe}_h{horizon}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

        # 2) Summary plot — mean±std, corner shares, entropy vs T, parse rate
        nan = float("nan")
        def _f(x):
            return nan if x is None else x

        fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))
        means = [_f(r["mean"]) for r in rows]
        stds = [_f(r["std"]) for r in rows]
        axs[0].errorbar(ts, means, yerr=stds, marker="o", capsize=4, color="#4c72b0")
        axs[0].set_xlabel("T")
        axs[0].set_ylabel("contribution")
        axs[0].set_ylim(0, endowment)
        axs[0].set_title("mean ± std")
        axs[0].grid(alpha=0.3)

        p_zero = [_f(r["p_zero"]) for r in rows]
        p_endow = [_f(r["p_endowment"]) for r in rows]
        axs[1].plot(ts, p_zero, marker="o", label=f"p(action == 0)", color="#dd8452")
        axs[1].plot(ts, p_endow, marker="s", label=f"p(action == {endowment})", color="#55a868")
        axs[1].set_xlabel("T")
        axs[1].set_ylabel("share")
        axs[1].set_ylim(-0.05, 1.05)
        axs[1].set_title("corner shares")
        axs[1].grid(alpha=0.3)
        axs[1].legend()

        ent = [_f(r["entropy"]) for r in rows]
        axs[2].plot(ts, ent, marker="o", color="#8172b2")
        axs[2].set_xlabel("T")
        axs[2].set_ylabel("entropy (nats)")
        axs[2].set_title(f"binned entropy (10 bins on [0,{endowment}])")
        axs[2].grid(alpha=0.3)
        axs[2].set_ylim(0, math.log(10) + 0.1)

        # parse-success rate as a separate "operational T-ceiling" indicator
        parse_rate = [r["n_parsed"] / r["n_total"] if r["n_total"] else 0 for r in rows]
        axs[3].plot(ts, parse_rate, marker="o", color="#c44e52")
        axs[3].set_xlabel("T")
        axs[3].set_ylabel("parse success rate")
        axs[3].set_ylim(-0.05, 1.05)
        axs[3].set_title("parse success rate")
        axs[3].grid(alpha=0.3)

        fig.suptitle(f"{model}  h={horizon}")
        fig.tight_layout()
        p = plots_dir / f"summary_{model_safe}_h{horizon}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    return paths


def _fmt(x, fmt=".2f"):
    return "—" if x is None else f"{x:{fmt}}"


def _print_table(rows: List[Dict[str, Any]]) -> None:
    header = (
        f"{'model':<32} {'h':>3} {'T':>5} {'n':>3} "
        f"{'mean':>6} {'med':>6} {'std':>5} {'p0':>5} {'pE':>5} {'H':>5} {'mode':>6} {'errs':>4}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model'][:32]:<32} {r['horizon']:>3} {r['temperature']:>5.2f} {r['n_total']:>3} "
            f"{_fmt(r['mean']):>6} {_fmt(r['median']):>6} {_fmt(r['std']):>5} "
            f"{_fmt(r['p_zero'], '.2f'):>5} {_fmt(r['p_endowment'], '.2f'):>5} "
            f"{_fmt(r['entropy']):>5} {_fmt(r['mode_action'], '.1f'):>6} {r['n_parse_errors']:>4}"
        )


def _build_report(rows: List[Dict[str, Any]], plots: List[Path], endowment: float) -> str:
    out = [
        "# Эксперимент 1.1 — распределение вклада по температуре",
        "",
        f"Эндоумент: {endowment}. Метрики из дизайна (`docs/06/3. Дизайн эксперимента.md`).",
        "",
        "## Сводная таблица",
        "",
        "| model | h | T | n | mean | median | std | p(0) | p(E) | entropy | mode | errs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        out.append(
            f"| {r['model']} | {r['horizon']} | {r['temperature']} | {r['n_total']} | "
            f"{_fmt(r['mean'])} | {_fmt(r['median'])} | {_fmt(r['std'])} | "
            f"{_fmt(r['p_zero'], '.2f')} | {_fmt(r['p_endowment'], '.2f')} | "
            f"{_fmt(r['entropy'])} | {_fmt(r['mode_action'], '.1f')} | {r['n_parse_errors']} |"
        )
    out += ["", "Колонки: `p(0)` — доля вкладов 0; `p(E)` — доля вкладов = endowment; `entropy` — Shannon entropy 10-биновой гистограммы (натов); `mode` — наиболее частое значение."]
    out += ["", "## Графики", ""]
    for p in plots:
        out.append(f"- `{p.relative_to(p.parent.parent)}`")
    return "\n".join(out) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 1.1 outputs.")
    parser.add_argument(
        "--summary", type=Path, required=True,
        help="Path to run_summary.json produced by experiment0.py",
    )
    parser.add_argument("--endowment", type=float, default=20.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(args.summary, args.endowment)
