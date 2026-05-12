"""Analyze Experiment 1.2 (full-game PGG) pilot outputs.

Reads one or more run_summary.json produced by experiment1_2.py and computes,
per (model, session):
  - mean / std contribution per round
  - decay slope (OLS regression of round_mean_contribution ~ round)
  - mean(R1), mean(Rlast), mean(R1) - mean(Rlast)
  - share at corners (0 / endowment) per round
  - conditional cooperation: corr(player_i_contrib_{r+1}, mean_others_contrib_r)
  - heterogeneity: std across 4 players per round, averaged across rounds

Aggregates per model: agreement between sessions on decay sign and magnitude.

Outputs (in --output dir, defaults to exp1_2_pilot/):
  exp1_2_analysis.json
  exp1_2_report.md
  plots/
    rounds_<model_safe>.png
    decay_summary.png

Usage:
    venv/bin/python -m src.services.analyze_exp1_2 \\
        --summary outputs/exp1_2_pilot/run_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _ols_slope(xs: List[float], ys: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Return (slope, intercept) for y ~ x; None if degenerate."""
    if len(xs) < 2:
        return None, None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None, None
    slope = sxy / sxx
    intercept = my - slope * mx
    return slope, intercept


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def _session_metrics(session: Dict[str, Any], endowment: float) -> Dict[str, Any]:
    """Compute per-session metrics from a session_summary dict."""
    rounds = [r for r in session.get("rounds", []) if r.get("payoffs") is not None]
    num_players = session.get("num_players", 4)

    round_means: List[Optional[float]] = []
    round_stds: List[Optional[float]] = []
    round_actions: List[List[float]] = []
    p_zero_round: List[float] = []
    p_endow_round: List[float] = []

    for r in rounds:
        acts = r.get("actions") or []
        if not acts:
            round_means.append(None)
            round_stds.append(None)
            p_zero_round.append(0.0)
            p_endow_round.append(0.0)
            round_actions.append([])
            continue
        round_actions.append(acts)
        round_means.append(statistics.mean(acts))
        round_stds.append(statistics.pstdev(acts) if len(acts) > 1 else 0.0)
        p_zero_round.append(sum(1 for a in acts if a == 0) / len(acts))
        p_endow_round.append(sum(1 for a in acts if a == endowment) / len(acts))

    # Decay slope on completed rounds where round_means is defined
    completed_rounds = [(i + 1, m) for i, m in enumerate(round_means) if m is not None]
    slope, intercept = (None, None)
    if len(completed_rounds) >= 2:
        slope, intercept = _ols_slope(
            [float(r) for r, _ in completed_rounds],
            [m for _, m in completed_rounds],
        )

    r1 = round_means[0] if round_means and round_means[0] is not None else None
    rlast = round_means[-1] if round_means and round_means[-1] is not None else None
    r1_minus_rlast = (r1 - rlast) if (r1 is not None and rlast is not None) else None

    # Conditional cooperation: for each player p and round r >= 2,
    # compare player_p_action(r) vs mean of others' actions at r-1.
    cc_xs: List[float] = []
    cc_ys: List[float] = []
    for r_idx in range(1, len(round_actions)):
        prev = round_actions[r_idx - 1]
        cur = round_actions[r_idx]
        if not prev or not cur or len(prev) != num_players or len(cur) != num_players:
            continue
        for p in range(num_players):
            others_mean = (sum(prev) - prev[p]) / (num_players - 1)
            cc_xs.append(others_mean)
            cc_ys.append(cur[p])
    conditional_cooperation = _pearson(cc_xs, cc_ys)

    # Per-round heterogeneity (already in round_stds), averaged over completed rounds.
    nonempty_stds = [s for s in round_stds if s is not None]
    mean_round_std = statistics.mean(nonempty_stds) if nonempty_stds else None

    return {
        "rounds_completed": len([m for m in round_means if m is not None]),
        "round_means": round_means,
        "round_stds": round_stds,
        "round_p_zero": p_zero_round,
        "round_p_endowment": p_endow_round,
        "decay_slope": slope,
        "decay_intercept": intercept,
        "round1_mean": r1,
        "round_last_mean": rlast,
        "round1_minus_last": r1_minus_rlast,
        "conditional_cooperation_r": conditional_cooperation,
        "mean_within_round_std": mean_round_std,
    }


def _aggregate_per_model(per_session: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate session-level metrics into a per-model verdict."""
    slopes = [s["metrics"]["decay_slope"] for s in per_session if s["metrics"]["decay_slope"] is not None]
    r1_minus_last = [s["metrics"]["round1_minus_last"] for s in per_session if s["metrics"]["round1_minus_last"] is not None]
    cc = [s["metrics"]["conditional_cooperation_r"] for s in per_session if s["metrics"]["conditional_cooperation_r"] is not None]

    slope_signs = {("-" if s < 0 else ("+" if s > 0 else "0")) for s in slopes}
    decay_consistent = (len(slopes) >= 2) and (len(slope_signs) == 1) and (slope_signs != {"0"})

    return {
        "n_sessions_analyzed": len(per_session),
        "n_incomplete": sum(1 for s in per_session if s["incomplete"]),
        "slopes": slopes,
        "slope_sign_consistent": decay_consistent,
        "mean_slope": statistics.mean(slopes) if slopes else None,
        "r1_minus_last": r1_minus_last,
        "mean_r1_minus_last": statistics.mean(r1_minus_last) if r1_minus_last else None,
        "mean_conditional_cooperation_r": statistics.mean(cc) if cc else None,
    }


def analyze(summary_paths: List[Path], endowment: float, out_dir: Path) -> Dict[str, Any]:
    sessions_by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_sessions: List[Dict[str, Any]] = []
    for sp in summary_paths:
        data = json.loads(sp.read_text())
        for s in data.get("sessions", []):
            all_sessions.append(s)
            sessions_by_model[s["model_requested"]].append(s)

    per_model: List[Dict[str, Any]] = []
    for model in sorted(sessions_by_model.keys()):
        sess = sorted(sessions_by_model[model], key=lambda s: s["session"])
        session_records: List[Dict[str, Any]] = []
        for s in sess:
            metrics = _session_metrics(s, endowment)
            session_records.append({
                "session": s["session"],
                "incomplete": s.get("incomplete", False),
                "abort_reason": s.get("abort_reason"),
                "rounds_completed": metrics["rounds_completed"],
                "metrics": metrics,
            })
        per_model.append({
            "model": model,
            "sessions": session_records,
            "aggregate": _aggregate_per_model(session_records),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_paths = _make_plots(per_model, endowment, plots_dir)

    analysis = {
        "endowment": endowment,
        "summary_paths": [str(p) for p in summary_paths],
        "per_model": per_model,
    }
    (out_dir / "exp1_2_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False)
    )
    (out_dir / "exp1_2_report.md").write_text(_build_report(per_model, plot_paths, endowment))

    _print_table(per_model)
    print(f"\nSaved: {out_dir / 'exp1_2_analysis.json'}")
    print(f"Saved: {out_dir / 'exp1_2_report.md'}")
    for p in plot_paths:
        print(f"Saved: {p}")
    return analysis


def _make_plots(per_model: List[Dict[str, Any]], endowment: float, plots_dir: Path) -> List[Path]:
    paths: List[Path] = []

    # Per-model: round-by-round mean contribution, one curve per session.
    for m in per_model:
        model = m["model"]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for s in m["sessions"]:
            means = s["metrics"]["round_means"]
            xs = [i + 1 for i, v in enumerate(means) if v is not None]
            ys = [v for v in means if v is not None]
            label = f"session {s['session']}"
            if s["incomplete"]:
                label += " (incomplete)"
            ax.plot(xs, ys, marker="o", label=label)
        ax.axhline(endowment / 2, color="grey", lw=0.7, ls="--", alpha=0.6, label="endowment/2")
        ax.axhline(endowment, color="grey", lw=0.5, ls=":", alpha=0.4)
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(-0.5, endowment + 0.5)
        ax.set_xlabel("round")
        ax.set_ylabel("mean contribution (across 4 players)")
        ax.set_title(f"{model}  T=0.7  transparency=on")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        p = plots_dir / f"rounds_{_safe_name(model)}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    # Decay summary across all models.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = []
    slope_means: List[float] = []
    slope_sessions: List[List[float]] = []
    for m in per_model:
        slopes = m["aggregate"]["slopes"]
        if not slopes:
            continue
        labels.append(m["model"][-28:])
        slope_means.append(m["aggregate"]["mean_slope"])
        slope_sessions.append(slopes)

    if labels:
        ys = np.arange(len(labels))
        ax.barh(ys, slope_means, color="#4c72b0", alpha=0.6, label="mean slope")
        # Overlay individual session slopes
        for y, sess in zip(ys, slope_sessions):
            ax.scatter(sess, [y] * len(sess), color="black", marker="x", s=40, zorder=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("decay slope (Δ contribution per round)")
        ax.set_title("Decay slope by model (×: per-session; bar: session-mean)")
        ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    p = plots_dir / "decay_summary.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    return paths


def _fmt(x: Optional[float], spec: str = ".2f") -> str:
    return "—" if x is None else f"{x:{spec}}"


def _print_table(per_model: List[Dict[str, Any]]) -> None:
    header = (
        f"{'model':<40} {'sess':>4} {'R1':>5} {'Rlast':>6} {'R1-Rlast':>9} "
        f"{'slope':>6} {'cc_r':>6} {'std_pl':>6} {'incompl':>7}"
    )
    print(header)
    print("-" * len(header))
    for m in per_model:
        for s in m["sessions"]:
            met = s["metrics"]
            print(
                f"{m['model'][:40]:<40} {s['session']:>4} "
                f"{_fmt(met['round1_mean']):>5} {_fmt(met['round_last_mean']):>6} "
                f"{_fmt(met['round1_minus_last']):>9} {_fmt(met['decay_slope'], '.3f'):>6} "
                f"{_fmt(met['conditional_cooperation_r']):>6} "
                f"{_fmt(met['mean_within_round_std']):>6} {str(s['incomplete']):>7}"
            )
        agg = m["aggregate"]
        cons = "consistent" if agg["slope_sign_consistent"] else "mixed"
        print(
            f"{'  ↳ aggregate':<40} {agg['n_sessions_analyzed']:>4} "
            f"{'':>5} {'':>6} {_fmt(agg['mean_r1_minus_last']):>9} "
            f"{_fmt(agg['mean_slope'], '.3f'):>6} {_fmt(agg['mean_conditional_cooperation_r']):>6} "
            f"{'':>6} {cons:>7}"
        )


def _build_report(per_model: List[Dict[str, Any]], plot_paths: List[Path], endowment: float) -> str:
    out = [
        "# Эксперимент 1.2 — decay of cooperation (pilot)",
        "",
        f"Endowment: {endowment}. Полная PGG, 10 раундов, 4 копии модели, T=0.7, "
        "transparency=on. Дизайн: `docs/06/3. Дизайн эксперимента.md`.",
        "",
        "## Сводная таблица по сессиям",
        "",
        "| model | session | rounds | R1 | R_last | R1−R_last | slope | cc_r | mean_std_within_round | incomplete |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in per_model:
        for s in m["sessions"]:
            met = s["metrics"]
            out.append(
                f"| {m['model']} | {s['session']} | {s['rounds_completed']} | "
                f"{_fmt(met['round1_mean'])} | {_fmt(met['round_last_mean'])} | "
                f"{_fmt(met['round1_minus_last'])} | {_fmt(met['decay_slope'], '.3f')} | "
                f"{_fmt(met['conditional_cooperation_r'])} | "
                f"{_fmt(met['mean_within_round_std'])} | {s['incomplete']} |"
            )
    out += [
        "",
        "Колонки: `R1` и `R_last` — средний по 4 игрокам вклад в первом и последнем "
        "сыгранном раунде; `R1−R_last` — простая разность (положительная = есть decay); "
        "`slope` — OLS-наклон `mean_contribution(round) ~ round` по всем сыгранным "
        "раундам сессии (отрицательный = decay); `cc_r` — корреляция Пирсона между "
        "вкладом игрока в раунде r+1 и средним вкладом остальных в раунде r "
        "(>0 = conditional cooperation); `mean_std_within_round` — средняя дисперсия "
        "вкладов между 4 игроками внутри одного раунда, усреднённая по раундам "
        "(индикатор гетерогенности).",
        "",
        "## Сводка по моделям",
        "",
        "| model | n_sessions | n_incomplete | mean_slope | slope_sign_consistent | mean R1−R_last | mean cc_r |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in per_model:
        a = m["aggregate"]
        out.append(
            f"| {m['model']} | {a['n_sessions_analyzed']} | {a['n_incomplete']} | "
            f"{_fmt(a['mean_slope'], '.3f')} | {a['slope_sign_consistent']} | "
            f"{_fmt(a['mean_r1_minus_last'])} | {_fmt(a['mean_conditional_cooperation_r'])} |"
        )
    out += ["", "## Графики", ""]
    for p in plot_paths:
        out.append(f"- `{p.relative_to(p.parent.parent)}`")
    return "\n".join(out) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 1.2 outputs.")
    parser.add_argument(
        "--summary", type=Path, nargs="+", required=True,
        help="One or more run_summary.json files produced by experiment1_2.py.",
    )
    parser.add_argument("--endowment", type=float, default=20.0)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for analysis + plots. Default: parent of first summary.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.output or args.summary[0].parent
    analyze(args.summary, args.endowment, out_dir)
