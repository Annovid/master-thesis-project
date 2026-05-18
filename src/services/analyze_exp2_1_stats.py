"""Statistical tests for Experiment 2.1.

This script intentionally uses only the Python standard library so it can run in
the thesis environment without numpy/scipy/statsmodels. It estimates an OLS
fixed-effects model on raw samples:

    g ~ prompt_label + model

with neutral and openai/gpt-4o as baselines. P-values use a normal
approximation, which is acceptable here as a lightweight diagnostic layer but
should not be treated as a substitute for a full statistical analysis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROMPT_ORDER = [
    "persona",
    "self_interest",
    "group_welfare",
    "inequality_aversion",
    "warm_glow",
    "conditional_coop",
    "expectations",
    "social_norms",
    "emotional_perspective",
]

MODEL_ORDER = [
    "meta-llama/llama-4-maverick",
    "anthropic/claude-opus-4.7",
    "google/gemini-2.5-pro",
]

BASE_PROMPT = "neutral"
BASE_MODEL = "openai/gpt-4o"


def load_rows(path: Path) -> List[dict]:
    summary_path = path if path.is_file() else path / "run_summary.json"
    data = json.loads(summary_path.read_text())
    rows = []
    for run in data["runs"]:
        y = run.get("parsed_action")
        if y is None:
            continue
        rows.append(
            {
                "y": float(y),
                "prompt": run.get("prompt_label", BASE_PROMPT),
                "model": run.get("model_requested") or run.get("model"),
            }
        )
    return rows


def transpose(a: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*a)]


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def matvec(a: Sequence[Sequence[float]], x: Sequence[float]) -> List[float]:
    return [sum(ai * xi for ai, xi in zip(row, x)) for row in a]


def invert(a: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(a)
    aug = [list(row) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [v / div for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [rv - factor * cv for rv, cv in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def ols(rows: Sequence[dict]) -> Tuple[List[str], List[float], List[float], float, float, int]:
    names = ["Intercept"] + [f"prompt:{p}" for p in PROMPT_ORDER] + [f"model:{m}" for m in MODEL_ORDER]
    x = []
    y = []
    for row in rows:
        x.append(
            [1.0]
            + [1.0 if row["prompt"] == p else 0.0 for p in PROMPT_ORDER]
            + [1.0 if row["model"] == m else 0.0 for m in MODEL_ORDER]
        )
        y.append(row["y"])

    xt = transpose(x)
    xtx = matmul(xt, x)
    xtx_inv = invert(xtx)
    xty = [sum(row_i * yi for row_i, yi in zip(row, y)) for row in xt]
    beta = matvec(xtx_inv, xty)

    yhat = matvec(x, beta)
    resid = [yi - yhi for yi, yhi in zip(y, yhat)]
    n = len(y)
    k = len(beta)
    df = n - k
    sse = sum(e * e for e in resid)
    mean_y = sum(y) / n
    tss = sum((yi - mean_y) ** 2 for yi in y)
    sigma2 = sse / df
    se = [math.sqrt(max(0.0, sigma2 * xtx_inv[i][i])) for i in range(k)]
    r2 = 1.0 - sse / tss if tss else 0.0
    rmse = math.sqrt(sse / n)
    return names, beta, se, r2, rmse, df


def pvalue_normal(t: float) -> float:
    return math.erfc(abs(t) / math.sqrt(2.0))


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def print_table(names: Sequence[str], beta: Sequence[float], se: Sequence[float], r2: float, rmse: float, df: int) -> None:
    print(f"OLS fixed-effects model: contribution ~ prompt + model")
    print(f"Baselines: prompt={BASE_PROMPT}, model={BASE_MODEL}")
    print(f"df={df}, R2={r2:.3f}, RMSE={rmse:.3f}")
    print("p-values use a normal approximation.")
    print()
    print(f"{'term':<34} {'coef':>8} {'se':>8} {'ci_low':>8} {'ci_high':>8} {'z':>8} {'p':>8}")
    print("-" * 88)
    for name, b, s in zip(names, beta, se):
        z = b / s if s else float("inf")
        p = pvalue_normal(z) if math.isfinite(z) else 0.0
        lo = b - 1.96 * s
        hi = b + 1.96 * s
        print(f"{name:<34} {b:8.2f} {s:8.2f} {lo:8.2f} {hi:8.2f} {z:8.2f} {fmt_p(p):>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight statistical tests for Exp. 2.1.")
    parser.add_argument("--input", type=Path, default=Path("outputs/exp2_1_h10_clean"))
    args = parser.parse_args()
    rows = load_rows(args.input)
    names, beta, se, r2, rmse, df = ols(rows)
    print_table(names, beta, se, r2, rmse, df)


if __name__ == "__main__":
    main()
