"""Estimate realized cost of one or more experiment runs.

Pulls per-model pricing from OpenRouter (`/api/v1/models`) and reads token
counts from `meta.usage` in run_summary.json files. Sums input / output /
reasoning costs per model and reports a per-model + grand total.

Pricing is cached in OpenRouter as USD per token. We multiply by the actual
prompt_tokens / completion_tokens / completion_tokens_details.reasoning_tokens
returned by each call.

Usage:
    venv/bin/python -m src.services.cost_estimate \
        --summary outputs/exp0_iter1/run_summary.json \
        --summary outputs/exp1_1_gpt4o_probe/run_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests
from dotenv import load_dotenv


# OpenRouter returns dated upstream ids in `meta.model` (e.g.
# `anthropic/claude-4.7-opus-20260416`) while the pricing endpoint keys by
# stable ids (e.g. `anthropic/claude-opus-4.7`). This map covers the models
# we have used so far. Add new entries as needed.
MODEL_ID_REMAP: Dict[str, str] = {
    "anthropic/claude-4.7-opus-20260416": "anthropic/claude-opus-4.7",
    "openai/gpt-5.5-pro-20260423": "openai/gpt-5.5-pro",
    "openai/gpt-4o-2024-08-06": "openai/gpt-4o",
    "openai/gpt-4o-2024-11-20": "openai/gpt-4o",
    "openai/gpt-4o-2024-05-13": "openai/gpt-4o",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "meta-llama/llama-4-maverick",
}


def fetch_prices() -> Dict[str, Dict[str, Any]]:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    r = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    r.raise_for_status()
    return {m["id"]: m.get("pricing") or {} for m in r.json()["data"]}


def _to_float(x: Any) -> float:
    try:
        return float(x or 0)
    except (TypeError, ValueError):
        return 0.0


def cell_cost(pricing: Dict[str, Any], usage: Dict[str, Any]) -> Tuple[float, float, float, int, int, int]:
    """Return (input_cost, output_cost, reasoning_cost, pt, ct, rt)."""
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    rt = int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)

    in_rate = _to_float(pricing.get("prompt"))
    out_rate = _to_float(pricing.get("completion"))
    reas_rate = _to_float(pricing.get("internal_reasoning"))

    in_cost = pt * in_rate
    if reas_rate > 0:
        # Reasoning tokens billed at a separate rate (e.g. Gemini); strip them
        # from the completion portion to avoid double-counting.
        out_cost = max(ct - rt, 0) * out_rate
        reas_cost = rt * reas_rate
    else:
        # Reasoning tokens, where applicable (e.g. gpt-5.5-pro), are typically
        # included in completion_tokens at the completion rate.
        out_cost = ct * out_rate
        reas_cost = 0.0
    return in_cost, out_cost, reas_cost, pt, ct, rt


def aggregate(summaries: Iterable[Path], prices: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    totals: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "in_t": 0, "out_t": 0, "reas_t": 0,
                 "in_$": 0.0, "out_$": 0.0, "reas_$": 0.0, "missing_price": False}
    )
    for path in summaries:
        if not path.exists():
            print(f"WARN: missing {path}")
            continue
        runs = json.loads(path.read_text())["runs"]
        for r in runs:
            usage = (r.get("meta") or {}).get("usage") or {}
            if not usage:
                continue
            mid = r["model"]
            key = MODEL_ID_REMAP.get(mid, mid)
            pricing = prices.get(key, {})
            in_c, out_c, reas_c, pt, ct, rt = cell_cost(pricing, usage)
            t = totals[key]
            t["calls"] += 1
            t["in_t"] += pt
            t["out_t"] += ct
            t["reas_t"] += rt
            t["in_$"] += in_c
            t["out_$"] += out_c
            t["reas_$"] += reas_c
            if not pricing:
                t["missing_price"] = True
    return totals


def print_table(totals: Dict[str, Dict[str, Any]]) -> None:
    header = (
        f"{'model':<34} {'calls':>5} {'in_tok':>8} {'out_tok':>8} {'reas_tok':>9} "
        f"{'$ in':>8} {'$ out':>8} {'$ reas':>8} {'$ total':>9}"
    )
    print(header)
    print("-" * len(header))
    grand = 0.0
    for k in sorted(totals):
        v = totals[k]
        total = v["in_$"] + v["out_$"] + v["reas_$"]
        grand += total
        flag = "  (no price!)" if v["missing_price"] else ""
        print(
            f"{k:<34} {v['calls']:>5} {v['in_t']:>8} {v['out_t']:>8} {v['reas_t']:>9} "
            f"${v['in_$']:>7.3f} ${v['out_$']:>7.3f} ${v['reas_$']:>7.3f} ${total:>8.3f}{flag}"
        )
    print("-" * len(header))
    print(
        f"{'GRAND TOTAL':<34} {sum(v['calls'] for v in totals.values()):>5} "
        f"{'':>8} {'':>8} {'':>9} {'':>8} {'':>8} {'':>8} ${grand:>8.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate realized cost of experiment runs.")
    parser.add_argument(
        "--summary", action="append", type=Path, required=True,
        help="Path to run_summary.json. Repeatable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prices = fetch_prices()
    totals = aggregate(args.summary, prices)
    print_table(totals)
