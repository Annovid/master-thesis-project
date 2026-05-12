#!/usr/bin/env python3
"""Collect all completed Experiment 1.2 sessions into a single flat file.

Writes two outputs:
  <output>.json  — full records (one row per player-round observation)
  <output>.csv   — same, comma-separated (ready for pandas / Excel)

Each row represents one player's action in one round of one session:
  model, session, round, player_id,
  contribution, payoff,
  parse_error, retries, transport_retries,
  elapsed_s, model_returned, usage_*,
  endowment, multiplier, transparency, reasoning, temperature

Only sessions with incomplete=False AND all rounds present are included.
Failed-attempt dirs are automatically skipped.

Usage:
    python scripts/collect_exp1_2.py
    python scripts/collect_exp1_2.py --input outputs/exp1_2_pilot --output outputs/exp1_2_collected
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


SKIP_DIR_PATTERNS = ("failed_attempt",)


def _iter_session_summaries(root: Path) -> List[Path]:
    paths = []
    for p in sorted(root.rglob("session_summary.json")):
        if any(pat in str(p) for pat in SKIP_DIR_PATTERNS):
            continue
        paths.append(p)
    return paths


def _flatten_session(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return one dict per (round, player) observation."""
    rows = []
    meta = {
        "model":        data["model_requested"],
        "session":      data["session"],
        "temperature":  data["temperature"],
        "endowment":    data["endowment"],
        "multiplier":   data["multiplier"],
        "transparency": data["transparency"],
        "reasoning":    data["reasoning"],
        "provider":     data["provider"],
    }

    for rnd in data.get("rounds", []):
        if rnd.get("incomplete"):
            continue
        round_num = rnd["round"]
        payoffs   = rnd.get("payoffs") or []

        for p in rnd.get("players", []):
            pid = p["player_id"]
            usage = p.get("usage") or {}
            rows.append({
                **meta,
                "round":              round_num,
                "player_id":          pid,
                "contribution":       p.get("parsed_action"),
                "payoff":             payoffs[pid - 1] if pid - 1 < len(payoffs) else None,
                "parse_error":        p.get("parse_error"),
                "retries":            p.get("retries", 0),
                "transport_retries":  p.get("transport_retries", 0),
                "elapsed_s":          p.get("elapsed_s"),
                "model_returned":     p.get("model_returned"),
                "usage_prompt_tokens":      usage.get("prompt_tokens"),
                "usage_completion_tokens":  usage.get("completion_tokens"),
                "usage_total_tokens":       usage.get("total_tokens"),
                "response_sha256":    p.get("response_sha256"),
            })
    return rows


CSV_COLUMNS = [
    "model", "session", "round", "player_id",
    "contribution", "payoff",
    "parse_error", "retries", "transport_retries",
    "elapsed_s", "model_returned",
    "usage_prompt_tokens", "usage_completion_tokens", "usage_total_tokens",
    "temperature", "endowment", "multiplier", "transparency", "reasoning",
    "provider", "response_sha256",
]


def collect(input_dir: Path, output_stem: Path) -> None:
    summaries = _iter_session_summaries(input_dir)
    if not summaries:
        print(f"No session_summary.json files found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    complete_sessions: List[Dict[str, Any]] = []
    skipped: List[str] = []
    all_rows: List[Dict[str, Any]] = []

    for sp in summaries:
        data = json.loads(sp.read_text())
        n_rounds = data.get("n_rounds", 10)
        rounds_done = len([r for r in data.get("rounds", []) if not r.get("incomplete")])

        if data.get("incomplete") or rounds_done < n_rounds:
            skipped.append(
                f"  SKIP  {sp.relative_to(input_dir)}  "
                f"(incomplete={data.get('incomplete')}  rounds={rounds_done}/{n_rounds})"
            )
            continue

        complete_sessions.append(data)
        all_rows.extend(_flatten_session(data))

    # ── print summary ────────────────────────────────────────────────────────
    print(f"Found {len(summaries)} session(s), {len(complete_sessions)} complete, {len(skipped)} skipped.")
    if skipped:
        print("\nSkipped:")
        for s in skipped:
            print(s)

    if not all_rows:
        print("Nothing to write.", file=sys.stderr)
        sys.exit(1)

    # ── write JSON ───────────────────────────────────────────────────────────
    json_path = output_stem.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(input_dir),
        "n_sessions": len(complete_sessions),
        "n_rows": len(all_rows),
        "sessions": complete_sessions,   # full session records
        "observations": all_rows,        # flat player-round rows
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote {json_path}  ({len(all_rows)} observations)")

    # ── write CSV ────────────────────────────────────────────────────────────
    csv_path = output_stem.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {csv_path}  ({len(all_rows)} rows × {len(CSV_COLUMNS)} columns)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect completed Exp 1.2 sessions into one file.")
    parser.add_argument(
        "--input", type=Path, default=Path("outputs/exp1_2_pilot"),
        help="Root experiment output directory (default: outputs/exp1_2_pilot)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/exp1_2_collected"),
        help="Output path stem (without extension). Both .json and .csv are written. "
             "(default: outputs/exp1_2_collected)",
    )
    args = parser.parse_args()
    collect(args.input, args.output)


if __name__ == "__main__":
    main()
