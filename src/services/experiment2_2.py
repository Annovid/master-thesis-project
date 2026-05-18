"""Experiment 2.2 — Endowment heterogeneity sweep (one-shot PGG).

Motivation: Exp 2.1 showed that all social prompt framings collapse to the
same focal answer (g=10, 50% of endowment=20) because equal endowments make
equal-absolute, equal-relative, and self-interest norms indistinguishable.
Heterogeneous endowments break this: a rich player contributing 10 tokens is
50% of 20 but only 12.5% of 80 — different norms now prescribe different
numbers, and the model must genuinely choose.

Grid: 4 models × 6 endowment profiles × 2 prompt conditions = 48 cells.
Fixed T=0, n_rounds=1.

Prompt conditions used (best from Exp 2.1):
  - neutral   : no extra framing  (baseline)
  - persona   : "Imagine you are a typical human participant ..."

Endowment profiles:
  equal_20    : [20, 20, 20, 20]   — Exp 2.1 baseline
  equal_50    : [50, 50, 50, 50]   — scale check
  equal_80    : [80, 80, 80, 80]   — rich equal
  mild_ineq   : [20, 20, 50, 50]   — 2 poor / 2 rich
  classic_ineq: [20, 50, 80, 80]   — Hargreaves Heap / Sreedhar profile
  strong_ineq : [10, 20, 80, 160]  — breaks 50% focal point

Key observation per cell: g_i and g_i/E_i per player role (poor/mid/rich).
With heterogeneous endowments the 4 players in a session get different prompts
even at T=0, giving 4 distinct observations per cell.

Output layout:
  outputs/exp2_2/
    run_config.json
    run_summary.json
    <model_safe>/<profile_label>/<prompt_label>/
      session_0/
        session_summary.json
        round_1/
          player_<p>.txt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from src.services.experiment1_2 import run_session, safe_model_name
from src.services.experiment2_1 import PROMPT_CONDITIONS


ENDOWMENT_PROFILES: Dict[str, List[float]] = {
    "equal_20":     [20, 20, 20, 20],
    "equal_50":     [50, 50, 50, 50],
    "equal_80":     [80, 80, 80, 80],
    "mild_ineq":    [20, 20, 50, 50],
    "classic_ineq": [20, 50, 80, 80],
    "strong_ineq":  [10, 20, 80, 160],
}

DEFAULT_PROMPT_LABELS = ["neutral", "persona"]


def _save_summary(output_dir: Path, config: Dict[str, Any], sessions: List[Dict[str, Any]]) -> None:
    path = output_dir / "run_summary.json"
    path.write_text(json.dumps({**config, "sessions": sessions}, indent=2, ensure_ascii=False))


def run_grid(
    models: List[str],
    provider: str,
    multiplier: float,
    reasoning: bool,
    output_dir: Path,
    profile_labels: List[str] | None = None,
    prompt_labels: List[str] | None = None,
) -> None:
    load_dotenv()
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = profile_labels or list(ENDOWMENT_PROFILES.keys())
    prompts = prompt_labels or DEFAULT_PROMPT_LABELS

    config = {
        "experiment": "2.2",
        "models": models,
        "provider": provider,
        "temperature": 0.0,
        "n_rounds": 1,
        "multiplier": multiplier,
        "reasoning": reasoning,
        "profile_labels": profiles,
        "prompt_labels": prompts,
        "endowment_profiles": {k: ENDOWMENT_PROFILES[k] for k in profiles},
        "env_provider": os.getenv("LLM_PROVIDER"),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    all_sessions: List[Dict[str, Any]] = []
    total = len(models) * len(profiles) * len(prompts)
    done = 0

    for model in models:
        for profile_label in profiles:
            endowments = ENDOWMENT_PROFILES[profile_label]
            num_players = len(endowments)
            for prompt_label in prompts:
                condition_text = PROMPT_CONDITIONS.get(prompt_label, "")
                cell_dir = output_dir / safe_model_name(model) / profile_label / prompt_label
                cell_dir.mkdir(parents=True, exist_ok=True)

                done += 1
                print(f"\n[{done}/{total}] model={model}  profile={profile_label}  prompt={prompt_label}")
                print(f"  endowments={endowments}")
                t0 = time.time()

                summary = run_session(
                    model=model,
                    provider=provider,
                    temperature=0.0,
                    session_idx=0,
                    num_players=num_players,
                    endowment=endowments[0],
                    multiplier=multiplier,
                    n_rounds=1,
                    transparency=True,
                    reasoning=reasoning,
                    output_dir=cell_dir,
                    prompt_label=prompt_label,
                    prompt_condition=condition_text,
                    endowments=endowments,
                )
                summary["profile_label"] = profile_label
                summary["cell_elapsed_s"] = round(time.time() - t0, 2)
                all_sessions.append(summary)
                _save_summary(output_dir, config, all_sessions)

                if summary.get("rounds") and not summary.get("incomplete"):
                    actions = summary["rounds"][0].get("actions", [])
                    shares = [
                        f"P{i+1}:{a:.0f}/{endowments[i]:.0f}({a/endowments[i]*100:.0f}%)"
                        for i, a in enumerate(actions)
                    ]
                    print(f"  contributions: {', '.join(shares)}")
                else:
                    print(f"  INCOMPLETE: {summary.get('abort_reason','?')}")

    print(f"\nDone. {len(all_sessions)} cells. Output: {output_dir}")
    _print_table(all_sessions, models, profiles, prompts)


def _print_table(
    sessions: List[Dict[str, Any]],
    models: List[str],
    profiles: List[str],
    prompts: List[str],
) -> None:
    # Index (model, profile, prompt) -> list of (endowment, contribution) per player
    data: Dict[tuple, Any] = {}
    for s in sessions:
        m = s["model_requested"]
        profile = s.get("profile_label", "?")
        prompt = s.get("prompt_label", "neutral")
        endows = ENDOWMENT_PROFILES.get(profile, [])
        if s.get("rounds") and not s.get("incomplete"):
            actions = s["rounds"][0].get("actions", [])
            data[(m, profile, prompt)] = list(zip(endows, actions))

    for prompt_label in prompts:
        print(f"\n--- Prompt: {prompt_label} ---")
        col_w = 22
        model_w = 28
        header = f"{'Model':<{model_w}}" + "".join(f"{p[:col_w-1]:>{col_w}}" for p in profiles)
        print(header)
        print("─" * len(header))
        for m in models:
            short = m.split("/")[-1][:model_w - 2]
            row = f"{short:<{model_w}}"
            for profile in profiles:
                pairs = data.get((m, profile, prompt_label))
                if pairs is None:
                    cell = "·"
                else:
                    parts = [f"{a:.0f}/{e:.0f}({a/e*100:.0f}%)" for e, a in pairs]
                    cell = " ".join(parts)
                    if len(cell) > col_w - 1:
                        avg_share = sum(a / e for e, a in pairs) / len(pairs) * 100
                        cell = f"avg={avg_share:.0f}%"
                row += f"{cell:>{col_w}}"
            print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2.2: endowment heterogeneity sweep (one-shot PGG)."
    )
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--provider", type=str, default=os.getenv("LLM_PROVIDER", "auto"))
    parser.add_argument("--multiplier", type=float, default=1.6)
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--profile-labels", type=str, nargs="+", default=None,
        help=f"Subset of profiles. Available: {list(ENDOWMENT_PROFILES)}.")
    parser.add_argument("--prompt-labels", type=str, nargs="+", default=None,
        help="Subset of prompt labels (default: neutral persona).")
    parser.add_argument("--output", type=Path, default=Path("outputs/exp2_2"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_grid(
        models=args.models,
        provider=args.provider,
        multiplier=args.multiplier,
        reasoning=not args.no_reasoning,
        output_dir=args.output,
        profile_labels=args.profile_labels,
        prompt_labels=args.prompt_labels,
    )
