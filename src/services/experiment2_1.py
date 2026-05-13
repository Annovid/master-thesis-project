"""Experiment 2.1 — Prompt-condition sweep (one-shot PGG).

Grid: models × prompt_conditions, 1 session per cell, n_rounds=1, T=0.
Goal: find which prompt framing produces contributions closest to the
human baseline of 40-60 % of endowment (8-12 out of 20 tokens).

Output layout:
  outputs/exp2_1/
    run_config.json
    run_summary.json          # all session summaries
    <model_safe>/<prompt_label>/
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


PROMPT_CONDITIONS: Dict[str, str] = {
    "neutral": "",
    "persona": (
        "Imagine you are a typical human participant in a behavioral economics experiment. "
        "Make your decision as a real person would, considering both your interests and what "
        "you expect from others."
    ),
    "inequality_aversion": (
        "In addition to your own payoff, consider fairness among players. "
        "Large inequalities in contributions are unfair. "
        "Avoid free-riding on others' contributions."
    ),
    "conditional_coop": (
        "Your contribution should reflect what you expect others to contribute. "
        "If you expect others to contribute generously, match their effort. "
        "If you expect free-riding, reduce your own contribution rather than being exploited."
    ),
    "expectations": (
        "Before choosing, explicitly reason about how much each other player is likely to "
        "contribute. Then choose a contribution consistent with those expectations."
    ),
    "social_norms": (
        "Note: in studies with similar games, participants typically contribute 40-60% of "
        "their endowment (about 8-12 tokens out of 20). "
        "Contributing significantly less than others is considered unfair."
    ),
    "emotional": (
        "You experience real emotions in this game. Seeing others free-ride while you "
        "contribute feels genuinely unfair and frustrating. Let your sense of fairness and "
        "emotional reactions guide your decision."
    ),
}


def _save_grid_summary(output_dir: Path, config: Dict[str, Any], sessions: List[Dict[str, Any]]) -> None:
    path = output_dir / "run_summary.json"
    payload = {**config, "sessions": sessions}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def run_grid(
    models: List[str],
    provider: str,
    num_players: int,
    endowment: float,
    multiplier: float,
    transparency: bool,
    reasoning: bool,
    output_dir: Path,
    prompt_labels: List[str] | None = None,
) -> None:
    load_dotenv()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = prompt_labels or list(PROMPT_CONDITIONS.keys())

    config = {
        "experiment": "2.1",
        "models": models,
        "provider": provider,
        "temperature": 0.0,
        "n_rounds": 1,
        "num_players": num_players,
        "endowment": endowment,
        "multiplier": multiplier,
        "transparency": transparency,
        "reasoning": reasoning,
        "prompt_labels": labels,
        "env_provider": os.getenv("LLM_PROVIDER"),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    all_sessions: List[Dict[str, Any]] = []
    total = len(models) * len(labels)
    done = 0

    for model in models:
        for label in labels:
            condition_text = PROMPT_CONDITIONS[label]
            cell_dir = output_dir / safe_model_name(model) / label
            cell_dir.mkdir(parents=True, exist_ok=True)

            done += 1
            print(f"\n[{done}/{total}] model={model}  prompt={label}")
            t0 = time.time()

            summary = run_session(
                model=model,
                provider=provider,
                temperature=0.0,
                session_idx=0,
                num_players=num_players,
                endowment=endowment,
                multiplier=multiplier,
                n_rounds=1,
                transparency=transparency,
                reasoning=reasoning,
                output_dir=cell_dir,
                prompt_label=label,
                prompt_condition=condition_text,
            )
            summary["cell_elapsed_s"] = round(time.time() - t0, 2)
            all_sessions.append(summary)
            _save_grid_summary(output_dir, config, all_sessions)

            contribution = None
            if summary.get("rounds"):
                actions = summary["rounds"][0].get("actions", [])
                if actions:
                    contribution = actions[0]
            pct = f"{contribution / endowment * 100:.0f}%" if contribution is not None else "?"
            print(f"  contribution={contribution}  ({pct} of endowment)  elapsed={summary['cell_elapsed_s']:.1f}s")

    print(f"\nDone. {len(all_sessions)} cells. Output: {output_dir}")
    _print_table(all_sessions, models, labels, endowment)


def _print_table(
    sessions: List[Dict[str, Any]],
    models: List[str],
    labels: List[str],
    endowment: float,
) -> None:
    contrib: Dict[tuple, Any] = {}
    for s in sessions:
        m = s["model_requested"]
        lbl = s.get("prompt_label", "neutral")
        if s.get("rounds"):
            actions = s["rounds"][0].get("actions", [])
            contrib[(m, lbl)] = actions[0] if actions else None

    col_w = 14
    header = f"{'model':<28}" + "".join(f"{lbl[:col_w]:>{col_w}}" for lbl in labels)
    print("\n" + header)
    print("-" * len(header))
    for m in models:
        short = m.split("/")[-1][:26]
        row = f"{short:<28}"
        for lbl in labels:
            val = contrib.get((m, lbl))
            cell = f"{val:.0f} ({val/endowment*100:.0f}%)" if val is not None else "?"
            row += f"{cell:>{col_w}}"
        print(row)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2.1: one-shot PGG prompt-condition sweep."
    )
    parser.add_argument("--models", type=str, nargs="+", required=True,
        help="Model IDs to test (e.g. openai/gpt-4o meta-llama/llama-4-maverick).")
    parser.add_argument("--provider", type=str, default=os.getenv("LLM_PROVIDER", "auto"))
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--endowment", type=float, default=20.0)
    parser.add_argument("--multiplier", type=float, default=1.6)
    parser.add_argument("--no-transparency", action="store_true")
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--prompt-labels", type=str, nargs="+", default=None,
        help=f"Subset of prompt labels to run. Available: {list(PROMPT_CONDITIONS)}.")
    parser.add_argument("--output", type=Path, default=Path("outputs/exp2_1"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_grid(
        models=args.models,
        provider=args.provider,
        num_players=args.num_players,
        endowment=args.endowment,
        multiplier=args.multiplier,
        transparency=not args.no_transparency,
        reasoning=not args.no_reasoning,
        output_dir=args.output,
        prompt_labels=args.prompt_labels,
    )
