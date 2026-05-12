"""Orchestrator for the remaining exp 1.1 probes (Opus 4.7, Gemini 2.5 Pro,
GPT-5.5-pro). Implements an early-termination guard:

For each model, first probe T_max with n=2. If both samples fail to parse
(2 consecutive parse errors), skip the rest of that model's grid — do NOT
spend budget on lower temperatures.

Otherwise, top up T_max to n=10 and run the lower temperatures at n=10.

Output: outputs/exp1_1/<model_safe>/horizon_10/temp_<T>/sample_<i>.txt
        outputs/exp1_1/run_summary.json   (merged across all models via
                                           experiment0._save_summary)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

from src.services.experiment0 import run_experiment


# (model, T_max, [non-zero T grid in ascending order], OPENROUTER_MAX_TOKENS)
MODELS: List[Tuple[str, float, List[float], str]] = [
    ("anthropic/claude-opus-4.7", 1.0, [0.3, 0.7, 1.0], "16000"),
    ("google/gemini-2.5-pro", 2.0, [0.6, 1.4, 2.0], "16000"),
    ("openai/gpt-5.5-pro", 2.0, [0.6, 1.4, 2.0], "4000"),
]

OUT_DIR = Path("outputs/exp1_1")


def _consecutive_failures_at(summary_path: Path, model_substr: str, t: float) -> int:
    """Return the longest run of consecutive parse errors at samples 0..n-1
    for (model containing model_substr, h=10, temperature=t)."""
    if not summary_path.exists():
        return 0
    runs = json.loads(summary_path.read_text())["runs"]
    cell = sorted(
        [
            r for r in runs
            if model_substr.split("/")[-1].split("-20")[0] in r["model"]
            and r["horizon"] == 10
            and r["temperature"] == t
        ],
        key=lambda r: r["sample_index"],
    )
    consec = 0
    best = 0
    for r in cell:
        if r["parsed_action"] is None:
            consec += 1
            best = max(best, consec)
        else:
            consec = 0
    return best


def main() -> None:
    load_dotenv()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "run_summary.json"

    skipped = []
    completed = []

    for model, t_max, ts, max_tok in MODELS:
        os.environ["OPENROUTER_MAX_TOKENS"] = max_tok
        print(f"\n========== {model}  T_max={t_max}  max_tokens={max_tok} ==========")

        # 1) Probe T_max with n=2
        print(f"  [probe] T={t_max}, n=2 ...")
        run_experiment(
            models=[model], provider="openrouter",
            horizons=[10], temperatures=[t_max],
            samples_per_cell=2, num_players=4,
            endowment=20.0, multiplier=1.6,
            output_dir=OUT_DIR, start_sample=0,
        )

        consec = _consecutive_failures_at(summary_path, model, t_max)
        if consec >= 2:
            print(f"  [SKIP] 2+ consecutive parse errors at T={t_max} → not running lower T for this model")
            skipped.append(model)
            continue

        # 2) Top up T_max to n=10 (samples 2..9)
        print(f"  [topup] T={t_max}, samples 2..9 ...")
        run_experiment(
            models=[model], provider="openrouter",
            horizons=[10], temperatures=[t_max],
            samples_per_cell=8, num_players=4,
            endowment=20.0, multiplier=1.6,
            output_dir=OUT_DIR, start_sample=2,
        )

        # 3) Lower temperatures at n=10
        lower = [t for t in ts if t != t_max]
        if lower:
            print(f"  [grid] T={lower}, n=10 ...")
            run_experiment(
                models=[model], provider="openrouter",
                horizons=[10], temperatures=lower,
                samples_per_cell=10, num_players=4,
                endowment=20.0, multiplier=1.6,
                output_dir=OUT_DIR, start_sample=0,
            )
        completed.append(model)

    print("\n========== SUMMARY ==========")
    for m in completed:
        print(f"  COMPLETED: {m}")
    for m in skipped:
        print(f"  SKIPPED:   {m}  (T_max parse-fail)")
    print(f"\nMerged summary: {summary_path}")


if __name__ == "__main__":
    main()
