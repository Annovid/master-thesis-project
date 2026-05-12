"""Runner for Experiment 1.1 — one-shot PGG contribution distribution over temperature.

For each (model, temperature, repeat) cell, instantiates a fresh 4-player PGG
configured for a 10-round horizon, asks the LLM agent (Player 1) to think step
by step and end with 'Answer = N', then records the reply and parsed action.

Run from project root:
    python scripts/exp1_1_runner.py --dry-run
    python scripts/exp1_1_runner.py            # full grid
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from src.connectors.openrouter_connector import OpenRouterConnector
from src.gamesim.agents.llm_agent import LLMAgent
from src.gamesim.game.state import GameState
from src.gamesim.games.public_goods_game import PublicGoodsGame

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat-v3",
]
DEFAULT_TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
DEFAULT_REPEATS = 4
HORIZON_ROUNDS = 10
NUM_PLAYERS = 4
ENDOWMENT = 20.0
MULTIPLIER = 1.6
TRANSPARENCY = True


@dataclass
class CellResult:
    model: str
    temperature: float
    repeat: int
    parsed_action: float | None
    response: str
    prompt: str
    chat_id: str | None
    model_used: str | None
    usage: dict | None
    log_path: str | None
    reused: bool
    error: str | None
    elapsed_s: float


def run_one_cell(model: str, temperature: float, repeat: int) -> CellResult:
    """Run a single (model, temperature, repeat) cell and return its record."""
    t0 = time.time()
    game = PublicGoodsGame(
        num_players=NUM_PLAYERS,
        endowment=ENDOWMENT,
        multiplier=MULTIPLIER,
        transparency=TRANSPARENCY,
        reasoning=True,
    )
    state = GameState(max_rounds=HORIZON_ROUNDS)
    connector = OpenRouterConnector(model=model, temperature=temperature)
    agent = LLMAgent(
        name="Player1",
        connector=connector,
        temperature=temperature,
        reasoning=True,
    )
    try:
        action, details = agent.act(state, game, player_id=0)
        return CellResult(
            model=model,
            temperature=temperature,
            repeat=repeat,
            parsed_action=float(action) if action is not None else None,
            response=details.get("response", ""),
            prompt=details.get("prompt", ""),
            chat_id=details.get("chat_id"),
            model_used=details.get("model"),
            usage=details.get("usage"),
            log_path=details.get("log_path"),
            reused=bool(details.get("reused", False)),
            error=details.get("error"),
            elapsed_s=round(time.time() - t0, 3),
        )
    except Exception as exc:
        return CellResult(
            model=model,
            temperature=temperature,
            repeat=repeat,
            parsed_action=None,
            response="",
            prompt="",
            chat_id=None,
            model_used=None,
            usage=None,
            log_path=None,
            reused=False,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            elapsed_s=round(time.time() - t0, 3),
        )


def iter_cells(models: Iterable[str], temperatures: Iterable[float], repeats: int):
    for model in models:
        for temperature in temperatures:
            for repeat in range(1, repeats + 1):
                yield model, temperature, repeat


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 1.1 runner")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--temperatures", nargs="+", type=float, default=DEFAULT_TEMPERATURES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir. Default: artifacts/exp1_1/<timestamp>/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run only first 1 model x 2 temperatures x 1 repeat = 2 calls.")
    args = parser.parse_args()

    load_dotenv()

    if args.dry_run:
        models = args.models[:1]
        temperatures = args.temperatures[:2]
        repeats = 1
    else:
        models = list(args.models)
        temperatures = list(args.temperatures)
        repeats = args.repeats

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = "-dryrun" if args.dry_run else ""
    output_dir = Path(args.output_dir) if args.output_dir else Path("artifacts") / "exp1_1" / f"{ts}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.jsonl"
    manifest_path = output_dir / "manifest.json"

    cells = list(iter_cells(models, temperatures, repeats))
    total = len(cells)

    manifest = {
        "timestamp": ts,
        "dry_run": args.dry_run,
        "models": models,
        "temperatures": temperatures,
        "repeats": repeats,
        "horizon_rounds": HORIZON_ROUNDS,
        "num_players": NUM_PLAYERS,
        "endowment": ENDOWMENT,
        "multiplier": MULTIPLIER,
        "transparency": TRANSPARENCY,
        "reasoning": True,
        "total_planned_calls": total,
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"MANIFEST {manifest_path}")
    print(f"PLAN total_calls={total} models={len(models)} temperatures={len(temperatures)} repeats={repeats}")

    n_ok = 0
    n_err = 0
    n_reused = 0
    with results_path.open("w") as f:
        for idx, (model, temperature, repeat) in enumerate(cells, start=1):
            print(f"CELL {idx}/{total} model={model} T={temperature} repeat={repeat}")
            result = run_one_cell(model, temperature, repeat)
            f.write(json.dumps(asdict(result)) + "\n")
            f.flush()
            if result.error and not result.reused:
                n_err += 1
            elif result.reused:
                n_reused += 1
            else:
                n_ok += 1

    print(f"DONE results={results_path} ok={n_ok} reused={n_reused} err={n_err}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
