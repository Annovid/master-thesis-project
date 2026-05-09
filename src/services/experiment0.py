"""Experiment 0 — determinism of LLM in one-shot Public Goods Game at T=0.

Grid: model x horizon x temperature x n_samples.
Per design (docs/06/3. Дизайн эксперимента.md), horizon varies between
{1 round, 10 rounds} and T=0; n=10 per cell.

Output layout:
  outputs/exp0/<model_safe>/horizon_<h>/temp_<t>/sample_<i>.txt
  outputs/exp0/run_summary.json
  outputs/exp0/<model_safe>/horizon_<h>/prompt.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

from src.gamesim.games.public_goods_game import PublicGoodsGame
from src.gamesim.game.state import GameState
from src.services.single_request import build_connector


@dataclass
class RunRecord:
    model: str
    provider: str
    horizon: int
    temperature: float
    sample_index: int
    response_path: str
    response_sha256: str
    parsed_action: float | None
    parse_error: str | None
    meta: Dict[str, Any]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def build_prompt(game: PublicGoodsGame, horizon: int) -> str:
    state = GameState(max_rounds=horizon)
    return game.get_prompt(state=state, player_id=0, max_rounds=horizon)


def safe_parse(game: PublicGoodsGame, response: str) -> tuple[float | None, str | None]:
    try:
        return float(game.parse_action(response)), None
    except ValueError as e:
        return None, str(e)


def _save_summary(output_dir: Path, config_snapshot: Dict[str, Any], runs: List[RunRecord]) -> None:
    """Merge new runs into existing run_summary.json (dedup by response_path)."""
    summary_path = output_dir / "run_summary.json"
    existing: List[Dict[str, Any]] = []
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text()).get("runs", [])
        except Exception:
            existing = []

    new_dicts = [asdict(r) for r in runs]
    new_paths = {d["response_path"] for d in new_dicts}
    kept = [r for r in existing if r.get("response_path") not in new_paths]
    merged = kept + new_dicts
    merged.sort(key=lambda r: (r.get("model", ""), r.get("horizon", 0), r.get("temperature", 0), r.get("sample_index", 0)))

    summary = {**config_snapshot, "runs": merged}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def run_experiment(
    models: List[str],
    provider: str,
    horizons: List[int],
    temperatures: List[float],
    samples_per_cell: int,
    num_players: int,
    endowment: float,
    multiplier: float,
    output_dir: Path,
    start_sample: int = 0,
) -> Dict[str, Any]:
    load_dotenv()

    output_dir.mkdir(parents=True, exist_ok=True)

    game = PublicGoodsGame(
        num_players=num_players,
        endowment=endowment,
        multiplier=multiplier,
        transparency=False,
    )

    config_snapshot = {
        "models": models,
        "provider": provider,
        "horizons": horizons,
        "temperatures": temperatures,
        "samples_per_cell": samples_per_cell,
        "start_sample": start_sample,
        "num_players": num_players,
        "endowment": endowment,
        "multiplier": multiplier,
        "env_provider": os.getenv("LLM_PROVIDER"),
        "openrouter_max_tokens": os.getenv("OPENROUTER_MAX_TOKENS"),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config_snapshot, indent=2, ensure_ascii=False)
    )

    runs: List[RunRecord] = []

    for model in models:
        model_dir = output_dir / safe_model_name(model)
        model_dir.mkdir(parents=True, exist_ok=True)

        for horizon in horizons:
            prompt = build_prompt(game, horizon)
            horizon_dir = model_dir / f"horizon_{horizon}"
            horizon_dir.mkdir(parents=True, exist_ok=True)
            (horizon_dir / "prompt.txt").write_text(prompt)

            for temperature in temperatures:
                connector = build_connector(
                    {"model": model, "temperature": temperature},
                    default_provider=provider,
                )
                temp_dir = horizon_dir / f"temp_{temperature}"
                temp_dir.mkdir(parents=True, exist_ok=True)

                for sample_idx in range(start_sample, start_sample + samples_per_cell):
                    try:
                        response, meta = connector.query(prompt)
                    except Exception as e:
                        print(
                            f"FAIL model={model} horizon={horizon} temp={temperature} "
                            f"sample={sample_idx} err={e!r}"
                        )
                        _save_summary(output_dir, config_snapshot, runs)
                        continue

                    parsed, parse_err = safe_parse(game, response)
                    resp_sha = sha256_text(response)
                    actual_model = (
                        meta.get("model") if isinstance(meta, dict) else model
                    ) or model

                    response_file = temp_dir / f"sample_{sample_idx}.txt"
                    with response_file.open("w") as f:
                        f.write(f"MODEL_REQUESTED: {model}\n")
                        f.write(f"MODEL_RETURNED: {actual_model}\n")
                        f.write(f"PROVIDER: {provider}\n")
                        f.write(f"HORIZON: {horizon}\n")
                        f.write(f"TEMPERATURE: {temperature}\n")
                        f.write(f"SAMPLE: {sample_idx}\n")
                        f.write(f"PARSED_ACTION: {parsed}\n")
                        f.write(f"PARSE_ERROR: {parse_err}\n")
                        f.write(f"RESPONSE_SHA256: {resp_sha}\n\n")
                        f.write("PROMPT:\n")
                        f.write(prompt)
                        f.write("\n\nRESPONSE:\n")
                        f.write(response)
                        f.write("\n")

                    runs.append(
                        RunRecord(
                            model=actual_model,
                            provider=provider,
                            horizon=horizon,
                            temperature=temperature,
                            sample_index=sample_idx,
                            response_path=str(response_file.relative_to(output_dir)),
                            response_sha256=resp_sha,
                            parsed_action=parsed,
                            parse_error=parse_err,
                            meta=meta if isinstance(meta, dict) else {},
                        )
                    )

                    print(
                        f"OK model={model} horizon={horizon} temp={temperature} "
                        f"sample={sample_idx} parsed={parsed} sha={resp_sha[:8]}"
                    )
                    _save_summary(output_dir, config_snapshot, runs)

    _save_summary(output_dir, config_snapshot, runs)
    return {**config_snapshot, "runs": [asdict(r) for r in runs]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 0: PGG one-shot determinism across model x horizon x T."
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=["gpt-4o-mini"],
        help="One or more model ids. With provider=openrouter use '<vendor>/<model>'.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=os.getenv("LLM_PROVIDER", "auto"),
        help="auto|openai|openrouter|gateway|mock",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0])
    parser.add_argument("--samples-per-cell", type=int, default=10)
    parser.add_argument(
        "--start-sample", type=int, default=0,
        help="Starting sample index. Use to top up missing samples without rerunning earlier ones.",
    )
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--endowment", type=float, default=20.0)
    parser.add_argument("--multiplier", type=float, default=1.6)
    parser.add_argument("--output", type=Path, default=Path("outputs/exp0"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        models=args.models,
        provider=args.provider,
        horizons=args.horizons,
        temperatures=args.temperatures,
        samples_per_cell=args.samples_per_cell,
        num_players=args.num_players,
        endowment=args.endowment,
        multiplier=args.multiplier,
        output_dir=args.output,
        start_sample=args.start_sample,
    )
