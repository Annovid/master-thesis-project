"""Experiment 2.1 — Prompt-condition sweep for the first PGG decision.

Grid: models × prompt_conditions × samples, one first-round decision per call.

Default settings intentionally match the Exp. 1.1 neutral prompt shape:
horizon=10, T=0, transparency=False, reasoning=True. This makes the neutral
condition a direct baseline for the prompt-condition cells.

Output layout:
  outputs/exp2_1_h10/
    run_config.json
    run_summary.json
    <model_safe>/<prompt_label>/horizon_10/prompt.txt
    <model_safe>/<prompt_label>/horizon_10/temp_0.0/sample_<i>.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from src.gamesim.game.state import GameState
from src.gamesim.games.public_goods_game import PublicGoodsGame
from src.services.single_request import build_connector


PROMPT_CONDITIONS: Dict[str, str] = {
    "neutral": "",
    "persona": (
        "Imagine you are a typical human participant in a behavioral economics experiment. "
        "Make the decision from that perspective."
    ),
    "self_interest": (
        "Focus on maximizing your own monetary payoff in this game."
    ),
    "group_welfare": (
        "Focus on maximizing the total monetary payoff of all players as a group."
    ),
    "inequality_aversion": (
        "Consider whether your decision would lead to unequal outcomes across players."
    ),
    "warm_glow": (
        "Consider that contributing to the public pot may feel personally worthwhile "
        "because it helps the group."
    ),
    "conditional_coop": (
        "You may condition your contribution on what you expect the other players to contribute."
    ),
    "expectations": (
        "Before choosing, explicitly reason about how much each other player is likely to "
        "contribute."
    ),
    "social_norms": (
        "Note: in studies with similar public goods games, participants often contribute "
        "40-60% of their endowment."
    ),
    "emotional_perspective": (
        "Imagine a typical participant who may feel frustrated by free-riding and may feel "
        "good about fair cooperation. Make the decision from that perspective."
    ),
}


@dataclass
class RunRecord:
    model: str
    model_requested: str
    provider: str
    horizon: int
    temperature: float
    prompt_label: str
    prompt_condition: str
    sample_index: int
    response_path: str
    response_sha256: str
    parsed_action: float | None
    parse_error: str | None
    meta: Dict[str, Any]


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prompt(game: PublicGoodsGame, horizon: int) -> str:
    """Build the Exp. 1.1-compatible first-decision prompt.

    Keep this local to Exp. 2.1 so later changes to PublicGoodsGame formatting
    do not silently break the neutral baseline comparison.
    """
    state = GameState(max_rounds=horizon)
    history_str = game._format_history(state, player_id=0)
    rounds_word = "round" if horizon == 1 else "rounds"
    endowment = str(float(game.endowment))

    lines = [
        f"You are playing the Public Goods Game for {horizon} {rounds_word} with {game.num_players} players in total.",
        "You are Player 1.",
        "",
        f"Each round, every player receives an endowment of {endowment} tokens and independently chooses how many tokens to contribute to a public pot (any amount from 0 to {endowment}).",
        f"The sum of all contributions is multiplied by {game.multiplier} and divided equally among all {game.num_players} players.",
        f"Your payoff for a round = endowment - your_contribution + ({game.multiplier} * sum_of_all_contributions) / {game.num_players}.",
        f"Your final score is the sum of your payoffs across all {horizon} {rounds_word}.",
    ]

    if game.transparency:
        lines.append("After each round, the individual contributions of all players are revealed to everyone.")
    else:
        lines.append("Between rounds, only your own contribution and round payoff are revealed to you.")

    lines += [
        "",
        "History of previous rounds:",
        history_str,
        "",
    ]

    if game.prompt_condition:
        lines += [game.prompt_condition, ""]

    lines.append("Decide your contribution for the current round.")
    if game.reasoning:
        lines.append("End your reply with a single line of the form: Answer = N")
        lines.append(f"where N is a number between 0 and {endowment}.")
    else:
        lines.append(
            f"Reply with only a single number between 0 and {endowment}. No explanations, no extra text, no labels."
        )
    return "\n".join(lines)


def safe_parse(game: PublicGoodsGame, response: str) -> tuple[float | None, str | None]:
    try:
        return float(game.parse_action(response)), None
    except ValueError as e:
        return None, str(e)


def _save_grid_summary(output_dir: Path, config: Dict[str, Any], runs: List[RunRecord]) -> None:
    path = output_dir / "run_summary.json"
    existing: List[Dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text()).get("runs", [])
        except Exception:
            existing = []

    new_dicts = [asdict(r) for r in runs]
    new_paths = {r["response_path"] for r in new_dicts}
    kept = [r for r in existing if r.get("response_path") not in new_paths]
    merged = kept + new_dicts
    merged.sort(
        key=lambda r: (
            r.get("model_requested", ""),
            r.get("prompt_label", ""),
            r.get("horizon", 0),
            r.get("temperature", 0),
            r.get("sample_index", 0),
        )
    )
    payload = {**config, "runs": merged}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_summary_runs(output_dir: Path) -> List[Dict[str, Any]]:
    path = output_dir / "run_summary.json"
    if not path.exists():
        return []
    try:
        runs = json.loads(path.read_text()).get("runs", [])
    except Exception:
        return []
    return runs if isinstance(runs, list) else []


def _run_sample(
    *,
    model: str,
    provider: str,
    temperature: float,
    prompt: str,
    game: PublicGoodsGame,
    horizon: int,
    label: str,
    condition_text: str,
    sample_idx: int,
    output_dir: Path,
    temp_dir: Path,
    max_transport_retries: int,
    transport_backoff_s: float,
) -> RunRecord:
    connector = build_connector(
        {"model": model, "temperature": temperature},
        default_provider=provider,
    )
    transport_retries = 0
    while True:
        try:
            response, meta = connector.query(prompt)
            break
        except Exception as exc:
            if transport_retries >= max_transport_retries:
                raise
            wait = transport_backoff_s * (2 ** transport_retries)
            print(
                f"TRANSPORT_RETRY model={model} prompt={label} sample={sample_idx} "
                f"attempt={transport_retries + 1}/{max_transport_retries} "
                f"wait={wait:.1f}s err={type(exc).__name__}: {exc}"
            )
            transport_retries += 1
            time.sleep(wait)

    parsed, parse_err = safe_parse(game, response)
    response_sha = sha256_text(response)
    actual_model = (meta.get("model") if isinstance(meta, dict) else None) or model

    response_file = temp_dir / f"sample_{sample_idx}.txt"
    with response_file.open("w") as f:
        f.write(f"MODEL_REQUESTED: {model}\n")
        f.write(f"MODEL_RETURNED: {actual_model}\n")
        f.write(f"PROVIDER: {provider}\n")
        f.write(f"HORIZON: {horizon}\n")
        f.write(f"TEMPERATURE: {temperature}\n")
        f.write(f"PROMPT_LABEL: {label}\n")
        f.write(f"SAMPLE: {sample_idx}\n")
        f.write(f"PARSED_ACTION: {parsed}\n")
        f.write(f"PARSE_ERROR: {parse_err}\n")
        f.write(f"RESPONSE_SHA256: {response_sha}\n\n")
        f.write("PROMPT:\n")
        f.write(prompt)
        f.write("\n\nRESPONSE:\n")
        f.write(response)
        f.write("\n")

    return RunRecord(
        model=actual_model,
        model_requested=model,
        provider=provider,
        horizon=horizon,
        temperature=temperature,
        prompt_label=label,
        prompt_condition=condition_text,
        sample_index=sample_idx,
        response_path=str(response_file.relative_to(output_dir)),
        response_sha256=response_sha,
        parsed_action=parsed,
        parse_error=parse_err,
        meta={
            **(meta if isinstance(meta, dict) else {}),
            "transport_retries": transport_retries,
        },
    )


def run_grid(
    models: List[str],
    provider: str,
    horizon: int,
    temperature: float,
    samples_per_cell: int,
    start_sample: int,
    num_players: int,
    endowment: float,
    multiplier: float,
    transparency: bool,
    reasoning: bool,
    output_dir: Path,
    prompt_labels: List[str] | None = None,
    parallelism: int = 1,
    max_transport_retries: int = 4,
    transport_backoff_s: float = 5.0,
    skip_existing: bool = False,
) -> None:
    load_dotenv()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = prompt_labels or list(PROMPT_CONDITIONS.keys())

    config = {
        "experiment": "2.1",
        "mode": "one_shot_prompt_condition_sweep",
        "models": models,
        "provider": provider,
        "horizon": horizon,
        "temperature": temperature,
        "samples_per_cell": samples_per_cell,
        "start_sample": start_sample,
        "num_players": num_players,
        "endowment": endowment,
        "multiplier": multiplier,
        "transparency": transparency,
        "reasoning": reasoning,
        "prompt_labels": labels,
        "parallelism": parallelism,
        "max_transport_retries": max_transport_retries,
        "transport_backoff_s": transport_backoff_s,
        "skip_existing": skip_existing,
        "baseline_prompt_label": "neutral",
        "env_provider": os.getenv("LLM_PROVIDER"),
        "openrouter_max_tokens": os.getenv("OPENROUTER_MAX_TOKENS"),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    all_runs: List[RunRecord] = []
    total = len(models) * len(labels)
    done = 0

    for model in models:
        model_dir = output_dir / safe_model_name(model)
        model_dir.mkdir(parents=True, exist_ok=True)

        for label in labels:
            condition_text = PROMPT_CONDITIONS[label]
            game = PublicGoodsGame(
                num_players=num_players,
                endowment=endowment,
                multiplier=multiplier,
                transparency=transparency,
                reasoning=reasoning,
                prompt_condition=condition_text,
            )
            prompt = build_prompt(game, horizon)

            label_dir = model_dir / label / f"horizon_{horizon}"
            label_dir.mkdir(parents=True, exist_ok=True)
            (label_dir / "prompt.txt").write_text(prompt)

            done += 1
            print(f"\n[{done}/{total}] model={model}  prompt={label}")
            t0 = time.time()

            temp_dir = label_dir / f"temp_{temperature}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            sample_indices = list(range(start_sample, start_sample + samples_per_cell))
            if skip_existing:
                original_n = len(sample_indices)
                sample_indices = [
                    idx for idx in sample_indices
                    if not (temp_dir / f"sample_{idx}.txt").exists()
                ]
                skipped = original_n - len(sample_indices)
                if skipped:
                    print(f"  skip_existing: skipped {skipped} existing sample(s)")
                if not sample_indices:
                    elapsed = round(time.time() - t0, 2)
                    print(f"  elapsed={elapsed:.1f}s")
                    continue

            if parallelism <= 1:
                for sample_idx in sample_indices:
                    try:
                        record = _run_sample(
                            model=model,
                            provider=provider,
                            temperature=temperature,
                            prompt=prompt,
                            game=game,
                            horizon=horizon,
                            label=label,
                            condition_text=condition_text,
                            sample_idx=sample_idx,
                            output_dir=output_dir,
                            temp_dir=temp_dir,
                            max_transport_retries=max_transport_retries,
                            transport_backoff_s=transport_backoff_s,
                        )
                    except Exception as e:
                        print(
                            f"FAIL model={model} prompt={label} horizon={horizon} "
                            f"temp={temperature} sample={sample_idx} err={e!r}"
                        )
                        _save_grid_summary(output_dir, config, all_runs)
                        continue

                    all_runs.append(record)
                    print(
                        f"OK model={model} prompt={label} horizon={horizon} "
                        f"temp={temperature} sample={sample_idx} parsed={record.parsed_action} "
                        f"sha={record.response_sha256[:8]}"
                    )
                    _save_grid_summary(output_dir, config, all_runs)
            else:
                with ThreadPoolExecutor(max_workers=parallelism) as executor:
                    futures = {
                        executor.submit(
                            _run_sample,
                            model=model,
                            provider=provider,
                            temperature=temperature,
                            prompt=prompt,
                            game=game,
                            horizon=horizon,
                            label=label,
                            condition_text=condition_text,
                            sample_idx=sample_idx,
                            output_dir=output_dir,
                            temp_dir=temp_dir,
                            max_transport_retries=max_transport_retries,
                            transport_backoff_s=transport_backoff_s,
                        ): sample_idx
                        for sample_idx in sample_indices
                    }
                    for future in as_completed(futures):
                        sample_idx = futures[future]
                        try:
                            record = future.result()
                        except Exception as e:
                            print(
                                f"FAIL model={model} prompt={label} horizon={horizon} "
                                f"temp={temperature} sample={sample_idx} err={e!r}"
                            )
                            _save_grid_summary(output_dir, config, all_runs)
                            continue

                        all_runs.append(record)
                        print(
                            f"OK model={model} prompt={label} horizon={horizon} "
                            f"temp={temperature} sample={sample_idx} parsed={record.parsed_action} "
                            f"sha={record.response_sha256[:8]}"
                        )
                        _save_grid_summary(output_dir, config, all_runs)
                all_runs.sort(
                    key=lambda r: (
                        r.model_requested,
                        r.prompt_label,
                        r.horizon,
                        r.temperature,
                        r.sample_index,
                    )
                )
                _save_grid_summary(output_dir, config, all_runs)

            elapsed = round(time.time() - t0, 2)
            print(f"  elapsed={elapsed:.1f}s")

    summary_runs = _load_summary_runs(output_dir)
    print(
        f"\nDone. {len(all_runs)} new run(s), {len(summary_runs)} total run(s). "
        f"Output: {output_dir}"
    )
    _print_table(summary_runs or [asdict(r) for r in all_runs], models, labels, endowment)


def _print_table(
    runs: List[Dict[str, Any]],
    models: List[str],
    labels: List[str],
    endowment: float,
) -> None:
    contrib: Dict[tuple, List[float]] = {}
    for r in runs:
        val = r.get("parsed_action")
        if val is None:
            continue
        contrib.setdefault((r["model_requested"], r.get("prompt_label", "neutral")), []).append(val)

    col_w = 14
    header = f"{'model':<28}" + "".join(f"{lbl[:col_w]:>{col_w}}" for lbl in labels)
    print("\n" + header)
    print("-" * len(header))
    for m in models:
        short = m.split("/")[-1][:26]
        row = f"{short:<28}"
        for lbl in labels:
            vals = contrib.get((m, lbl), [])
            if vals:
                mean = sum(vals) / len(vals)
                cell = f"{mean:.1f} ({mean/endowment*100:.0f}%)"
            else:
                cell = "?"
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
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--samples-per-cell", type=int, default=4)
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--max-transport-retries", type=int, default=4)
    parser.add_argument("--transport-backoff-s", type=float, default=5.0)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-query samples whose sample_<i>.txt file already exists.",
    )
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--endowment", type=float, default=20.0)
    parser.add_argument("--multiplier", type=float, default=1.6)
    parser.add_argument(
        "--transparency",
        action="store_true",
        help="Reveal individual contributions between rounds. Off by default to match Exp. 1.1.",
    )
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--prompt-labels", type=str, nargs="+", default=None,
        help=f"Subset of prompt labels to run. Available: {list(PROMPT_CONDITIONS)}.")
    parser.add_argument("--output", type=Path, default=Path("outputs/exp2_1_h10"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_grid(
        models=args.models,
        provider=args.provider,
        horizon=args.horizon,
        temperature=args.temperature,
        samples_per_cell=args.samples_per_cell,
        start_sample=args.start_sample,
        num_players=args.num_players,
        endowment=args.endowment,
        multiplier=args.multiplier,
        transparency=args.transparency,
        reasoning=not args.no_reasoning,
        output_dir=args.output,
        prompt_labels=args.prompt_labels,
        parallelism=args.parallelism,
        max_transport_retries=args.max_transport_retries,
        transport_backoff_s=args.transport_backoff_s,
        skip_existing=args.skip_existing,
    )
