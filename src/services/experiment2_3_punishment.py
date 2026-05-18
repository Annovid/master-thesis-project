"""Experiment 2.3 -- repeated Public Goods Game with punishment.

This experiment validates the best Exp. 2.2 no-punishment approximations in a
second-stage punishment setting. Each round has two LLM decision stages:

  1. contribution to the public good;
  2. punishment points assigned to the other players after contributions are
     revealed.

With 4 players and 10 rounds, one session costs 80 model requests. The intended
240-request run uses three cells:

  - openai/gpt-4o | human_mixed_moderate
  - meta-llama/llama-4-maverick | human_mixed_moderate
  - openai/gpt-4o | neutral
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from src.services.experiment1_2 import safe_model_name, sha256_text
from src.services.experiment2_2_human_dynamics import (
    ConditionSpec,
    available_conditions,
    condition_spec,
    linear_slope,
    pstdev,
)
from src.services.single_request import build_connector


DEFAULT_CELLS = [
    "openai/gpt-4o|human_mixed_moderate",
    "meta-llama/llama-4-maverick|human_mixed_moderate",
    "openai/gpt-4o|neutral",
]


def build_llm_connector(model: str, temperature: float, provider: str):
    cfg: Dict[str, Any] = {"model": model, "temperature": temperature}
    if provider == "mock":
        cfg["type"] = "mock"
    return build_connector(cfg, default_provider=provider)


def parse_cells(cells: List[str], num_players: int) -> List[tuple[str, ConditionSpec]]:
    parsed: List[tuple[str, ConditionSpec]] = []
    for cell in cells:
        if "|" not in cell:
            raise ValueError(f"Cell must be MODEL|CONDITION, got: {cell}")
        model, condition_label = [part.strip() for part in cell.split("|", 1)]
        if not model or not condition_label:
            raise ValueError(f"Cell must be MODEL|CONDITION, got: {cell}")
        parsed.append((model, condition_spec(condition_label, num_players)))
    return parsed


def contribution_prompt(
    *,
    player_id: int,
    condition: ConditionSpec,
    history: List[Dict[str, Any]],
    num_players: int,
    endowment: float,
    multiplier: float,
    n_rounds: int,
    punishment_cost: float,
    punishment_impact: float,
) -> str:
    history_str = format_history(history, player_id)
    lines = [
        f"You are playing the Public Goods Game with punishment for {n_rounds} rounds with {num_players} players in total.",
        f"You are Player {player_id + 1}.",
        f"Each round, every player receives an endowment of {endowment:.0f} tokens.",
        f"Stage 1: every player independently chooses how many tokens to contribute to a public pot, any amount from 0 to {endowment:.0f}.",
        f"The sum of all contributions is multiplied by {multiplier} and divided equally among all {num_players} players.",
        f"Stage 2: after contributions are revealed, each player may assign punishment points to other players.",
        f"Each punishment point you assign costs you {punishment_cost:g} token and reduces the target player's payoff by {punishment_impact:g} tokens.",
        "You cannot punish yourself.",
        "Your final score is the sum of your payoffs across all rounds.",
        "",
        "History of previous rounds:",
        history_str,
        "",
    ]
    if condition.player_prompt_conditions[player_id]:
        lines += [condition.player_prompt_conditions[player_id], ""]
    lines += [
        "Decide your contribution for the current round.",
        "End your reply with a single line of the form: Answer = N",
        f"where N is a number between 0 and {endowment:.0f}.",
    ]
    return "\n".join(lines)


def punishment_prompt(
    *,
    player_id: int,
    condition: ConditionSpec,
    history: List[Dict[str, Any]],
    current_contributions: List[float],
    preliminary_payoffs: List[float],
    num_players: int,
    endowment: float,
    multiplier: float,
    n_rounds: int,
    punishment_cost: float,
    punishment_impact: float,
    max_punishment_points: int,
) -> str:
    contribs = ", ".join(
        f"Player {idx + 1}: {value:g}" for idx, value in enumerate(current_contributions)
    )
    payoff_lines = ", ".join(
        f"Player {idx + 1}: {value:.2f}" for idx, value in enumerate(preliminary_payoffs)
    )
    history_str = format_history(history, player_id)
    other_players = [idx + 1 for idx in range(num_players) if idx != player_id]
    keys = ", ".join(f'"{p}"' for p in other_players)
    example = "{" + ", ".join(f'"{p}": 0' for p in other_players) + "}"

    lines = [
        f"You are playing the Public Goods Game with punishment for {n_rounds} rounds with {num_players} players in total.",
        f"You are Player {player_id + 1}.",
        f"Each round starts with {endowment:.0f} tokens per player. Public-good multiplier: {multiplier}.",
        f"Each punishment point you assign costs you {punishment_cost:g} token and reduces the target player's payoff by {punishment_impact:g} tokens.",
        f"You may assign 0 to {max_punishment_points} punishment points to each other player. You cannot punish yourself.",
        "",
        "History of previous rounds:",
        history_str,
        "",
        "Current round contributions:",
        contribs,
        "Current round payoffs before punishment:",
        payoff_lines,
        "",
    ]
    if condition.player_prompt_conditions[player_id]:
        lines += [condition.player_prompt_conditions[player_id], ""]
    lines += [
        "Decide punishment points for the current round.",
        f"Use exactly these player keys: {keys}.",
        f"End your reply with a single line of the form: Punishment = {example}",
    ]
    return "\n".join(lines)


def format_history(history: List[Dict[str, Any]], player_id: int) -> str:
    if not history:
        return "No previous rounds."
    lines: List[str] = []
    for round_record in history:
        contribs = ", ".join(
            f"Player {idx + 1}: {value:g}"
            for idx, value in enumerate(round_record["contributions"])
        )
        my_given = sum(round_record["punishments"][player_id])
        my_received = sum(row[player_id] for row in round_record["punishments"])
        my_payoff = round_record["final_payoffs"][player_id]
        lines.append(
            f"Round {round_record['round']}: contributions - {contribs}; "
            f"you assigned {my_given:g} punishment point(s), received {my_received:g} "
            f"punishment point(s), and your final payoff was {my_payoff:.2f}."
        )
    return "\n".join(lines)


def parse_contribution(response: str, endowment: float) -> tuple[Optional[float], Optional[str]]:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    bare_number = re.compile(r"^[-+]?\d*\.?\d+$")
    answer_pattern = re.compile(r"answer\s*=\s*([-+]?\d*\.?\d+)", re.IGNORECASE)
    for line in reversed(lines):
        cleaned = line.strip('* _`"\'').strip().rstrip(".,;:")
        if bare_number.match(cleaned):
            value = float(cleaned)
            return validate_contribution(value, endowment)
    for line in reversed(lines):
        match = answer_pattern.search(line.strip('* _`').strip())
        if match:
            value = float(match.group(1).rstrip("."))
            return validate_contribution(value, endowment)
    return None, "Could not parse contribution. Expected Answer = N."


def validate_contribution(value: float, endowment: float) -> tuple[Optional[float], Optional[str]]:
    if not math.isfinite(value):
        return None, "Contribution is not finite."
    if value < 0 or value > endowment:
        return None, f"Contribution {value} is outside [0, {endowment}]."
    return value, None


def parse_punishment(
    response: str,
    *,
    player_id: int,
    num_players: int,
    max_punishment_points: int,
) -> tuple[Optional[List[float]], Optional[str]]:
    text = response.strip()
    answer_match = re.search(r"punishment\s*=\s*(\{.*?\})", text, re.IGNORECASE | re.DOTALL)
    candidates = [answer_match.group(1)] if answer_match else []
    candidates += re.findall(r"\{.*?\}", text, flags=re.DOTALL)

    for candidate in candidates:
        try:
            raw = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        row = [0.0] * num_players
        ok = True
        for idx in range(num_players):
            if idx == player_id:
                continue
            key = str(idx + 1)
            if key not in raw:
                ok = False
                break
            try:
                value = float(raw[key])
            except (TypeError, ValueError):
                ok = False
                break
            if not math.isfinite(value) or value < 0 or value > max_punishment_points:
                ok = False
                break
            row[idx] = value
        if ok:
            return row, None

    return None, "Could not parse punishment. Expected Punishment = JSON object with other player ids."


def query_with_retries(
    connector,
    conversation: List[Dict[str, str]],
    prompt: str,
    parse_fn: Callable[[str], tuple[Optional[Any], Optional[str]]],
    *,
    max_parse_retries: int,
    max_transport_retries: int,
    transport_backoff_s: float,
) -> Dict[str, Any]:
    pre_len = len(conversation)
    conversation.append({"role": "user", "content": prompt})
    parse_retries = 0
    transport_retries = 0

    while True:
        try:
            response, meta = connector.query_conversation(conversation)
        except Exception as exc:
            if transport_retries >= max_transport_retries:
                del conversation[pre_len:]
                raise
            wait = transport_backoff_s * (2 ** transport_retries)
            print(
                f"TRANSPORT_RETRY attempt={transport_retries + 1}/"
                f"{max_transport_retries} wait={wait:.1f}s err={type(exc).__name__}: {exc}"
            )
            transport_retries += 1
            time.sleep(wait)
            continue

        parsed, parse_error = parse_fn(response)
        if parsed is not None:
            conversation.append({"role": "assistant", "content": response})
            return {
                "response": response,
                "parsed": parsed,
                "parse_error": None,
                "meta": meta,
                "retries": parse_retries,
                "transport_retries": transport_retries,
            }

        if parse_retries >= max_parse_retries:
            del conversation[pre_len:]
            return {
                "response": response,
                "parsed": None,
                "parse_error": parse_error,
                "meta": meta,
                "retries": parse_retries,
                "transport_retries": transport_retries,
            }
        parse_retries += 1


def compute_preliminary_payoffs(
    contributions: List[float],
    *,
    endowment: float,
    multiplier: float,
) -> List[float]:
    share = multiplier * sum(contributions) / len(contributions)
    return [endowment - contribution + share for contribution in contributions]


def compute_final_payoffs(
    preliminary_payoffs: List[float],
    punishments: List[List[float]],
    *,
    punishment_cost: float,
    punishment_impact: float,
) -> List[float]:
    num_players = len(preliminary_payoffs)
    final: List[float] = []
    for idx in range(num_players):
        cost_given = punishment_cost * sum(punishments[idx])
        loss_received = punishment_impact * sum(row[idx] for row in punishments)
        final.append(preliminary_payoffs[idx] - cost_given - loss_received)
    return final


def write_stage_record(
    path: Path,
    *,
    model_requested: str,
    model_returned: str,
    provider: str,
    condition_label: str,
    player_profile: str,
    session_idx: int,
    round_num: int,
    player_id: int,
    stage: str,
    temperature: float,
    parsed: Any,
    parse_error: Optional[str],
    retries: int,
    transport_retries: int,
    response_sha256: str,
    prompt: str,
    response: str,
) -> None:
    with path.open("w") as f:
        f.write(f"MODEL_REQUESTED: {model_requested}\n")
        f.write(f"MODEL_RETURNED: {model_returned}\n")
        f.write(f"PROVIDER: {provider}\n")
        f.write(f"EXPERIMENT: 2.3\n")
        f.write(f"CONDITION: {condition_label}\n")
        f.write(f"PLAYER_PROFILE: {player_profile}\n")
        f.write(f"SESSION: {session_idx}\n")
        f.write(f"ROUND: {round_num}\n")
        f.write(f"PLAYER: {player_id + 1}\n")
        f.write(f"STAGE: {stage}\n")
        f.write(f"TEMPERATURE: {temperature}\n")
        f.write(f"PARSED: {json.dumps(parsed, ensure_ascii=False)}\n")
        f.write(f"PARSE_ERROR: {parse_error}\n")
        f.write(f"RETRIES: {retries}\n")
        f.write(f"TRANSPORT_RETRIES: {transport_retries}\n")
        f.write(f"RESPONSE_SHA256: {response_sha256}\n\n")
        f.write("PROMPT:\n")
        f.write(prompt)
        f.write("\n\nRESPONSE:\n")
        f.write(response)
        f.write("\n")


def compute_metrics(rounds: List[Dict[str, Any]], endowment: float) -> Dict[str, Any]:
    complete_rounds = [r for r in rounds if not r.get("incomplete")]
    if not complete_rounds:
        return {
            "complete_rounds": 0,
            "round_means": [],
            "r1_mean": None,
            "r_last_mean": None,
            "slope": None,
            "final_zero_rate": None,
        }

    contributions = [list(map(float, r["contributions"])) for r in complete_rounds]
    round_means = [sum(values) / len(values) for values in contributions]
    round_stds = [pstdev(values) for values in contributions]
    final_zero_rate = sum(1 for v in contributions[-1] if abs(v) < 1e-9) / len(contributions[-1])

    total_points = 0.0
    points_to_below_avg = 0.0
    received_pairs: List[tuple[float, float]] = []
    for round_record, round_contribs in zip(complete_rounds, contributions):
        mean_contrib = sum(round_contribs) / len(round_contribs)
        punishments = round_record["punishments"]
        received = [sum(row[idx] for row in punishments) for idx in range(len(round_contribs))]
        for target_id, contribution in enumerate(round_contribs):
            received_pairs.append((contribution, received[target_id]))
            for source_id in range(len(round_contribs)):
                if source_id == target_id:
                    continue
                points = float(punishments[source_id][target_id])
                total_points += points
                if contribution < mean_contrib:
                    points_to_below_avg += points

    corr = pearson_corr([x for x, _ in received_pairs], [y for _, y in received_pairs])
    below_share = points_to_below_avg / total_points if total_points > 0 else None
    return {
        "complete_rounds": len(complete_rounds),
        "round_means": round_means,
        "round_stds": round_stds,
        "r1_mean": round_means[0],
        "r_last_mean": round_means[-1],
        "overall_mean": sum(round_means) / len(round_means),
        "slope": linear_slope(round_means),
        "final_zero_rate": final_zero_rate,
        "avg_punishment_points_per_round": total_points / len(complete_rounds),
        "avg_punishment_points_per_player_round": total_points / (len(complete_rounds) * len(contributions[0])),
        "punishment_to_below_avg_share": below_share,
        "contribution_punishment_received_corr": corr,
        "last3_mean": sum(round_means[-3:]) / min(3, len(round_means)),
        "first3_mean": sum(round_means[:3]) / min(3, len(round_means)),
        "cooperation_sustain_ratio": (
            (sum(round_means[-3:]) / min(3, len(round_means)))
            / max(1e-9, (sum(round_means[:3]) / min(3, len(round_means))))
        ),
        "endowment": endowment,
    }


def pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0 or y_var == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / math.sqrt(x_var * y_var)


def run_stage_parallel(
    *,
    num_players: int,
    parallelism: int,
    fn: Callable[[int], Dict[str, Any]],
) -> List[Optional[Dict[str, Any]]]:
    results: List[Optional[Dict[str, Any]]] = [None] * num_players
    if parallelism <= 1:
        for player_id in range(num_players):
            results[player_id] = fn(player_id)
        return results

    with ThreadPoolExecutor(max_workers=min(parallelism, num_players)) as executor:
        futures = {executor.submit(fn, player_id): player_id for player_id in range(num_players)}
        for future in as_completed(futures):
            player_id = futures[future]
            results[player_id] = future.result()
    return results


def run_session(
    *,
    model: str,
    provider: str,
    temperature: float,
    session_idx: int,
    condition: ConditionSpec,
    num_players: int,
    endowment: float,
    multiplier: float,
    n_rounds: int,
    punishment_cost: float,
    punishment_impact: float,
    max_punishment_points: int,
    output_dir: Path,
    max_parse_retries: int,
    max_transport_retries: int,
    transport_backoff_s: float,
    parallelism: int,
) -> Dict[str, Any]:
    model_safe = safe_model_name(model)
    session_dir = output_dir / model_safe / condition.label / f"session_{session_idx}"
    session_dir.mkdir(parents=True, exist_ok=True)

    connectors = [build_llm_connector(model, temperature, provider) for _ in range(num_players)]
    conversations: List[List[Dict[str, str]]] = [[] for _ in range(num_players)]
    history: List[Dict[str, Any]] = []
    round_records: List[Dict[str, Any]] = []
    total_payoffs = [0.0] * num_players
    model_returned_seen: set[str] = set()
    incomplete = False
    abort_reason: Optional[str] = None
    t_session_start = time.time()

    for round_num in range(1, n_rounds + 1):
        round_dir = session_dir / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)

        contribution_prompts = [
            contribution_prompt(
                player_id=player_id,
                condition=condition,
                history=history,
                num_players=num_players,
                endowment=endowment,
                multiplier=multiplier,
                n_rounds=n_rounds,
                punishment_cost=punishment_cost,
                punishment_impact=punishment_impact,
            )
            for player_id in range(num_players)
        ]

        def contribution_turn(player_id: int) -> Dict[str, Any]:
            prompt = contribution_prompts[player_id]
            t0 = time.time()
            turn = query_with_retries(
                connectors[player_id],
                conversations[player_id],
                prompt,
                lambda response: parse_contribution(response, endowment),
                max_parse_retries=max_parse_retries,
                max_transport_retries=max_transport_retries,
                transport_backoff_s=transport_backoff_s,
            )
            response = turn["response"]
            meta = turn["meta"] or {}
            model_returned = (meta.get("model") if isinstance(meta, dict) else None) or model
            response_sha = sha256_text(response)
            player_profile = condition.player_profiles[player_id]
            player_file = round_dir / f"player_{player_id + 1}_contribution.txt"
            write_stage_record(
                player_file,
                model_requested=model,
                model_returned=model_returned,
                provider=provider,
                condition_label=condition.label,
                player_profile=player_profile,
                session_idx=session_idx,
                round_num=round_num,
                player_id=player_id,
                stage="contribution",
                temperature=temperature,
                parsed=turn["parsed"],
                parse_error=turn["parse_error"],
                retries=turn["retries"],
                transport_retries=turn["transport_retries"],
                response_sha256=response_sha,
                prompt=prompt,
                response=response,
            )
            return {
                "player_id": player_id + 1,
                "player_profile": player_profile,
                "parsed": turn["parsed"],
                "parse_error": turn["parse_error"],
                "retries": turn["retries"],
                "transport_retries": turn["transport_retries"],
                "response_sha256": response_sha,
                "response_path": str(player_file.relative_to(output_dir)),
                "model_returned": model_returned,
                "usage": meta.get("usage") if isinstance(meta, dict) else None,
                "elapsed_s": round(time.time() - t0, 2),
            }

        try:
            contribution_results = run_stage_parallel(
                num_players=num_players,
                parallelism=parallelism,
                fn=contribution_turn,
            )
        except Exception as exc:
            abort_reason = (
                f"transport error model={model} condition={condition.label} "
                f"session={session_idx} round={round_num} stage=contribution: {exc!r}"
            )
            print(f"FAIL {abort_reason}")
            incomplete = True
            break

        contributions: List[float] = []
        contribution_records: List[Dict[str, Any]] = []
        for player_id, result in enumerate(contribution_results):
            if result is None or result["parsed"] is None:
                abort_reason = (
                    f"parse error model={model} condition={condition.label} "
                    f"session={session_idx} round={round_num} player={player_id + 1} stage=contribution"
                )
                print(f"FAIL {abort_reason}")
                incomplete = True
                break
            model_returned_seen.add(result["model_returned"])
            contributions.append(float(result["parsed"]))
            contribution_records.append(result)
            print(
                f"OK model={model} condition={condition.label} session={session_idx} "
                f"round={round_num} player={player_id + 1} stage=contribution "
                f"profile={result['player_profile']} parsed={result['parsed']} "
                f"sha={result['response_sha256'][:8]} retries={result['retries']} "
                f"elapsed={result['elapsed_s']}s"
            )
        if incomplete:
            break

        preliminary_payoffs = compute_preliminary_payoffs(
            contributions,
            endowment=endowment,
            multiplier=multiplier,
        )
        punishment_prompts = [
            punishment_prompt(
                player_id=player_id,
                condition=condition,
                history=history,
                current_contributions=contributions,
                preliminary_payoffs=preliminary_payoffs,
                num_players=num_players,
                endowment=endowment,
                multiplier=multiplier,
                n_rounds=n_rounds,
                punishment_cost=punishment_cost,
                punishment_impact=punishment_impact,
                max_punishment_points=max_punishment_points,
            )
            for player_id in range(num_players)
        ]

        def punishment_turn(player_id: int) -> Dict[str, Any]:
            prompt = punishment_prompts[player_id]
            t0 = time.time()
            turn = query_with_retries(
                connectors[player_id],
                conversations[player_id],
                prompt,
                lambda response: parse_punishment(
                    response,
                    player_id=player_id,
                    num_players=num_players,
                    max_punishment_points=max_punishment_points,
                ),
                max_parse_retries=max_parse_retries,
                max_transport_retries=max_transport_retries,
                transport_backoff_s=transport_backoff_s,
            )
            response = turn["response"]
            meta = turn["meta"] or {}
            model_returned = (meta.get("model") if isinstance(meta, dict) else None) or model
            response_sha = sha256_text(response)
            player_profile = condition.player_profiles[player_id]
            player_file = round_dir / f"player_{player_id + 1}_punishment.txt"
            write_stage_record(
                player_file,
                model_requested=model,
                model_returned=model_returned,
                provider=provider,
                condition_label=condition.label,
                player_profile=player_profile,
                session_idx=session_idx,
                round_num=round_num,
                player_id=player_id,
                stage="punishment",
                temperature=temperature,
                parsed=turn["parsed"],
                parse_error=turn["parse_error"],
                retries=turn["retries"],
                transport_retries=turn["transport_retries"],
                response_sha256=response_sha,
                prompt=prompt,
                response=response,
            )
            return {
                "player_id": player_id + 1,
                "player_profile": player_profile,
                "parsed": turn["parsed"],
                "parse_error": turn["parse_error"],
                "retries": turn["retries"],
                "transport_retries": turn["transport_retries"],
                "response_sha256": response_sha,
                "response_path": str(player_file.relative_to(output_dir)),
                "model_returned": model_returned,
                "usage": meta.get("usage") if isinstance(meta, dict) else None,
                "elapsed_s": round(time.time() - t0, 2),
            }

        try:
            punishment_results = run_stage_parallel(
                num_players=num_players,
                parallelism=parallelism,
                fn=punishment_turn,
            )
        except Exception as exc:
            abort_reason = (
                f"transport error model={model} condition={condition.label} "
                f"session={session_idx} round={round_num} stage=punishment: {exc!r}"
            )
            print(f"FAIL {abort_reason}")
            incomplete = True
            break

        punishments: List[List[float]] = []
        punishment_records: List[Dict[str, Any]] = []
        for player_id, result in enumerate(punishment_results):
            if result is None or result["parsed"] is None:
                abort_reason = (
                    f"parse error model={model} condition={condition.label} "
                    f"session={session_idx} round={round_num} player={player_id + 1} stage=punishment"
                )
                print(f"FAIL {abort_reason}")
                incomplete = True
                break
            model_returned_seen.add(result["model_returned"])
            punishments.append([float(value) for value in result["parsed"]])
            punishment_records.append(result)
            print(
                f"OK model={model} condition={condition.label} session={session_idx} "
                f"round={round_num} player={player_id + 1} stage=punishment "
                f"profile={result['player_profile']} parsed={result['parsed']} "
                f"sha={result['response_sha256'][:8]} retries={result['retries']} "
                f"elapsed={result['elapsed_s']}s"
            )
        if incomplete:
            break

        final_payoffs = compute_final_payoffs(
            preliminary_payoffs,
            punishments,
            punishment_cost=punishment_cost,
            punishment_impact=punishment_impact,
        )
        for idx, payoff in enumerate(final_payoffs):
            total_payoffs[idx] += payoff

        round_record = {
            "round": round_num,
            "contributions": contributions,
            "preliminary_payoffs": preliminary_payoffs,
            "punishments": punishments,
            "final_payoffs": final_payoffs,
            "contribution_players": contribution_records,
            "punishment_players": punishment_records,
            "incomplete": False,
        }
        round_records.append(round_record)
        history.append(round_record)

    metrics = compute_metrics(round_records, endowment)
    session_summary = {
        "experiment": "2.3",
        "mode": "repeated_pgg_with_punishment_validation",
        "model_requested": model,
        "model_returned_seen": sorted(model_returned_seen),
        "provider": provider,
        "condition_label": condition.label,
        "condition_mode": condition.mode,
        "session": session_idx,
        "temperature": temperature,
        "num_players": num_players,
        "endowment": endowment,
        "multiplier": multiplier,
        "n_rounds": n_rounds,
        "punishment_cost": punishment_cost,
        "punishment_impact": punishment_impact,
        "max_punishment_points": max_punishment_points,
        "player_profiles": condition.player_profiles,
        "player_prompt_conditions": condition.player_prompt_conditions,
        "incomplete": incomplete,
        "abort_reason": abort_reason,
        "rounds": round_records,
        "metrics": metrics,
        "total_payoffs": total_payoffs if not incomplete else None,
        "session_path": str(session_dir.relative_to(output_dir)),
        "session_elapsed_s": round(time.time() - t_session_start, 2),
    }
    (session_dir / "session_summary.json").write_text(
        json.dumps(session_summary, indent=2, ensure_ascii=False)
    )
    return session_summary


def save_summary(output_dir: Path, config: Dict[str, Any], sessions: List[Dict[str, Any]]) -> None:
    path = output_dir / "run_summary.json"
    existing: List[Dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text()).get("sessions", [])
        except Exception:
            existing = []
    new_keys = {
        (s.get("model_requested"), s.get("condition_label"), s.get("session"))
        for s in sessions
    }
    kept = [
        s for s in existing
        if (s.get("model_requested"), s.get("condition_label"), s.get("session")) not in new_keys
    ]
    merged = kept + sessions
    merged.sort(key=lambda s: (s.get("model_requested", ""), s.get("condition_label", ""), s.get("session", 0)))
    path.write_text(json.dumps({**config, "sessions": merged}, indent=2, ensure_ascii=False))


def run_grid(
    *,
    cells: List[str],
    provider: str,
    temperature: float,
    session_idx: int,
    num_players: int,
    endowment: float,
    multiplier: float,
    n_rounds: int,
    punishment_cost: float,
    punishment_impact: float,
    max_punishment_points: int,
    output_dir: Path,
    max_parse_retries: int,
    max_transport_retries: int,
    transport_backoff_s: float,
    parallelism: int,
    skip_existing: bool,
) -> None:
    load_dotenv()
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_cells = parse_cells(cells, num_players)
    config = {
        "experiment": "2.3",
        "mode": "repeated_pgg_with_punishment_validation",
        "cells": cells,
        "provider": provider,
        "temperature": temperature,
        "session": session_idx,
        "num_players": num_players,
        "endowment": endowment,
        "multiplier": multiplier,
        "n_rounds": n_rounds,
        "punishment_cost": punishment_cost,
        "punishment_impact": punishment_impact,
        "max_punishment_points": max_punishment_points,
        "conditions": [asdict(condition) for _, condition in parsed_cells],
        "expected_requests": len(parsed_cells) * n_rounds * num_players * 2,
        "max_parse_retries": max_parse_retries,
        "max_transport_retries": max_transport_retries,
        "transport_backoff_s": transport_backoff_s,
        "parallelism": parallelism,
        "skip_existing": skip_existing,
        "env_provider": os.getenv("LLM_PROVIDER"),
        "openrouter_max_tokens": os.getenv("OPENROUTER_MAX_TOKENS"),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    sessions: List[Dict[str, Any]] = []
    total = len(parsed_cells)
    for idx, (model, condition) in enumerate(parsed_cells, start=1):
        session_dir = output_dir / safe_model_name(model) / condition.label / f"session_{session_idx}"
        summary_path = session_dir / "session_summary.json"
        print(f"\n[{idx}/{total}] model={model} condition={condition.label} session={session_idx}")
        if skip_existing and summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
                sessions.append(summary)
                save_summary(output_dir, config, sessions)
                print("  skip_existing: loaded existing session_summary.json")
                continue
            except Exception:
                print("  skip_existing: existing summary unreadable, rerunning")

        summary = run_session(
            model=model,
            provider=provider,
            temperature=temperature,
            session_idx=session_idx,
            condition=condition,
            num_players=num_players,
            endowment=endowment,
            multiplier=multiplier,
            n_rounds=n_rounds,
            punishment_cost=punishment_cost,
            punishment_impact=punishment_impact,
            max_punishment_points=max_punishment_points,
            output_dir=output_dir,
            max_parse_retries=max_parse_retries,
            max_transport_retries=max_transport_retries,
            transport_backoff_s=transport_backoff_s,
            parallelism=parallelism,
        )
        sessions.append(summary)
        save_summary(output_dir, config, sessions)
        print_session_line(summary)

    all_sessions = load_summary_sessions(output_dir)
    print(f"\nDone. {len(sessions)} touched session(s), {len(all_sessions)} total. Output: {output_dir}")
    print_table(all_sessions)


def load_summary_sessions(output_dir: Path) -> List[Dict[str, Any]]:
    path = output_dir / "run_summary.json"
    if not path.exists():
        return []
    try:
        sessions = json.loads(path.read_text()).get("sessions", [])
    except Exception:
        return []
    return sessions if isinstance(sessions, list) else []


def print_session_line(summary: Dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    print(
        "  metrics "
        f"R1={metrics.get('r1_mean')} "
        f"Rlast={metrics.get('r_last_mean')} "
        f"slope={metrics.get('slope')} "
        f"punish/round={metrics.get('avg_punishment_points_per_round')} "
        f"below_share={metrics.get('punishment_to_below_avg_share')} "
        f"corr={metrics.get('contribution_punishment_received_corr')}"
    )


def print_table(sessions: List[Dict[str, Any]]) -> None:
    print("\nExp2.3 summary")
    header = (
        f"{'model':<28}{'condition':<24}{'R1':>8}{'R10':>8}{'slope':>10}"
        f"{'pun/round':>12}{'below%':>10}{'corr':>10}"
    )
    print(header)
    print("-" * len(header))
    for session in sessions:
        if session.get("incomplete"):
            continue
        metrics = session.get("metrics", {})
        below = metrics.get("punishment_to_below_avg_share")
        corr = metrics.get("contribution_punishment_received_corr")
        print(
            f"{session.get('model_requested', '').split('/')[-1][:26]:<28}"
            f"{session.get('condition_label', '')[:22]:<24}"
            f"{fmt(metrics.get('r1_mean')):>8}"
            f"{fmt(metrics.get('r_last_mean')):>8}"
            f"{fmt(metrics.get('slope')):>10}"
            f"{fmt(metrics.get('avg_punishment_points_per_round')):>12}"
            f"{fmt(None if below is None else 100 * below):>10}"
            f"{fmt(corr):>10}"
        )


def fmt(value: Any) -> str:
    if value is None:
        return "?"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2.3: repeated Public Goods Game with punishment."
    )
    parser.add_argument(
        "--cells",
        type=str,
        nargs="+",
        default=DEFAULT_CELLS,
        help="Cells as MODEL|CONDITION. Defaults to 3 cells = 240 requests for 10 rounds.",
    )
    parser.add_argument("--provider", type=str, default=os.getenv("LLM_PROVIDER", "auto"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--endowment", type=float, default=20.0)
    parser.add_argument("--multiplier", type=float, default=1.6)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--punishment-cost", type=float, default=1.0)
    parser.add_argument("--punishment-impact", type=float, default=3.0)
    parser.add_argument("--max-punishment-points", type=int, default=5)
    parser.add_argument("--max-parse-retries", type=int, default=1)
    parser.add_argument("--max-transport-retries", type=int, default=4)
    parser.add_argument("--transport-backoff-s", type=float, default=5.0)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/exp2_3_punishment"))
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="Print available condition labels and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list_conditions:
        print("\n".join(available_conditions()))
        raise SystemExit(0)
    run_grid(
        cells=args.cells,
        provider=args.provider,
        temperature=args.temperature,
        session_idx=args.session,
        num_players=args.num_players,
        endowment=args.endowment,
        multiplier=args.multiplier,
        n_rounds=args.n_rounds,
        punishment_cost=args.punishment_cost,
        punishment_impact=args.punishment_impact,
        max_punishment_points=args.max_punishment_points,
        output_dir=args.output,
        max_parse_retries=args.max_parse_retries,
        max_transport_retries=args.max_transport_retries,
        transport_backoff_s=args.transport_backoff_s,
        parallelism=args.parallelism,
        skip_existing=args.skip_existing,
    )
