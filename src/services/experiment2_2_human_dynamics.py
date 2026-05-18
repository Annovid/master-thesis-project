"""Experiment 2.2 -- human-like repeated PGG dynamics.

This experiment extends Exp. 2.1 from one-shot prompt effects to repeated-game
dynamics. It tests whether prompt mechanisms that move the first contribution
toward the human range also produce human-like trajectories: moderate initial
cooperation, decay without punishment, final free-riding, and within-group
heterogeneity.

Default grid:
  models x conditions x sessions

Conditions:
  - neutral: homogeneous baseline
  - persona: homogeneous human-participant framing from Exp. 2.1
  - social_norms: homogeneous norm-information framing from Exp. 2.1
  - self_interest: homogeneous Nash-like benchmark
  - group_welfare: homogeneous full-cooperation benchmark
  - human_mixed: heterogeneous profiles in one group

Output layout:
  outputs/exp2_2_human_dynamics/
    run_config.json
    run_summary.json
    <model_safe>/<condition_label>/session_<s>/
      session_summary.json
      round_<r>/player_<p>.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.gamesim.game.state import GameState
from src.gamesim.games.public_goods_game import PublicGoodsGame
from src.services.experiment1_2 import (
    _player_turn,
    safe_model_name,
    sha256_text,
)
from src.services.experiment2_1 import PROMPT_CONDITIONS
from src.services.single_request import build_connector


def build_llm_connector(model: str, temperature: float, provider: str):
    cfg: Dict[str, Any] = {"model": model, "temperature": temperature}
    if provider == "mock":
        cfg["type"] = "mock"
    return build_connector(cfg, default_provider=provider)


HUMAN_PROFILE_PROMPTS: Dict[str, str] = {
    "conditional_cooperator": (
        "You are a typical participant who is willing to cooperate when others "
        "also cooperate, but you reduce your contribution when others contribute "
        "less than expected. Consider reciprocity across rounds."
    ),
    "self_interested": (
        "You are a participant mainly focused on your own monetary payoff. You "
        "prefer keeping tokens unless contributing seems necessary because of "
        "the behavior of the group."
    ),
    "moderate_free_rider": (
        "You are somewhat self-interested and prefer to contribute less than "
        "highly cooperative players, but you may still contribute a positive "
        "amount when the group is cooperating."
    ),
    "high_cooperator": (
        "You are a participant who tends to contribute generously and values "
        "successful group cooperation, but you may reduce contributions if "
        "others repeatedly free-ride."
    ),
    "inequality_averse": (
        "You dislike unequal outcomes between players. You prefer decisions "
        "that avoid earning much more or much less than others, while still "
        "considering your own payoff."
    ),
}


DEFAULT_CONDITIONS = [
    "neutral",
    "persona",
    "social_norms",
    "self_interest",
    "group_welfare",
    "human_mixed",
    "human_mixed_moderate",
]


@dataclass(frozen=True)
class ConditionSpec:
    label: str
    mode: str
    prompt_condition: str
    player_profiles: List[str]
    player_prompt_conditions: List[str]


class PerPlayerPromptPublicGoodsGame(PublicGoodsGame):
    """PGG variant that can inject a different prompt paragraph per player."""

    def __init__(
        self,
        *args: Any,
        player_prompt_conditions: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.player_prompt_conditions = player_prompt_conditions or [""] * self.num_players
        if len(self.player_prompt_conditions) != self.num_players:
            raise ValueError("player_prompt_conditions length must match num_players")

    def get_prompt(self, state: GameState, player_id: int, max_rounds: int) -> str:
        original = self.prompt_condition
        self.prompt_condition = self.player_prompt_conditions[player_id]
        try:
            return super().get_prompt(state, player_id, max_rounds)
        finally:
            self.prompt_condition = original


def condition_spec(label: str, num_players: int) -> ConditionSpec:
    if label in {"human_mixed", "human_mixed_moderate"}:
        if label == "human_mixed":
            profiles = [
                "conditional_cooperator",
                "conditional_cooperator",
                "self_interested",
                "high_cooperator",
            ]
        else:
            profiles = [
                "conditional_cooperator",
                "conditional_cooperator",
                "moderate_free_rider",
                "high_cooperator",
            ]
        if num_players != 4:
            raise ValueError(f"{label} currently expects num_players=4")
        return ConditionSpec(
            label=label,
            mode="heterogeneous_profiles",
            prompt_condition="",
            player_profiles=profiles,
            player_prompt_conditions=[HUMAN_PROFILE_PROMPTS[p] for p in profiles],
        )

    if label not in PROMPT_CONDITIONS:
        raise ValueError(f"Unknown condition '{label}'. Available: {available_conditions()}")
    prompt_condition = PROMPT_CONDITIONS[label]
    return ConditionSpec(
        label=label,
        mode="homogeneous_prompt",
        prompt_condition=prompt_condition,
        player_profiles=[label] * num_players,
        player_prompt_conditions=[prompt_condition] * num_players,
    )


def available_conditions() -> List[str]:
    return sorted(set(DEFAULT_CONDITIONS) | set(PROMPT_CONDITIONS))


def _write_player_record(
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
    temperature: float,
    parsed_action: Optional[float],
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
        f.write(f"CONDITION: {condition_label}\n")
        f.write(f"PLAYER_PROFILE: {player_profile}\n")
        f.write(f"SESSION: {session_idx}\n")
        f.write(f"ROUND: {round_num}\n")
        f.write(f"PLAYER: {player_id + 1}\n")
        f.write(f"TEMPERATURE: {temperature}\n")
        f.write(f"PARSED_ACTION: {parsed_action}\n")
        f.write(f"PARSE_ERROR: {parse_error}\n")
        f.write(f"RETRIES: {retries}\n")
        f.write(f"TRANSPORT_RETRIES: {transport_retries}\n")
        f.write(f"RESPONSE_SHA256: {response_sha256}\n\n")
        f.write("PROMPT:\n")
        f.write(prompt)
        f.write("\n\nRESPONSE:\n")
        f.write(response)
        f.write("\n")


def linear_slope(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(1, n + 1))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom


def pstdev(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def compute_metrics(rounds: List[Dict[str, Any]], endowment: float) -> Dict[str, Any]:
    complete_rounds = [r for r in rounds if not r.get("incomplete") and r.get("actions")]
    if not complete_rounds:
        return {
            "complete_rounds": 0,
            "round_means": [],
            "round_stds": [],
            "r1_mean": None,
            "r_last_mean": None,
            "overall_mean": None,
            "slope": None,
            "final_zero_rate": None,
            "player_slopes": [],
            "rank_stability": None,
            "free_rider_effect": None,
            "target_round_means": [],
            "trajectory_distance_to_no_punishment_human": None,
            "endpoint_distance_to_no_punishment_human": None,
            "distance_to_no_punishment_human": None,
        }

    actions_by_round = [list(map(float, r["actions"])) for r in complete_rounds]
    round_means = [sum(actions) / len(actions) for actions in actions_by_round]
    round_stds = [pstdev(actions) for actions in actions_by_round]
    num_players = len(actions_by_round[0])
    by_player = [
        [actions[player_id] for actions in actions_by_round]
        for player_id in range(num_players)
    ]
    player_slopes = [linear_slope(vals) for vals in by_player]
    final_actions = actions_by_round[-1]
    final_zero_rate = sum(1 for a in final_actions if abs(a) < 1e-9) / len(final_actions)

    first_ranks = sorted(range(num_players), key=lambda i: actions_by_round[0][i])
    last_ranks = sorted(range(num_players), key=lambda i: actions_by_round[-1][i])
    rank_stability = sum(1 for a, b in zip(first_ranks, last_ranks) if a == b) / num_players

    free_rider_effect = None
    if len(actions_by_round) >= 2:
        effects: List[float] = []
        for t in range(len(actions_by_round) - 1):
            low = min(actions_by_round[t])
            others_next = []
            for player_id, action in enumerate(actions_by_round[t]):
                if action > low:
                    others_next.append(actions_by_round[t + 1][player_id] - action)
            if others_next:
                effects.append(sum(others_next) / len(others_next))
        if effects:
            free_rider_effect = sum(effects) / len(effects)

    target_round_means = human_target_trajectory(len(round_means), endowment)
    trajectory_distance = human_trajectory_distance(round_means, endowment)
    endpoint_distance = human_endpoint_distance(round_means, final_zero_rate, endowment)
    combined_distance = human_distance(round_means, final_zero_rate, endowment)

    return {
        "complete_rounds": len(complete_rounds),
        "round_means": round_means,
        "round_stds": round_stds,
        "r1_mean": round_means[0],
        "r_last_mean": round_means[-1],
        "overall_mean": sum(round_means) / len(round_means),
        "slope": linear_slope(round_means),
        "final_zero_rate": final_zero_rate,
        "player_slopes": player_slopes,
        "rank_stability": rank_stability,
        "free_rider_effect": free_rider_effect,
        "target_round_means": target_round_means,
        "trajectory_distance_to_no_punishment_human": trajectory_distance,
        "endpoint_distance_to_no_punishment_human": endpoint_distance,
        "distance_to_no_punishment_human": combined_distance,
    }


def human_target_trajectory(n_rounds: int, endowment: float) -> List[float]:
    if n_rounds <= 0:
        return []
    start = 0.5 * endowment
    end = 0.1 * endowment
    if n_rounds == 1:
        return [start]
    return [
        start + (end - start) * (idx / (n_rounds - 1))
        for idx in range(n_rounds)
    ]


def human_trajectory_distance(round_means: List[float], endowment: float) -> Optional[float]:
    if not round_means:
        return None
    target = human_target_trajectory(len(round_means), endowment)
    return sum(abs(actual - expected) for actual, expected in zip(round_means, target)) / len(round_means)


def human_endpoint_distance(
    round_means: List[float],
    final_zero_rate: Optional[float],
    endowment: float,
) -> Optional[float]:
    if not round_means or final_zero_rate is None:
        return None
    target_r1 = 0.5 * endowment
    target_last = 0.1 * endowment
    target_slope = (target_last - target_r1) / max(1, len(round_means) - 1)
    target_zero_rate = 0.6
    slope = linear_slope(round_means)
    return (
        abs(round_means[0] - target_r1)
        + abs(round_means[-1] - target_last)
        + 2.0 * abs(slope - target_slope)
        + 5.0 * abs(final_zero_rate - target_zero_rate)
    )


def human_distance(round_means: List[float], final_zero_rate: Optional[float], endowment: float) -> Optional[float]:
    if not round_means or final_zero_rate is None:
        return None
    # Coarse no-punishment targets from the PGG literature: initial 40-60%,
    # low final contribution, negative slope, high final zero rate. The
    # trajectory term prevents early collapses from looking too good merely
    # because the final period reaches zero.
    trajectory = human_trajectory_distance(round_means, endowment)
    endpoint = human_endpoint_distance(round_means, final_zero_rate, endowment)
    if trajectory is None or endpoint is None:
        return None
    return trajectory + 0.5 * endpoint


def _save_summary(output_dir: Path, config: Dict[str, Any], sessions: List[Dict[str, Any]]) -> None:
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
    transparency: bool,
    reasoning: bool,
    output_dir: Path,
    max_parse_retries: int,
    max_transport_retries: int,
    transport_backoff_s: float,
    parallelism: int,
) -> Dict[str, Any]:
    model_safe = safe_model_name(model)
    session_dir = output_dir / model_safe / condition.label / f"session_{session_idx}"
    session_dir.mkdir(parents=True, exist_ok=True)

    game = PerPlayerPromptPublicGoodsGame(
        num_players=num_players,
        endowment=endowment,
        multiplier=multiplier,
        transparency=transparency,
        reasoning=reasoning,
        prompt_condition="",
        player_prompt_conditions=condition.player_prompt_conditions,
    )
    state = GameState(max_rounds=n_rounds)

    connectors = [build_llm_connector(model, temperature, provider) for _ in range(num_players)]
    conversations: List[List[Dict[str, str]]] = [[] for _ in range(num_players)]

    round_records: List[Dict[str, Any]] = []
    total_payoffs = [0.0] * num_players
    model_returned_seen: set[str] = set()
    incomplete = False
    abort_reason: Optional[str] = None
    t_session_start = time.time()

    for round_num in range(1, n_rounds + 1):
        round_dir = session_dir / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)

        player_records: List[Dict[str, Any]] = []
        round_actions: List[float] = []

        prompts = [game.get_prompt(state, player_id, n_rounds) for player_id in range(num_players)]

        def run_player_turn(player_id: int) -> Dict[str, Any]:
            prompt = prompts[player_id]
            t0 = time.time()
            turn = _player_turn(
                connectors[player_id],
                conversations[player_id],
                prompt,
                game,
                max_parse_retries=max_parse_retries,
                max_transport_retries=max_transport_retries,
                transport_backoff_s=transport_backoff_s,
            )

            response = turn["response"]
            parsed_action = turn["parsed_action"]
            parse_error = turn["parse_error"]
            retries = turn["retries"]
            transport_retries = turn.get("transport_retries", 0)
            meta = turn["meta"] or {}
            response_sha = sha256_text(response)
            model_returned = (meta.get("model") if isinstance(meta, dict) else None) or model
            model_returned_seen.add(model_returned)
            elapsed = round(time.time() - t0, 2)
            player_profile = condition.player_profiles[player_id]

            player_file = round_dir / f"player_{player_id + 1}.txt"
            _write_player_record(
                player_file,
                model_requested=model,
                model_returned=model_returned,
                provider=provider,
                condition_label=condition.label,
                player_profile=player_profile,
                session_idx=session_idx,
                round_num=round_num,
                player_id=player_id,
                temperature=temperature,
                parsed_action=parsed_action,
                parse_error=parse_error,
                retries=retries,
                transport_retries=transport_retries,
                response_sha256=response_sha,
                prompt=prompt,
                response=response,
            )

            return {
                "player_id": player_id + 1,
                "player_profile": player_profile,
                "parsed_action": parsed_action,
                "parse_error": parse_error,
                "retries": retries,
                "transport_retries": transport_retries,
                "response_sha256": response_sha,
                "response_path": str(player_file.relative_to(output_dir)),
                "chat_id": meta.get("chat_id") if isinstance(meta, dict) else None,
                "model_returned": model_returned,
                "usage": meta.get("usage") if isinstance(meta, dict) else None,
                "elapsed_s": elapsed,
                "model_returned_for_seen": model_returned,
            }

        turn_results: List[Optional[Dict[str, Any]]] = [None] * num_players
        if parallelism <= 1:
            for player_id in range(num_players):
                try:
                    turn_results[player_id] = run_player_turn(player_id)
                except Exception as exc:
                    abort_reason = (
                        f"transport error model={model} condition={condition.label} "
                        f"session={session_idx} round={round_num} player={player_id + 1}: {exc!r}"
                    )
                    print(f"FAIL {abort_reason}")
                    incomplete = True
                    break
        else:
            max_workers = min(parallelism, num_players)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_player_turn, player_id): player_id
                    for player_id in range(num_players)
                }
                for future in as_completed(futures):
                    player_id = futures[future]
                    try:
                        turn_results[player_id] = future.result()
                    except Exception as exc:
                        abort_reason = (
                            f"transport error model={model} condition={condition.label} "
                            f"session={session_idx} round={round_num} player={player_id + 1}: {exc!r}"
                        )
                        print(f"FAIL {abort_reason}")
                        incomplete = True

        for player_id, result in enumerate(turn_results):
            if result is None:
                if abort_reason is None:
                    abort_reason = (
                        f"missing result model={model} condition={condition.label} "
                        f"session={session_idx} round={round_num} player={player_id + 1}"
                    )
                    print(f"FAIL {abort_reason}")
                incomplete = True
                break

            model_returned_seen.add(result.pop("model_returned_for_seen"))
            player_records.append(result)
            parsed_action = result["parsed_action"]

            if parsed_action is None:
                abort_reason = (
                    f"parse error after retries model={model} condition={condition.label} "
                    f"session={session_idx} round={round_num} player={player_id + 1}"
                )
                print(f"FAIL {abort_reason}")
                incomplete = True
                break

            round_actions.append(float(parsed_action))
            print(
                f"OK model={model} condition={condition.label} session={session_idx} "
                f"round={round_num} player={player_id + 1} profile={result['player_profile']} "
                f"parsed={parsed_action} sha={result['response_sha256'][:8]} "
                f"retries={result['retries']} elapsed={result['elapsed_s']}s"
            )

        if incomplete:
            round_records.append({
                "round": round_num,
                "actions": round_actions,
                "payoffs": None,
                "players": player_records,
                "incomplete": True,
            })
            break

        payoffs = game.compute_payoffs(round_actions)
        for i, payoff in enumerate(payoffs):
            total_payoffs[i] += payoff
        state.add_round(round_actions)
        round_records.append({
            "round": round_num,
            "actions": round_actions,
            "payoffs": payoffs,
            "players": player_records,
            "incomplete": False,
        })

    metrics = compute_metrics(round_records, endowment)
    session_summary = {
        "experiment": "2.2",
        "mode": "human_like_repeated_pgg_dynamics",
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
        "transparency": transparency,
        "reasoning": reasoning,
        "prompt_condition": condition.prompt_condition,
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


def run_grid(
    *,
    models: List[str],
    provider: str,
    temperature: float,
    n_sessions: int,
    start_session: int,
    num_players: int,
    endowment: float,
    multiplier: float,
    n_rounds: int,
    transparency: bool,
    reasoning: bool,
    output_dir: Path,
    condition_labels: List[str],
    max_parse_retries: int,
    max_transport_retries: int,
    transport_backoff_s: float,
    skip_existing: bool,
    parallelism: int,
) -> None:
    load_dotenv()
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = [condition_spec(label, num_players) for label in condition_labels]
    config = {
        "experiment": "2.2",
        "mode": "human_like_repeated_pgg_dynamics",
        "models": models,
        "provider": provider,
        "temperature": temperature,
        "n_sessions": n_sessions,
        "start_session": start_session,
        "num_players": num_players,
        "endowment": endowment,
        "multiplier": multiplier,
        "n_rounds": n_rounds,
        "transparency": transparency,
        "reasoning": reasoning,
        "condition_labels": condition_labels,
        "conditions": [asdict(c) for c in conditions],
        "max_parse_retries": max_parse_retries,
        "max_transport_retries": max_transport_retries,
        "transport_backoff_s": transport_backoff_s,
        "skip_existing": skip_existing,
        "parallelism": parallelism,
        "env_provider": os.getenv("LLM_PROVIDER"),
        "openrouter_max_tokens": os.getenv("OPENROUTER_MAX_TOKENS"),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    sessions: List[Dict[str, Any]] = []
    total = len(models) * len(conditions) * n_sessions
    done = 0
    for model in models:
        for condition in conditions:
            for session_idx in range(start_session, start_session + n_sessions):
                done += 1
                session_dir = output_dir / safe_model_name(model) / condition.label / f"session_{session_idx}"
                summary_path = session_dir / "session_summary.json"
                print(
                    f"\n[{done}/{total}] model={model} condition={condition.label} "
                    f"session={session_idx}"
                )
                if skip_existing and summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text())
                        sessions.append(summary)
                        _save_summary(output_dir, config, sessions)
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
                    transparency=transparency,
                    reasoning=reasoning,
                    output_dir=output_dir,
                    max_parse_retries=max_parse_retries,
                    max_transport_retries=max_transport_retries,
                    transport_backoff_s=transport_backoff_s,
                    parallelism=parallelism,
                )
                sessions.append(summary)
                _save_summary(output_dir, config, sessions)
                _print_session_line(summary)

    summary_sessions = _load_summary_sessions(output_dir)
    print(f"\nDone. {len(sessions)} touched session(s), {len(summary_sessions)} total. Output: {output_dir}")
    print_table(summary_sessions, models, condition_labels)


def _load_summary_sessions(output_dir: Path) -> List[Dict[str, Any]]:
    path = output_dir / "run_summary.json"
    if not path.exists():
        return []
    try:
        sessions = json.loads(path.read_text()).get("sessions", [])
    except Exception:
        return []
    return sessions if isinstance(sessions, list) else []


def _print_session_line(summary: Dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    print(
        "  metrics "
        f"R1={metrics.get('r1_mean')} "
        f"Rlast={metrics.get('r_last_mean')} "
        f"slope={metrics.get('slope')} "
        f"zero_last={metrics.get('final_zero_rate')} "
        f"D={metrics.get('distance_to_no_punishment_human')}"
    )


def print_table(
    sessions: List[Dict[str, Any]],
    models: List[str],
    condition_labels: List[str],
) -> None:
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for s in sessions:
        if s.get("incomplete"):
            continue
        grouped.setdefault((s.get("model_requested"), s.get("condition_label")), []).append(s)

    print("\nMean distance to no-punishment human target (lower is better)")
    col_w = 16
    header = f"{'model':<28}" + "".join(f"{c[:col_w-1]:>{col_w}}" for c in condition_labels)
    print(header)
    print("-" * len(header))
    for model in models:
        short = model.split("/")[-1][:26]
        row = f"{short:<28}"
        for condition in condition_labels:
            vals = [
                s.get("metrics", {}).get("distance_to_no_punishment_human")
                for s in grouped.get((model, condition), [])
            ]
            vals = [float(v) for v in vals if v is not None]
            cell = f"{sum(vals) / len(vals):.2f}" if vals else "?"
            row += f"{cell:>{col_w}}"
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2.2: human-like repeated PGG dynamics."
    )
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--provider", type=str, default=os.getenv("LLM_PROVIDER", "auto"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n-sessions", type=int, default=4)
    parser.add_argument("--start-session", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--endowment", type=float, default=20.0)
    parser.add_argument("--multiplier", type=float, default=1.6)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--no-transparency", action="store_true")
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument(
        "--condition-labels",
        type=str,
        nargs="+",
        default=DEFAULT_CONDITIONS,
        help=f"Conditions to run. Available: {available_conditions()}",
    )
    parser.add_argument("--max-parse-retries", type=int, default=1)
    parser.add_argument("--max-transport-retries", type=int, default=4)
    parser.add_argument("--transport-backoff-s", type=float, default=5.0)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/exp2_2_human_dynamics"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_grid(
        models=args.models,
        provider=args.provider,
        temperature=args.temperature,
        n_sessions=args.n_sessions,
        start_session=args.start_session,
        num_players=args.num_players,
        endowment=args.endowment,
        multiplier=args.multiplier,
        n_rounds=args.n_rounds,
        transparency=not args.no_transparency,
        reasoning=not args.no_reasoning,
        output_dir=args.output,
        condition_labels=args.condition_labels,
        max_parse_retries=args.max_parse_retries,
        max_transport_retries=args.max_transport_retries,
        transport_backoff_s=args.transport_backoff_s,
        skip_existing=args.skip_existing,
        parallelism=args.parallelism,
    )
