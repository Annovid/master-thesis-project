"""Experiment 1.2 — decay of cooperation in full 10-round PGG (pilot).

Each session: 4 copies of the same model play a 10-round PGG against each
other with transparency=True (everyone sees individual contributions after
each round). Per-player full conversation history maintained throughout the
session.

Pilot defaults: n_sessions=2 per model, T=0.7. On parse error the runner
retries once with the same conversation state; if retry also fails, the
session aborts and is recorded as incomplete.

Output layout:
  outputs/exp1_2_pilot/
    run_config.json
    run_summary.json
    <model_safe>/
      session_<s>/
        session_summary.json
        round_<r>/
          player_<p>.txt    # metadata + prompt + response
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.gamesim.game.state import GameState
from src.gamesim.games.public_goods_game import PublicGoodsGame
from src.services.single_request import build_connector


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def _try_parse(game: PublicGoodsGame, response: str) -> tuple[Optional[float], Optional[str]]:
    try:
        action = game.parse_action(response)
        return float(action), None
    except ValueError as e:
        return None, str(e)


def _player_turn(
    connector,
    conversation: List[Dict[str, str]],
    prompt: str,
    game: PublicGoodsGame,
    max_parse_retries: int = 1,
    max_transport_retries: int = 4,
    transport_backoff_s: float = 5.0,
) -> Dict[str, Any]:
    """Issue one chat completion for a player.

    Returns dict with: response, parsed_action, parse_error, meta, retries,
    transport_retries. On unrecoverable transport failure, raises so caller
    aborts the session.

    Transport errors (any exception from `query_conversation`) trigger
    exponential backoff retry up to `max_transport_retries` times. Parse
    errors trigger up to `max_parse_retries` retries re-issuing the same
    user prompt with the same conversation state.
    """
    pre_len = len(conversation)
    conversation.append({"role": "user", "content": prompt})

    parse_retries = 0
    transport_retries = 0
    last_exc: Optional[Exception] = None

    while True:
        try:
            response, meta = connector.query_conversation(conversation)
            last_exc = None
        except Exception as exc:
            last_exc = exc
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

        parsed, parse_err = _try_parse(game, response)
        if parsed is not None:
            conversation.append({"role": "assistant", "content": response})
            return {
                "response": response,
                "parsed_action": parsed,
                "parse_error": None,
                "meta": meta,
                "retries": parse_retries,
                "transport_retries": transport_retries,
            }

        if parse_retries >= max_parse_retries:
            del conversation[pre_len:]
            return {
                "response": response,
                "parsed_action": None,
                "parse_error": parse_err,
                "meta": meta,
                "retries": parse_retries,
                "transport_retries": transport_retries,
            }
        parse_retries += 1


def _write_player_record(
    path: Path,
    *,
    model_requested: str,
    model_returned: str,
    provider: str,
    session_idx: int,
    round_num: int,
    player_id: int,
    temperature: float,
    parsed_action: Optional[float],
    parse_error: Optional[str],
    retries: int,
    response_sha256: str,
    prompt: str,
    response: str,
) -> None:
    with path.open("w") as f:
        f.write(f"MODEL_REQUESTED: {model_requested}\n")
        f.write(f"MODEL_RETURNED: {model_returned}\n")
        f.write(f"PROVIDER: {provider}\n")
        f.write(f"SESSION: {session_idx}\n")
        f.write(f"ROUND: {round_num}\n")
        f.write(f"PLAYER: {player_id + 1}\n")
        f.write(f"TEMPERATURE: {temperature}\n")
        f.write(f"PARSED_ACTION: {parsed_action}\n")
        f.write(f"PARSE_ERROR: {parse_error}\n")
        f.write(f"RETRIES: {retries}\n")
        f.write(f"RESPONSE_SHA256: {response_sha256}\n\n")
        f.write("PROMPT:\n")
        f.write(prompt)
        f.write("\n\nRESPONSE:\n")
        f.write(response)
        f.write("\n")


def _save_summary(output_dir: Path, config_snapshot: Dict[str, Any], sessions: List[Dict[str, Any]]) -> None:
    summary_path = output_dir / "run_summary.json"
    existing: List[Dict[str, Any]] = []
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text()).get("sessions", [])
        except Exception:
            existing = []

    new_keys = {(s["model_requested"], s["session"]) for s in sessions}
    kept = [s for s in existing if (s.get("model_requested"), s.get("session")) not in new_keys]
    merged = kept + sessions
    merged.sort(key=lambda s: (s.get("model_requested", ""), s.get("session", 0)))
    summary = {**config_snapshot, "sessions": merged}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def run_session(
    model: str,
    provider: str,
    temperature: float,
    session_idx: int,
    num_players: int,
    endowment: float,
    multiplier: float,
    n_rounds: int,
    transparency: bool,
    reasoning: bool,
    output_dir: Path,
    prompt_label: str = "neutral",
    prompt_condition: str = "",
    endowments: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Run one full game with `num_players` copies of `model`."""
    model_safe = safe_model_name(model)
    session_dir = output_dir / model_safe / f"session_{session_idx}"
    session_dir.mkdir(parents=True, exist_ok=True)

    game = PublicGoodsGame(
        num_players=num_players,
        endowment=endowment,
        endowments=endowments,
        multiplier=multiplier,
        transparency=transparency,
        reasoning=reasoning,
        prompt_condition=prompt_condition,
    )
    state = GameState(max_rounds=n_rounds)

    # One connector + one conversation per player. Connectors are independent
    # so each player has its own chat session on the provider side too.
    connectors = [
        build_connector({"model": model, "temperature": temperature}, default_provider=provider)
        for _ in range(num_players)
    ]
    conversations: List[List[Dict[str, str]]] = [[] for _ in range(num_players)]

    round_records: List[Dict[str, Any]] = []
    total_payoffs = [0.0] * num_players
    model_returned_seen: set = set()
    incomplete = False
    abort_reason: Optional[str] = None

    t_session_start = time.time()

    for round_num in range(1, n_rounds + 1):
        round_dir = session_dir / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)

        player_records: List[Dict[str, Any]] = []
        round_actions: List[float] = []

        for player_id in range(num_players):
            prompt = game.get_prompt(state, player_id, n_rounds)
            t0 = time.time()
            try:
                turn = _player_turn(
                    connectors[player_id],
                    conversations[player_id],
                    prompt,
                    game,
                )
            except Exception as exc:
                abort_reason = (
                    f"transport error model={model} session={session_idx} "
                    f"round={round_num} player={player_id + 1}: {exc!r}"
                )
                print(f"FAIL {abort_reason}")
                incomplete = True
                break

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

            player_file = round_dir / f"player_{player_id + 1}.txt"
            _write_player_record(
                player_file,
                model_requested=model,
                model_returned=model_returned,
                provider=provider,
                session_idx=session_idx,
                round_num=round_num,
                player_id=player_id,
                temperature=temperature,
                parsed_action=parsed_action,
                parse_error=parse_error,
                retries=retries,
                response_sha256=response_sha,
                prompt=prompt,
                response=response,
            )

            player_records.append({
                "player_id": player_id + 1,
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
            })

            if parsed_action is None:
                abort_reason = (
                    f"parse error after retries model={model} session={session_idx} "
                    f"round={round_num} player={player_id + 1}"
                )
                print(f"FAIL {abort_reason}")
                incomplete = True
                break

            round_actions.append(parsed_action)
            print(
                f"OK model={model} session={session_idx} round={round_num} "
                f"player={player_id + 1} parsed={parsed_action} sha={response_sha[:8]} "
                f"retries={retries} elapsed={elapsed}s"
            )

        if incomplete:
            # Record the (partial) round so transcripts are not orphaned.
            round_records.append({
                "round": round_num,
                "actions": round_actions,
                "payoffs": None,
                "players": player_records,
                "incomplete": True,
            })
            break

        payoffs = game.compute_payoffs(round_actions)
        for i in range(num_players):
            total_payoffs[i] += payoffs[i]
        state.add_round(round_actions)
        round_records.append({
            "round": round_num,
            "actions": round_actions,
            "payoffs": payoffs,
            "players": player_records,
            "incomplete": False,
        })

    session_summary = {
        "model_requested": model,
        "model_returned_seen": sorted(model_returned_seen),
        "provider": provider,
        "session": session_idx,
        "temperature": temperature,
        "num_players": num_players,
        "endowment": endowment,
        "multiplier": multiplier,
        "n_rounds": n_rounds,
        "transparency": transparency,
        "reasoning": reasoning,
        "prompt_label": prompt_label,
        "prompt_condition": prompt_condition,
        "endowments": game.endowments,
        "incomplete": incomplete,
        "abort_reason": abort_reason,
        "rounds": round_records,
        "total_payoffs": total_payoffs if not incomplete else None,
        "session_path": str(session_dir.relative_to(output_dir)),
        "session_elapsed_s": round(time.time() - t_session_start, 2),
    }

    (session_dir / "session_summary.json").write_text(
        json.dumps(session_summary, indent=2, ensure_ascii=False)
    )
    return session_summary


def run_experiment(
    models: List[str],
    provider: str,
    temperature: float,
    n_sessions: int,
    num_players: int,
    endowment: float,
    multiplier: float,
    n_rounds: int,
    transparency: bool,
    reasoning: bool,
    output_dir: Path,
    start_session: int = 0,
    prompt_label: str = "neutral",
    prompt_condition: str = "",
) -> Dict[str, Any]:
    load_dotenv()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
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
        "prompt_label": prompt_label,
        "prompt_condition": prompt_condition,
        "env_provider": os.getenv("LLM_PROVIDER"),
        "openrouter_max_tokens": os.getenv("OPENROUTER_MAX_TOKENS"),
        "openai_max_tokens": os.getenv("OPENAI_MAX_TOKENS"),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config_snapshot, indent=2, ensure_ascii=False)
    )

    sessions: List[Dict[str, Any]] = []
    for model in models:
        for s in range(start_session, start_session + n_sessions):
            summary = run_session(
                model=model,
                provider=provider,
                temperature=temperature,
                session_idx=s,
                num_players=num_players,
                endowment=endowment,
                multiplier=multiplier,
                n_rounds=n_rounds,
                transparency=transparency,
                reasoning=reasoning,
                output_dir=output_dir,
                prompt_label=prompt_label,
                prompt_condition=prompt_condition,
            )
            sessions.append(summary)
            _save_summary(output_dir, config_snapshot, sessions)

    _save_summary(output_dir, config_snapshot, sessions)
    return {**config_snapshot, "sessions": sessions}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 1.2: full 10-round PGG decay-of-cooperation pilot."
    )
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument(
        "--provider",
        type=str,
        default=os.getenv("LLM_PROVIDER", "auto"),
        help="auto|openai|openrouter|gateway|mock",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-sessions", type=int, default=2)
    parser.add_argument("--start-session", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--endowment", type=float, default=20.0)
    parser.add_argument("--multiplier", type=float, default=1.6)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument(
        "--no-transparency", action="store_true",
        help="Disable post-round revelation of individual contributions.",
    )
    parser.add_argument(
        "--no-reasoning", action="store_true",
        help="Use bare-number prompt instead of CoT + Answer = N.",
    )
    parser.add_argument("--prompt-label", type=str, default="neutral",
        help="Short identifier for the prompt condition (e.g. neutral, persona, social_norms).")
    parser.add_argument("--prompt-condition", type=str, default="",
        help="Extra paragraph inserted into the game prompt before 'Decide your contribution'.")
    parser.add_argument("--output", type=Path, default=Path("outputs/exp1_2_pilot"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        models=args.models,
        provider=args.provider,
        temperature=args.temperature,
        n_sessions=args.n_sessions,
        num_players=args.num_players,
        endowment=args.endowment,
        multiplier=args.multiplier,
        n_rounds=args.n_rounds,
        transparency=not args.no_transparency,
        reasoning=not args.no_reasoning,
        output_dir=args.output,
        start_session=args.start_session,
        prompt_label=args.prompt_label,
        prompt_condition=args.prompt_condition,
    )
