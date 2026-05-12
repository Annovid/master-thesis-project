import argparse
import os
from pathlib import Path

import yaml

from .games.base_game import Game
from .games.pd_game import PrisonersDilemma
from .games.public_goods_game import PublicGoodsGame

class Config:
    """Configuration for the simulation."""

    def __init__(
        self,
        game: Game,
        agent_configs: list[dict],
        rounds: int,
        transparency: bool = False,
        llm_provider: str | None = None,
    ) -> None:
        self.game = game
        self.agent_configs = agent_configs
        self.rounds = rounds
        self.transparency = transparency
        self.llm_provider = llm_provider

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    with config_path.open("r") as f:
        data = yaml.safe_load(f)

    game_type = data.get("game")
    if game_type is None:
        raise ValueError("Missing 'game' in config")
    rounds = data.get("rounds")
    if rounds is None:
        raise ValueError("Missing 'rounds' in config")
    transparency = data.get("transparency", False)
    llm_provider = data.get("llm_provider") or os.getenv("LLM_PROVIDER") or "auto"

    if game_type == "pd":
        payoff_matrix = data.get("payoff_matrix")
        if payoff_matrix is None:
            raise ValueError("Missing 'payoff_matrix' for PD game")
        # Convert to Payoff dataclasses
        from .game.rules import Payoff
        payoff_matrix = [
            [Payoff(row[0][0], row[0][1]), Payoff(row[1][0], row[1][1])]
            for row in payoff_matrix
        ]
        game = PrisonersDilemma(payoff_matrix)
    agent_configs = data.get("agents")
    if agent_configs is None:
        raise ValueError("Missing 'agents' in config")

    if game_type == "pd":
        payoff_matrix = data.get("payoff_matrix")
        if payoff_matrix is None:
            raise ValueError("Missing 'payoff_matrix' for PD game")
        # Convert to Payoff dataclasses
        from .game.rules import Payoff
        payoff_matrix = [
            [Payoff(row[0][0], row[0][1]), Payoff(row[1][0], row[1][1])]
            for row in payoff_matrix
        ]
        game = PrisonersDilemma(payoff_matrix)
    elif game_type == "pg":
        num_players = data.get("num_players")
        if num_players is None:
            raise ValueError("Missing 'num_players' for PG game")
        endowment = data.get("endowment")
        if endowment is None:
            raise ValueError("Missing 'endowment' for PG game")
        multiplier = data.get("multiplier")
        if multiplier is None:
            raise ValueError("Missing 'multiplier' for PG game")
        game = PublicGoodsGame(num_players, endowment, multiplier, transparency, agent_configs)
    else:
        raise ValueError(f"Unknown game: {game_type}")

    if len(agent_configs) != game.num_players:
        raise ValueError(f"Number of agents {len(agent_configs)} does not match game players {game.num_players}")

    return Config(game, agent_configs, rounds, transparency, llm_provider)
