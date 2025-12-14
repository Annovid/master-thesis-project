import argparse
import yaml
from pathlib import Path

from .games.base_game import Game
from .games.pd_game import PrisonersDilemma
from .games.public_goods_game import PublicGoodsGame

class Config:
    """Configuration for the simulation."""

    def __init__(
        self,
        game: Game,
        agent_configs: list[dict],
        rounds: int
    ) -> None:
        self.game = game
        self.agent_configs = agent_configs
        self.rounds = rounds

def load_config() -> Config:
    """Load configuration from YAML file."""
    parser = argparse.ArgumentParser(description="Run game simulation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r") as f:
        data = yaml.safe_load(f)

    game_type = data["game"]
    rounds = data["rounds"]

    if game_type == "pd":
        payoff_matrix = data["payoff_matrix"]
        # Convert to Payoff dataclasses
        from .game.rules import Payoff
        payoff_matrix = [
            [Payoff(row[0][0], row[0][1]), Payoff(row[1][0], row[1][1])]
            for row in payoff_matrix
        ]
        game = PrisonersDilemma(payoff_matrix)
    elif game_type == "pg":
        num_players = data["num_players"]
        endowment = data["endowment"]
        multiplier = data["multiplier"]
        game = PublicGoodsGame(num_players, endowment, multiplier)
    else:
        raise ValueError(f"Unknown game: {game_type}")

    agent_configs = data["agents"]
    if len(agent_configs) != game.num_players:
        raise ValueError(f"Number of agents {len(agent_configs)} does not match game players {game.num_players}")

    return Config(game, agent_configs, rounds)