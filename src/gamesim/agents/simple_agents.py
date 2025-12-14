import random
from typing import Any

from .base import Agent
from ..constants import COOPERATE, DEFECT
from ..game.state import GameState
from ..games.base_game import Game, Action

class AlwaysCooperate(Agent):
    """Agent that always cooperates in PD or contributes half in PG."""

    def act(self, state: GameState, game: Game, player_id: int) -> tuple[Action, dict[str, Any]]:
        if hasattr(game, 'endowment'):  # PG
            action = game.endowment / 2
        else:
            action = COOPERATE
        return action, {"type": "simple", "strategy": "always_cooperate"}

class AlwaysDefect(Agent):
    """Agent that always defects in PD or contributes 0 in PG."""

    def act(self, state: GameState, game: Game, player_id: int) -> tuple[Action, Any]:
        if hasattr(game, 'endowment'):  # PG
            action = 0.0
        else:
            action = DEFECT
        return action, {"type": "simple", "strategy": "always_defect"}

class RandomAgent(Agent):
    """Agent that chooses actions randomly."""

    def act(self, state: GameState, game: Game, player_id: int) -> tuple[Action, Any]:
        if hasattr(game, 'endowment'):  # PG
            action = random.uniform(0, game.endowment)
        else:
            action = random.choice([COOPERATE, DEFECT])
        return action, {"type": "simple", "strategy": "random", "action": action}