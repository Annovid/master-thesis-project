from abc import ABC, abstractmethod
from typing import Any

from ..game.state import GameState
from ..games.base_game import Game, Action

class Agent(ABC):
    """Base class for game agents."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def act(self, state: GameState, game: Game, player_id: int) -> tuple[Action, dict[str, Any]]:
        """Decide an action based on the current game state and game rules.
        
        Returns:
            tuple: (action, details_dict) where details include LLM response, parsing, etc.
        """