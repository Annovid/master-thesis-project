from abc import ABC, abstractmethod
from typing import Any, List

from ..game.state import GameState

Action = Any  # Can be int, float, etc. depending on game

class Game(ABC):
    """Abstract base class for games."""

    @abstractmethod
    def get_prompt(self, state: GameState, player_id: int, max_rounds: int) -> str:
        """Generate prompt for LLM based on current state, player, and total rounds."""

    @abstractmethod
    def parse_action(self, response: str) -> Action:
        """Parse LLM response into an action."""

    @abstractmethod
    def compute_payoffs(self, actions: List[Action]) -> List[float]:
        """Compute payoffs for all players given their actions."""

    @abstractmethod
    def validate_action(self, action: Action) -> bool:
        """Validate if an action is allowed."""

    @property
    @abstractmethod
    def num_players(self) -> int:
        """Number of players in the game."""
