from dataclasses import dataclass
from typing import List

from ..constants import COOPERATE, DEFECT

Action = int  # 0: Cooperate, 1: Defect

from typing import List as TypingList

@dataclass
class RoundActions:
    """Actions taken by players in a round."""
    actions: TypingList[Action]

class GameState:
    """Represents the current state of the game."""

    def __init__(self, max_rounds: int) -> None:
        """Initialize game state with maximum rounds."""
        self.max_rounds = max_rounds
        self.current_round = 0
        self.history: TypingList[RoundActions] = []

    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return self.current_round >= self.max_rounds

    def add_round(self, actions: TypingList[Action]) -> None:
        """Add a round's actions to history."""
        self.history.append(RoundActions(actions))
        self.current_round += 1

    def get_history(self) -> TypingList[RoundActions]:
        """Get a copy of the action history."""
        return self.history.copy()