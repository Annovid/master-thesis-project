import ast
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class Payoff:
    """Represents the payoff for both players."""
    player1: int
    player2: int

PayoffMatrix = List[List[Payoff]]  # 2x2 for PD

class GameRules:
    """Handles the rules and payoff matrix for the game."""

    def __init__(self, payoff_matrix: PayoffMatrix) -> None:
        """Initialize with a 2x2 payoff matrix."""
        if len(payoff_matrix) != 2 or any(len(row) != 2 for row in payoff_matrix):
            raise ValueError("Payoff matrix must be 2x2")
        # Convert tuples to Payoff dataclasses if necessary
        self.payoff_matrix = [
            [p if isinstance(p, Payoff) else Payoff(p[0], p[1]) for p in row]
            for row in payoff_matrix
        ]

    @classmethod
    def from_string(cls, matrix_str: str) -> "GameRules":
        """Create GameRules from a string representation of the matrix."""
        try:
            matrix = ast.literal_eval(matrix_str)
            return cls(matrix)
        except (ValueError, SyntaxError) as e:
            logger.error(f"Invalid matrix format: {e}")
            raise ValueError(f"Invalid matrix format: {e}") from e

    def get_payoff(self, action1: int, action2: int) -> Payoff:
        """Get the payoff for given actions."""
        return self.payoff_matrix[action1][action2]