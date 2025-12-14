import logging
from typing import List

from .base_game import Game, Action
from ..constants import COOPERATE, DEFECT, ACTION_NAMES
from ..game.rules import Payoff
from ..game.state import GameState

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = """
You are playing the Prisoner's Dilemma game for {max_rounds} rounds. The payoff matrix is:
- Both Cooperate: (3, 3)
- You Cooperate, Opponent Defects: (0, 5)
- You Defect, Opponent Cooperates: (5, 0)
- Both Defect: (1, 1)

History of previous rounds: {history}

Choose your action: Cooperate or Defect. Respond with only 'Cooperate' or 'Defect'.
"""

class PrisonersDilemma(Game):
    """Prisoner's Dilemma game."""

    def __init__(self, payoff_matrix: List[List[Payoff]] | None = None) -> None:
        if payoff_matrix is None:
            # Default PD matrix
            self.payoff_matrix = [
                [Payoff(3, 3), Payoff(0, 5)],
                [Payoff(5, 0), Payoff(1, 1)]
            ]
        else:
            self.payoff_matrix = payoff_matrix

    @property
    def num_players(self) -> int:
        return 2

    def get_prompt(self, state: GameState, player_id: int, max_rounds: int) -> str:
        """Generate prompt for the player."""
        history_str = self._format_history(state, player_id)
        return DEFAULT_PROMPT_TEMPLATE.format(history=history_str, max_rounds=max_rounds)

    def parse_action(self, response: str) -> Action:
        """Parse response into action."""
        resp = response.strip().lower()
        if "cooperate" in resp:
            return COOPERATE
        elif "defect" in resp:
            return DEFECT
        else:
            logger.warning(f"Unclear response: {response}, defaulting to cooperate")
            return COOPERATE

    def compute_payoffs(self, actions: List[Action]) -> List[float]:
        """Compute payoffs."""
        a1, a2 = actions
        payoff = self.payoff_matrix[a1][a2]
        return [payoff.player1, payoff.player2]

    def validate_action(self, action: Action) -> bool:
        """Validate action."""
        return action in (COOPERATE, DEFECT)

    def _format_history(self, state: GameState, player_id: int) -> str:
        """Format history for the player."""
        if not state.history:
            return "No previous rounds."
        lines = []
        for i, round_actions in enumerate(state.history, 1):
            actions = round_actions.actions
            my_action = ACTION_NAMES[actions[player_id]]
            opp_action = ACTION_NAMES[actions[1 - player_id]]
            lines.append(f"Round {i}: You {my_action}, Opponent {opp_action}")
        return "\n".join(lines)