import logging
from typing import List

from .base_game import Game, Action
from ..game.state import GameState

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = """
You are playing the Public Goods Game.

Each player has an endowment of {endowment}.
You can contribute any amount from 0 to {endowment} to the public pot.
The total contributions will be multiplied by {multiplier} and divided equally among all players.

History of previous rounds: {history}

Respond with a number representing your contribution (e.g., 5.0).
"""

class PublicGoodsGame(Game):
    """Public Goods Game."""

    def __init__(self, num_players: int = 4, endowment: float = 10.0, multiplier: float = 2.0) -> None:
        self._num_players = num_players
        self.endowment = endowment
        self.multiplier = multiplier

    @property
    def num_players(self) -> int:
        return self._num_players

    def get_prompt(self, state: GameState, player_id: int) -> str:
        """Generate prompt for the player."""
        history_str = self._format_history(state, player_id)
        return DEFAULT_PROMPT_TEMPLATE.format(
            endowment=self.endowment,
            multiplier=self.multiplier,
            history=history_str
        )

    def parse_action(self, response: str) -> Action:
        """Parse response into contribution."""
        try:
            contrib = float(response.strip())
            return contrib
        except ValueError:
            logger.warning(f"Invalid contribution: {response}, defaulting to 0")
            return 0.0

    def compute_payoffs(self, actions: List[Action]) -> List[float]:
        """Compute payoffs."""
        total_contrib = sum(actions)
        share = (total_contrib * self.multiplier) / self.num_players
        payoffs = []
        for contrib in actions:
            payoff = self.endowment - contrib + share
            payoffs.append(payoff)
        return payoffs

    def validate_action(self, action: Action) -> bool:
        """Validate contribution."""
        return isinstance(action, (int, float)) and 0 <= action <= self.endowment

    def _format_history(self, state: GameState, player_id: int) -> str:
        """Format history for the player."""
        if not state.history:
            return "No previous rounds."
        lines = []
        for i, round_actions in enumerate(state.history, 1):
            actions = round_actions.actions
            my_contrib = actions[player_id]
            total = sum(actions)
            share = (total * self.multiplier) / self.num_players
            payoff = self.endowment - my_contrib + share
            lines.append(f"Round {i}: You contributed {my_contrib}, total pot {total}, your payoff {payoff:.2f}")
        return "\n".join(lines)