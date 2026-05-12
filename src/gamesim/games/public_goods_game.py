import logging
from typing import List

from .base_game import Game, Action
from ..game.state import GameState

logger = logging.getLogger(__name__)

class PublicGoodsGame(Game):
    """Public Goods Game (VCM)."""

    def __init__(self, num_players: int = 4, endowment: float = 20.0, multiplier: float = 1.6, transparency: bool = False, agent_configs: list[dict] | None = None, reasoning: bool = False, prompt_condition: str = "") -> None:
        self._num_players = num_players
        self.endowment = endowment
        self.multiplier = multiplier
        self.transparency = transparency
        self.agent_configs = agent_configs or []
        self.reasoning = reasoning
        self.prompt_condition = prompt_condition

    @property
    def num_players(self) -> int:
        return self._num_players

    def get_prompt(self, state: GameState, player_id: int, max_rounds: int) -> str:
        """Generate a neutral prompt (rules + payoff formula + Answer = N)."""
        history_str = self._format_history(state, player_id)
        rounds_word = "round" if max_rounds == 1 else "rounds"

        lines = [
            f"You are playing the Public Goods Game for {max_rounds} {rounds_word} with {self.num_players} players in total.",
            f"You are Player {player_id + 1}.",
            "",
            f"Each round, every player receives an endowment of {self.endowment} tokens and independently chooses how many tokens to contribute to a public pot (any amount from 0 to {self.endowment}).",
            f"The sum of all contributions is multiplied by {self.multiplier} and divided equally among all {self.num_players} players.",
            f"Your payoff for a round = endowment - your_contribution + ({self.multiplier} * sum_of_all_contributions) / {self.num_players}.",
            f"Your final score is the sum of your payoffs across all {max_rounds} {rounds_word}.",
        ]

        if self.transparency:
            lines.append("After each round, the individual contributions of all players are revealed to everyone.")
        else:
            lines.append("Between rounds, only your own contribution and round payoff are revealed to you.")

        lines += [
            "",
            "History of previous rounds:",
            history_str,
            "",
        ]
        if self.prompt_condition:
            lines += [self.prompt_condition, ""]
        lines.append("Decide your contribution for the current round.")
        if self.reasoning:
            lines.append(
                f"Think step by step about your decision: describe your reasoning, what you expect from the other players, and why you pick this contribution."
            )
            lines.append(
                f"On the last line of your reply, write exactly 'Answer = N' where N is your chosen contribution (a single number between 0 and {self.endowment})."
            )
        else:
            lines.append(
                f"Reply with only a single number between 0 and {self.endowment}. No explanations, no extra text, no labels."
            )
        return "\n".join(lines)

    def parse_action(self, response: str, additional_info: dict | None = None) -> Action:
        """Parse response into contribution.

        Primary form: the entire reply is a single number (possibly with markdown
        wrappers or trailing punctuation). Falls back to "Answer = N" pattern.
        """
        import re

        lines = [line.strip() for line in response.splitlines() if line.strip()]
        logger.info(f"Parsed lines: {lines}")

        bare_number = re.compile(r"^[-+]?\d*\.?\d+$")
        answer_pattern = re.compile(r"answer\s*=\s*([-+]?\d*\.?\d+)", re.IGNORECASE)

        for line in reversed(lines):
            cleaned = line.strip('* _`"\'').strip().rstrip('.,;:')
            if bare_number.match(cleaned):
                try:
                    return float(cleaned)
                except ValueError as e:
                    logger.error(f"Error parsing bare number '{cleaned}': {e}")
                    break

        for line in reversed(lines):
            cleaned = line.strip('* _`').strip()
            match = answer_pattern.search(cleaned)
            if match:
                num_str = match.group(1).rstrip('.')
                logger.info(f"Num str: '{num_str}'")
                try:
                    return float(num_str)
                except ValueError as e:
                    logger.error(f"Error parsing num_str: {e}")
                    break

        # Save error info
        error_info = {
            "model": additional_info.get("model") if additional_info else "unknown",
            "prompt": additional_info.get("prompt") if additional_info else "unknown",
            "response": response,
            "temperature": additional_info.get("temperature") if additional_info else "unknown",
            "reasoning": additional_info.get("reasoning") if additional_info else False,
            "chat_id": additional_info.get("chat_id") if additional_info else "unknown",
            "error": "Could not parse Answer from response"
        }
        import json
        from datetime import datetime
        from pathlib import Path
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        error_path = Path("artifacts") / "errors" / f"error_{timestamp}.json"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with error_path.open("w") as f:
            json.dump(error_info, f, indent=2)
        raise ValueError(f"Could not parse Answer from response: {response}")

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
            if self.transparency:
                # Show all contributions
                contribs_str = ", ".join(f"Player {j+1}: {actions[j]}" for j in range(self.num_players))
                total = sum(actions)
                share = (total * self.multiplier) / self.num_players
                my_payoff = self.endowment - actions[player_id] + share
                lines.append(f"Round {i}: Contributions - {contribs_str}, total pot {total}, your payoff {my_payoff:.2f}")
            else:
                # Show only own
                my_contrib = actions[player_id]
                total = sum(actions)
                share = (total * self.multiplier) / self.num_players
                payoff = self.endowment - my_contrib + share
                lines.append(f"Round {i}: You contributed {my_contrib}, total pot {total}, your payoff {payoff:.2f}")
        return "\n".join(lines)
