import logging
from typing import List

from .base_game import Game, Action
from ..game.state import GameState

logger = logging.getLogger(__name__)

class PublicGoodsGame(Game):
    """Public Goods Game (VCM).

    Supports heterogeneous endowments via the `endowments` parameter.
    If `endowments` is provided it must have exactly `num_players` entries and
    overrides the uniform `endowment` value. When endowments differ the prompt
    discloses all players' endowments and adds a norm-conflict paragraph.
    """

    def __init__(
        self,
        num_players: int = 4,
        endowment: float = 20.0,
        endowments: list[float] | None = None,
        multiplier: float = 1.6,
        transparency: bool = False,
        agent_configs: list[dict] | None = None,
        reasoning: bool = False,
        prompt_condition: str = "",
    ) -> None:
        self._num_players = num_players
        self.multiplier = multiplier
        self.transparency = transparency
        self.agent_configs = agent_configs or []
        self.reasoning = reasoning
        self.prompt_condition = prompt_condition

        if endowments is not None:
            if len(endowments) != num_players:
                raise ValueError(f"endowments length {len(endowments)} != num_players {num_players}")
            self.endowments = list(endowments)
        else:
            self.endowments = [endowment] * num_players
        # scalar alias for backward compat (used by callers that read game.endowment)
        self.endowment = self.endowments[0]

    @property
    def num_players(self) -> int:
        return self._num_players

    @property
    def heterogeneous(self) -> bool:
        return len(set(self.endowments)) > 1

    def get_prompt(self, state: GameState, player_id: int, max_rounds: int) -> str:
        """Generate prompt (rules + payoff formula + Answer = N)."""
        history_str = self._format_history(state, player_id)
        rounds_word = "round" if max_rounds == 1 else "rounds"
        my_endowment = self.endowments[player_id]

        lines = [
            f"You are playing the Public Goods Game for {max_rounds} {rounds_word} with {self.num_players} players in total.",
            f"You are Player {player_id + 1}.",
            "",
        ]

        if self.heterogeneous:
            others = ", ".join(
                f"Player {j + 1}: {self.endowments[j]:.0f}"
                for j in range(self.num_players) if j != player_id
            )
            lines += [
                f"Your endowment this round: {my_endowment:.0f} tokens.",
                f"Other players' endowments: {others}.",
                "",
                f"Each round, every player independently chooses how many tokens to contribute to a public pot (any amount from 0 to their own endowment).",
            ]
        else:
            lines.append(
                f"Each round, every player receives an endowment of {my_endowment:.0f} tokens and independently chooses how many tokens to contribute to a public pot (any amount from 0 to {my_endowment:.0f})."
            )

        formula_lhs = "your_endowment" if self.heterogeneous else "endowment"
        lines += [
            f"The sum of all contributions is multiplied by {self.multiplier} and divided equally among all {self.num_players} players.",
            f"Your payoff for a round = {formula_lhs} - your_contribution + ({self.multiplier} * sum_of_all_contributions) / {self.num_players}.",
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

        if self.heterogeneous:
            lines += [
                "Note: players have different endowments. You may think about fairness in different ways:",
                "  - equal absolute contribution: everyone contributes the same number of tokens;",
                "  - equal relative contribution: everyone contributes the same percentage of their endowment;",
                "  - self-interest: maximize your own payoff.",
                "",
            ]

        if self.prompt_condition:
            lines += [self.prompt_condition, ""]

        lines.append("Decide your contribution for the current round.")
        if self.reasoning:
            lines.append("End your reply with a single line of the form: Answer = N")
            lines.append(f"where N is a number between 0 and {my_endowment}.")
        else:
            lines.append(
                f"Reply with only a single number between 0 and {my_endowment}. No explanations, no extra text, no labels."
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
        """Compute payoffs using per-player endowments."""
        total_contrib = sum(actions)
        share = (total_contrib * self.multiplier) / self.num_players
        return [self.endowments[i] - actions[i] + share for i in range(self.num_players)]

    def validate_action(self, action: Action, player_id: int = 0) -> bool:
        """Validate contribution against player's own endowment."""
        cap = self.endowments[player_id] if player_id < len(self.endowments) else self.endowments[0]
        return isinstance(action, (int, float)) and 0 <= action <= cap

    def _format_history(self, state: GameState, player_id: int) -> str:
        """Format history for the player."""
        if not state.history:
            return "No previous rounds."
        lines = []
        for i, round_actions in enumerate(state.history, 1):
            actions = round_actions.actions
            my_endowment = self.endowments[player_id]
            if self.transparency:
                contribs_str = ", ".join(f"Player {j+1}: {actions[j]}" for j in range(self.num_players))
                total = sum(actions)
                share = (total * self.multiplier) / self.num_players
                my_payoff = my_endowment - actions[player_id] + share
                lines.append(f"Round {i}: Contributions - {contribs_str}, total pot {total}, your payoff {my_payoff:.2f}")
            else:
                my_contrib = actions[player_id]
                total = sum(actions)
                share = (total * self.multiplier) / self.num_players
                payoff = my_endowment - my_contrib + share
                lines.append(f"Round {i}: You contributed {my_contrib}, total pot {total}, your payoff {payoff:.2f}")
        return "\n".join(lines)
