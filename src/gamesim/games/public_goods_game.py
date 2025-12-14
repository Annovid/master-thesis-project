import logging
from typing import List

from .base_game import Game, Action
from ..game.state import GameState

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = """
You are playing the Public Goods Game for {max_rounds} rounds.

Each player has an endowment of {endowment}.
You can contribute any amount from 0 to {endowment} to the public pot.
The total contributions will be multiplied by {multiplier} and divided equally among all players.

History of previous rounds: {history}

Respond with a number representing your contribution (e.g., 5.0).
"""

class PublicGoodsGame(Game):
    """Public Goods Game."""

    def __init__(self, num_players: int = 4, endowment: float = 10.0, multiplier: float = 2.0, transparency: bool = False, agent_configs: list[dict] | None = None) -> None:
        self._num_players = num_players
        self.endowment = endowment
        self.multiplier = multiplier
        self.transparency = transparency
        self.agent_configs = agent_configs or []

    @property
    def num_players(self) -> int:
        return self._num_players

    def get_prompt(self, state: GameState, player_id: int, max_rounds: int) -> str:
        """Generate prompt for the player."""
        history_str = self._format_history(state, player_id)
        prompt = f"You are playing the Public Goods Game for {max_rounds} rounds.\n\n"
        prompt += f"You are Player {player_id + 1}.\n\n"
        
        if self.transparency:
            # Describe all players
            for j in range(self.num_players):
                agent_config = self.agent_configs[j] if j < len(self.agent_configs) else {}
                agent_type = agent_config.get("type", "unknown")
                if j == player_id:
                    prompt += f"Player {j + 1} is you ({agent_type} agent).\n"
                else:
                    if agent_type == "simple":
                        strategy = agent_config.get("strategy", "unknown")
                        if strategy == "always_cooperate":
                            prompt += f"Player {j + 1} always contributes their full endowment ({self.endowment}) to the public pot.\n"
                        elif strategy == "always_defect":
                            prompt += f"Player {j + 1} never contributes to the public pot (contributes 0).\n"
                        else:
                            prompt += f"Player {j + 1} is a {agent_type} agent with strategy '{strategy}'.\n"
                    elif agent_type == "llm":
                        model = agent_config.get("model", "unknown")
                        prompt += f"Player {j + 1} is an {agent_type} agent using model '{model}'.\n"
                    else:
                        prompt += f"Player {j + 1} is an {agent_type} agent.\n"
        
        prompt += f"\nEach player has an endowment of {self.endowment}.\n"
        prompt += f"You can contribute any amount from 0 to {self.endowment} to the public pot.\n"
        prompt += f"Note: You cannot contribute more than your current endowment.\n"
        prompt += f"The total contributions will be multiplied by {self.multiplier} and divided equally among all players.\n\n"
        prompt += f"History of previous rounds: {history_str}\n\n"
        prompt += "Respond with a number representing your contribution (e.g., 5.0)."
        return prompt

    def parse_action(self, response: str, additional_info: dict | None = None) -> Action:
        """Parse response into contribution."""
        # Parse the last non-empty line for "Answer = num"
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        logger.info(f"Parsed lines: {lines}")
        if lines:
            last_line = lines[-1]
            logger.info(f"Last line: '{last_line}'")
            if last_line.startswith("Answer = "):
                try:
                    num_str = last_line.split("Answer = ")[1].strip()
                    logger.info(f"Num str: '{num_str}'")
                    contrib = float(num_str)
                    return contrib
                except (ValueError, IndexError) as e:
                    logger.error(f"Error parsing num_str: {e}")
                    pass
        # Save error info
        error_info = {
            "model": additional_info.get("model") if additional_info else "unknown",
            "prompt": additional_info.get("prompt") if additional_info else "unknown",
            "response": response,
            "temperature": additional_info.get("temperature") if additional_info else "unknown",
            "reasoning": additional_info.get("reasoning") if additional_info else False,
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