import logging
from typing import Any, List

from .base import Agent
from ..game.state import GameState
from ..games.base_game import Game, Action
from ...connectors.base import LLMConnector

logger = logging.getLogger(__name__)

class LLMAgent(Agent):
    """Agent that uses an LLM to decide actions with conversation context."""

    def __init__(self, name: str, connector: LLMConnector, temperature: float = 0.0, reasoning: bool = False) -> None:
        super().__init__(name)
        self.connector = connector
        self.temperature = temperature
        self.reasoning = reasoning
        self.conversation: List[dict[str, str]] = []

    def act(self, state: GameState, game: Game, player_id: int) -> tuple[Action, dict[str, Any]]:
        """Decide action using LLM with conversation history."""
        prompt = game.get_prompt(state, player_id, state.max_rounds)
        if self.reasoning:
            prompt += "\n\nProvide your reasoning step by step, and end with 'Answer = num' on the last line.\n"

        # Add user message to conversation
        self.conversation.append({"role": "user", "content": prompt})
        
        # Query with full conversation
        response = self.connector.query_conversation(self.conversation)
        
        # Add assistant response to conversation
        self.conversation.append({"role": "assistant", "content": response})
        
        additional_info = {
            "model": self.connector.model if hasattr(self.connector, 'model') else "unknown",
            "temperature": self.temperature,
            "prompt": prompt
        }
        action = game.parse_action(response)
        if not game.validate_action(action):
            logger.warning(f"Invalid action {action}, using default")
            action = 0 if not hasattr(game, 'endowment') else 0.0
        details = {
            "type": "llm",
            "prompt": prompt,
            "response": response,
            "parsed_action": action,
            "conversation_length": len(self.conversation)
        }
        return action, details