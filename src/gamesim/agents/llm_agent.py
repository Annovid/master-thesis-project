import logging
from typing import Any, List

from .base import Agent
from ..game.state import GameState
from ..games.base_game import Game, Action
from ...connectors.base import LLMConnector

logger = logging.getLogger(__name__)

class LLMAgent(Agent):
    """Agent that uses an LLM to decide actions with conversation context."""

    def __init__(self, name: str, connector: LLMConnector, temperature: float = 0.0) -> None:
        super().__init__(name)
        self.connector = connector
        self.temperature = temperature
        self.conversation: List[dict[str, str]] = []

    def act(self, state: GameState, game: Game, player_id: int) -> tuple[Action, dict[str, Any]]:
        """Decide action using LLM with conversation history."""
        prompt = game.get_prompt(state, player_id)
        
        # Add user message to conversation
        self.conversation.append({"role": "user", "content": prompt})
        
        # Query with full conversation
        response = self.connector.query_conversation(self.conversation)
        
        # Add assistant response to conversation
        self.conversation.append({"role": "assistant", "content": response})
        
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