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
        self.last_action: Action | None = None
        self.last_response: str | None = None

    def act(self, state: GameState, game: Game, player_id: int) -> tuple[Action, dict[str, Any]]:
        """Decide action using LLM with conversation history."""
        prompt = game.get_prompt(state, player_id, state.max_rounds)

        # Add user message to conversation
        self.conversation.append({"role": "user", "content": prompt})

        from datetime import datetime
        from pathlib import Path

        try:
            # Query with full conversation
            response, meta = self.connector.query_conversation(self.conversation)

            # Add assistant response to conversation
            self.conversation.append({"role": "assistant", "content": response})

            # Persist prompt/response metadata
            ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
            log_dir = Path("artifacts") / "chats"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"chat_{ts}.txt"
            with log_path.open("w") as f:
                f.write(f"chat_id: {meta.get('chat_id')}")
                f.write("\n")
                f.write(f"model: {meta.get('model')}")
                f.write("\n\n")
                f.write("PROMPT:\n")
                f.write(prompt)
                f.write("\n\n")
                f.write("RESPONSE:\n")
                f.write(response)
                f.write("\n")

            additional_info = {
                "model": self.connector.model if hasattr(self.connector, 'model') else "unknown",
                "temperature": self.temperature,
                "prompt": prompt,
                "chat_id": meta.get("chat_id"),
                "model": meta.get("model"),
                "usage": meta.get("usage"),
            }

            action = game.parse_action(response)
            if not game.validate_action(action):
                logger.warning(f"Invalid action {action}, using default")
                action = 0 if not hasattr(game, 'endowment') else 0.0

            # Save last successful
            self.last_action = action
            self.last_response = response

            details = {
                "type": "llm",
                "prompt": prompt,
                "response": response,
                "parsed_action": action,
                "conversation_length": len(self.conversation),
                "chat_id": meta.get("chat_id"),
                "model": meta.get("model"),
                "usage": meta.get("usage"),
                "log_path": str(log_path),
                "reused": False,
            }
            # Emit single-line status with parsed action
            print(f"OK {log_path} player_id={player_id + 1} model={meta.get('model')} parsed_action={action}")

            return action, details

        except Exception as exc:
            logger.error("LLM agent error for player %s: %s", player_id + 1, exc, exc_info=True)

            # Reuse last successful action/response if available
            if self.last_action is not None:
                reused_action = self.last_action
                reused_response = self.last_response or ""
            else:
                reused_action = 0 if not hasattr(game, 'endowment') else 0.0
                reused_response = ""

            # Record error artifact
            ts_err = datetime.now().strftime("%Y%m%d%H%M%S%f")
            err_dir = Path("artifacts") / "errors"
            err_dir.mkdir(parents=True, exist_ok=True)
            err_path = err_dir / f"error_{ts_err}.log"
            with err_path.open("w") as f:
                f.write(f"player_id: {player_id + 1}\n")
                f.write(f"model: {getattr(self.connector, 'model', 'unknown')}\n")
                f.write(f"error: {exc}\n")
                f.write("prompt:\n")
                f.write(prompt)
                f.write("\n")
                f.write("reused_response:\n")
                f.write(reused_response)
                f.write("\n")

            details = {
                "type": "llm",
                "prompt": prompt,
                "response": reused_response,
                "parsed_action": reused_action,
                "conversation_length": len(self.conversation),
                "chat_id": None,
                "model": getattr(self.connector, 'model', 'unknown'),
                "usage": None,
                "log_path": str(err_path),
                "reused": True,
                "error": str(exc),
            }

            # Emit single-line status for reused action
            print(f"ERROR_REUSED {err_path} player_id={player_id + 1} model={getattr(self.connector, 'model', 'unknown')} parsed_action={reused_action}")

            return reused_action, details
