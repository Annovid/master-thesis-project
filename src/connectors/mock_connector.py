from typing import List, Dict

from .base import LLMConnector

class MockConnector(LLMConnector):
    """Mock connector for testing, always returns 'Cooperate'."""

    def query(self, prompt: str) -> tuple[str, dict]:
        """Return a mock response and metadata."""
        return "Cooperate", {"chat_id": "mock", "model": "mock"}

    def query_conversation(self, messages: List[Dict[str, str]]) -> tuple[str, dict]:
        """Return a mock response for conversation and metadata."""
        return "Cooperate", {"chat_id": "mock", "model": "mock"}
