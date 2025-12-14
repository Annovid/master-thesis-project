from typing import List, Dict

from .base import LLMConnector

class MockConnector(LLMConnector):
    """Mock connector for testing, always returns 'Cooperate'."""

    def query(self, prompt: str) -> str:
        """Return a mock response."""
        return "Cooperate"

    def query_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Return a mock response for conversation."""
        return "Cooperate"