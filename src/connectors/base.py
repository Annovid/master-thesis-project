from abc import ABC, abstractmethod
from typing import List, Dict

class LLMConnector(ABC):
    """Abstract base class for LLM connectors."""

    @abstractmethod
    def query(self, prompt: str) -> tuple[str, dict]:
        """Send a prompt to the LLM and return (response, metadata)."""

    @abstractmethod
    def query_conversation(self, messages: List[Dict[str, str]]) -> tuple[str, dict]:
        """Send a conversation to the LLM and return (response, metadata)."""
