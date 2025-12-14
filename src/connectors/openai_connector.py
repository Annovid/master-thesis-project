import logging
import os
from typing import List, Dict

from openai import OpenAI

from .base import LLMConnector

logger = logging.getLogger(__name__)

MAX_TOKENS = 100
TEMPERATURE = 0.0

class OpenAIConnector(LLMConnector):
    """Connector for OpenAI API."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-3.5-turbo", temperature: float = 0.0) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def query(self, prompt: str) -> str:
        """Query the OpenAI model with the prompt."""
        return self.query_conversation([{"role": "user", "content": prompt}])

    def query_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Query the OpenAI model with conversation messages."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            raise