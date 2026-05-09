from __future__ import annotations

import logging
import os
from typing import List, Dict

from openai import OpenAI

from .base import LLMConnector

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", 1000))


class OpenAIConnector(LLMConnector):
    """Direct OpenAI API connector (for native OpenAI models)."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def query(self, prompt: str) -> tuple[str, dict]:
        return self.query_conversation([{"role": "user", "content": prompt}])

    def query_conversation(self, messages: List[Dict[str, str]]) -> tuple[str, dict]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=self.temperature,
            )
            message = response.choices[0].message
            content = (message.content or "").strip()
            meta = {
                "chat_id": response.id,
                "model": response.model,
                "usage": response.usage.model_dump() if hasattr(response, "usage") else None,
            }
            return content, meta
        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            raise
