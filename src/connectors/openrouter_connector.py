import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .base import LLMConnector

logger = logging.getLogger(__name__)


class OpenRouterError(Exception):
    """Raised when OpenRouter returns an error response."""


DEFAULT_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", 1000))


@dataclass
class OpenRouterConnector(LLMConnector):
    """Connector for OpenRouter API compatible with LLMConnector interface."""

    api_key: Optional[str] = None
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.0
    base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    site_url: Optional[str] = os.getenv("OPENROUTER_SITE_URL")
    site_name: Optional[str] = os.getenv("OPENROUTER_SITE_NAME")
    timeout: int = int(os.getenv("OPENROUTER_TIMEOUT", 120))
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Optional OpenRouter attribution headers
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-OpenRouter-Title"] = self.site_name

        return headers

    def _chat_request(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        logger.info("OpenRouter request model=%s", model or self.model)
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
        }

        payload["temperature"] = temperature if temperature is not None else self.temperature
        payload["max_tokens"] = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        if extra_body:
            payload.update(extra_body)

        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )

        if not response.ok:
            raise OpenRouterError(
                f"OpenRouter request failed: {response.status_code} {response.text}"
            )

        logger.info("OpenRouter response status=%s", response.status_code)
        return response.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def query(self, prompt: str) -> tuple[str, dict]:
        """Query the OpenRouter model with a single prompt."""
        return self.query_conversation([{"role": "user", "content": prompt}])

    def query_conversation(self, messages: List[Dict[str, str]]) -> tuple[str, dict]:
        """Query the OpenRouter model with conversation messages."""
        try:
            data = self._chat_request(messages)

            message = data["choices"][0]["message"]
            content = (message.get("content") or "").strip()
            usage = data.get("usage")

            meta = {
                "chat_id": data.get("id"),
                "model": data.get("model"),
                "usage": usage,
            }

            return content, meta
        except Exception as e:
            logger.error(f"Error querying OpenRouter: {e}")
            raise
