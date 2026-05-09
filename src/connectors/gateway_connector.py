from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from .base import LLMConnector


logger = logging.getLogger(__name__)


class LLMApiGatewayError(Exception):
    """Raised when LLM API Gateway returns an error response."""


class LLMApiGatewayConnector(LLMConnector):
    """Connector for a generic LLM API Gateway compatible with LLMConnector interface."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        base_url: str | None = None,
        timeout: int | None = None,
        extra_headers: Optional[Dict[str, str]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("LLM_GATEWAY_API_KEY")
        if not self.api_key:
            raise ValueError("LLM API Gateway key not provided")

        self.base_url = base_url or os.getenv("LLM_GATEWAY_BASE_URL", "https://api.llm-gateway.local")
        self.timeout = timeout or int(os.getenv("LLM_GATEWAY_TIMEOUT", 120))
        self.model = model
        self.temperature = temperature
        self.session = session or requests.Session()
        self.extra_headers = extra_headers or {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
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
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        elif self.temperature is not None:
            payload["temperature"] = self.temperature

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if extra_body:
            payload.update(extra_body)

        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )

        if not response.ok:
            raise LLMApiGatewayError(
                f"LLM API Gateway request failed: {response.status_code} {response.text}"
            )

        logger.info("LLM API Gateway response status=%s", response.status_code)
        return response.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def query(self, prompt: str) -> tuple[str, dict]:
        """Query the gateway model with a single prompt."""
        return self.query_conversation([{"role": "user", "content": prompt}])

    def query_conversation(self, messages: List[Dict[str, str]]) -> tuple[str, dict]:
        """Query the gateway model with conversation messages."""
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
            logger.error("Error querying LLM API Gateway: %s", e)
            raise
