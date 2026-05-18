"""Model id aliases accepted by local configs and CLIs."""

from __future__ import annotations


MODEL_ALIASES = {
    "claude": "anthropic/claude-sonnet-4.5",
    "claude-sonnet": "anthropic/claude-sonnet-4.5",
    "anthropic/claude": "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet": "anthropic/claude-sonnet-4.5",
    "claude-opus": "anthropic/claude-opus-4.7",
    "anthropic/claude-opus": "anthropic/claude-opus-4.7",
}


def resolve_model_alias(model: str) -> str:
    """Return the canonical provider model id for supported shorthand aliases."""
    return MODEL_ALIASES.get(model.strip().lower(), model)
