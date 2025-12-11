"""Base LLM Provider - Abstract interface for LLM providers.

This module defines the abstract base class that all LLM providers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    raw_response: Any = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers (Anthropic, Upstage, OpenAI, etc.) must implement
    this interface to be used interchangeably in the agent system.
    """

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """Initialize the LLM provider.

        Args:
            api_key: API key for authentication.
            **kwargs: Additional provider-specific configuration.
        """
        self._api_key = api_key
        self._config = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'anthropic', 'upstage')."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider."""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model to use. If None, uses default_model.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0 to 2.0).
            system_prompt: Optional system prompt.
            **kwargs: Additional provider-specific parameters.

        Returns:
            LLMResponse containing the model's response.
        """
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Send a streaming chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model to use. If None, uses default_model.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0 to 2.0).
            system_prompt: Optional system prompt.
            **kwargs: Additional provider-specific parameters.

        Yields:
            String chunks of the response as they arrive.
        """
        pass

    def get_available_models(self) -> list[str]:
        """Return list of available models for this provider.

        Subclasses should override this to return their specific models.
        """
        return [self.default_model]

    async def health_check(self) -> dict[str, Any]:
        """Check if the provider is healthy and accessible.

        Returns:
            Dictionary with health status information.
        """
        return {
            "provider": self.provider_name,
            "status": "unknown",
        }
