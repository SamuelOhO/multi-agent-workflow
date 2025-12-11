"""Anthropic LLM Provider implementation.

This module provides the Anthropic Claude API integration.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

import anthropic

from src.llm.base import BaseLLMProvider, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider.

    Provides access to Claude models through the Anthropic API.
    """

    AVAILABLE_MODELS = [
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            base_url: Optional custom base URL for the API.
            **kwargs: Additional configuration.
        """
        resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        super().__init__(api_key=resolved_api_key, **kwargs)

        self._client = anthropic.Anthropic(
            api_key=resolved_api_key,
            base_url=base_url,
        )
        self._async_client = anthropic.AsyncAnthropic(
            api_key=resolved_api_key,
            base_url=base_url,
        )

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "anthropic"

    @property
    def default_model(self) -> str:
        """Return the default model."""
        return "claude-haiku-4-5-20251001"

    def get_available_models(self) -> list[str]:
        """Return list of available Anthropic models."""
        return self.AVAILABLE_MODELS.copy()

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to Anthropic.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model to use. If None, uses default_model.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            system_prompt: Optional system prompt.
            **kwargs: Additional Anthropic-specific parameters.

        Returns:
            LLMResponse containing Claude's response.
        """
        used_model = model or self.default_model

        # Build request parameters
        request_params: dict[str, Any] = {
            "model": used_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system_prompt:
            request_params["system"] = system_prompt

        # Add any extra kwargs
        request_params.update(kwargs)

        response = await self._async_client.messages.create(**request_params)

        # Extract content
        content = ""
        if response.content and len(response.content) > 0:
            first_block = response.content[0]
            if hasattr(first_block, "text"):
                content = first_block.text

        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
            raw_response=response,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Send a streaming chat completion request to Anthropic.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model to use. If None, uses default_model.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            system_prompt: Optional system prompt.
            **kwargs: Additional Anthropic-specific parameters.

        Yields:
            String chunks of the response as they arrive.
        """
        used_model = model or self.default_model

        request_params: dict[str, Any] = {
            "model": used_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system_prompt:
            request_params["system"] = system_prompt

        request_params.update(kwargs)

        async with self._async_client.messages.stream(**request_params) as stream:
            async for text in stream.text_stream:
                yield text

    async def health_check(self) -> dict[str, Any]:
        """Check if the Anthropic API is accessible.

        Returns:
            Dictionary with health status information.
        """
        try:
            # Simple test call
            response = await self._async_client.messages.create(
                model=self.default_model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return {
                "provider": self.provider_name,
                "status": "healthy",
                "model": response.model,
            }
        except Exception as e:
            return {
                "provider": self.provider_name,
                "status": "unhealthy",
                "error": str(e),
            }
