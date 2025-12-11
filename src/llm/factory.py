"""LLM Provider Factory.

This module provides factory functions for creating LLM provider instances.
"""

from __future__ import annotations

import os
from typing import Any

from src.llm.base import BaseLLMProvider
from src.llm.anthropic import AnthropicProvider
from src.llm.upstage import UpstageProvider


class LLMProviderFactory:
    """Factory for creating LLM provider instances.

    Supports creating providers by name with automatic configuration
    from environment variables.
    """

    # Registry of available providers
    _providers: dict[str, type[BaseLLMProvider]] = {
        "anthropic": AnthropicProvider,
        "upstage": UpstageProvider,
    }

    # Model to provider mapping for automatic provider detection
    _model_provider_map: dict[str, str] = {
        # Anthropic models
        "claude-opus-4-20250514": "anthropic",
        "claude-sonnet-4-20250514": "anthropic",
        "claude-haiku-4-5-20251001": "anthropic",
        "claude-3-5-sonnet-20241022": "anthropic",
        "claude-3-5-haiku-20241022": "anthropic",
        "claude-3-opus-20240229": "anthropic",
        # Upstage models
        "solar-pro2": "upstage",
        "solar-pro": "upstage",
        "solar-mini": "upstage",
    }

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: type[BaseLLMProvider],
    ) -> None:
        """Register a new LLM provider.

        Args:
            name: Provider name (e.g., 'openai', 'google').
            provider_class: Provider class implementing BaseLLMProvider.
        """
        cls._providers[name] = provider_class

    @classmethod
    def register_model(cls, model_name: str, provider_name: str) -> None:
        """Register a model to provider mapping.

        Args:
            model_name: Model name (e.g., 'gpt-4').
            provider_name: Provider name (e.g., 'openai').
        """
        cls._model_provider_map[model_name] = provider_name

    @classmethod
    def get_provider_for_model(cls, model: str) -> str | None:
        """Get the provider name for a given model.

        Args:
            model: Model name.

        Returns:
            Provider name or None if not found.
        """
        # Exact match
        if model in cls._model_provider_map:
            return cls._model_provider_map[model]

        # Prefix matching for model variants
        for model_prefix, provider in cls._model_provider_map.items():
            if model.startswith(model_prefix.split("-")[0]):
                return provider

        return None

    @classmethod
    def create(
        cls,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """Create an LLM provider instance.

        Args:
            provider: Provider name. If None, inferred from model.
            model: Model name (used to infer provider if not specified).
            api_key: API key. If None, reads from environment.
            **kwargs: Additional provider-specific configuration.

        Returns:
            BaseLLMProvider instance.

        Raises:
            ValueError: If provider cannot be determined or is unknown.
        """
        # Determine provider
        if provider is None and model:
            provider = cls.get_provider_for_model(model)

        if provider is None:
            # Default to anthropic
            provider = "anthropic"

        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider: {provider}. Available: {available}"
            )

        # Get API key from environment if not provided
        if api_key is None:
            env_key_map = {
                "anthropic": "ANTHROPIC_API_KEY",
                "upstage": "UPSTAGE_API_KEY",
            }
            env_var = env_key_map.get(provider, f"{provider.upper()}_API_KEY")
            api_key = os.getenv(env_var)

        provider_class = cls._providers[provider]
        return provider_class(api_key=api_key, **kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider names.
        """
        return list(cls._providers.keys())

    @classmethod
    def list_models(cls, provider: str | None = None) -> list[str]:
        """List all registered models, optionally filtered by provider.

        Args:
            provider: Optional provider name to filter by.

        Returns:
            List of model names.
        """
        if provider:
            return [
                model
                for model, prov in cls._model_provider_map.items()
                if prov == provider
            ]
        return list(cls._model_provider_map.keys())


def get_provider(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Convenience function to create an LLM provider.

    This is a shortcut for LLMProviderFactory.create().

    Args:
        provider: Provider name. If None, inferred from model.
        model: Model name (used to infer provider if not specified).
        api_key: API key. If None, reads from environment.
        **kwargs: Additional provider-specific configuration.

    Returns:
        BaseLLMProvider instance.
    """
    return LLMProviderFactory.create(
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs,
    )
