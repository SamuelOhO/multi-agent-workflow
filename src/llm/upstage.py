"""Upstage LLM Provider implementation.

This module provides Upstage Solar API integration for chat completions
and document digitization (OCR).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO

import httpx
from openai import AsyncOpenAI, OpenAI

from src.llm.base import BaseLLMProvider, LLMResponse


class UpstageProvider(BaseLLMProvider):
    """Upstage Solar API provider.

    Provides access to Solar models through the Upstage API using
    OpenAI-compatible interface.
    """

    BASE_URL = "https://api.upstage.ai/v1"

    AVAILABLE_MODELS = [
        "solar-pro2",
        "solar-pro",
        "solar-mini",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Upstage provider.

        Args:
            api_key: Upstage API key. If None, reads from UPSTAGE_API_KEY env var.
            base_url: Optional custom base URL. Defaults to Upstage API.
            **kwargs: Additional configuration.
        """
        resolved_api_key = api_key or os.getenv("UPSTAGE_API_KEY", "")
        super().__init__(api_key=resolved_api_key, **kwargs)

        self._base_url = base_url or self.BASE_URL

        # Use OpenAI client with Upstage base URL
        self._client = OpenAI(
            api_key=resolved_api_key,
            base_url=self._base_url,
        )
        self._async_client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=self._base_url,
        )

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "upstage"

    @property
    def default_model(self) -> str:
        """Return the default model."""
        return "solar-pro2"

    def get_available_models(self) -> list[str]:
        """Return list of available Upstage models."""
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
        """Send a chat completion request to Upstage.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model to use. If None, uses default_model.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters.

        Returns:
            LLMResponse containing Solar's response.
        """
        used_model = model or self.default_model

        # Build messages with system prompt
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = await self._async_client.chat.completions.create(
            model=used_model,
            messages=full_messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            **kwargs,
        )

        content = ""
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content or ""

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            finish_reason=response.choices[0].finish_reason if response.choices else None,
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
        """Send a streaming chat completion request to Upstage.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model to use. If None, uses default_model.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters.

        Yields:
            String chunks of the response as they arrive.
        """
        used_model = model or self.default_model

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        stream = await self._async_client.chat.completions.create(
            model=used_model,
            messages=full_messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    yield delta.content

    async def health_check(self) -> dict[str, Any]:
        """Check if the Upstage API is accessible.

        Returns:
            Dictionary with health status information.
        """
        try:
            response = await self._async_client.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10,
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


class UpstageOCR:
    """Upstage Document Digitization (OCR) service.

    Provides OCR capabilities for document processing using the
    Upstage Document Digitization API.
    """

    BASE_URL = "https://api.upstage.ai/v1/document-digitization"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize the OCR service.

        Args:
            api_key: Upstage API key. If None, reads from UPSTAGE_API_KEY env var.
            base_url: Optional custom base URL.
        """
        self._api_key = api_key or os.getenv("UPSTAGE_API_KEY", "")
        self._base_url = base_url or self.BASE_URL

    async def digitize(
        self,
        file_path: str | Path | None = None,
        file_content: BinaryIO | bytes | None = None,
        filename: str = "document",
        model: str = "ocr",
        output_format: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Digitize a document using OCR.

        Args:
            file_path: Path to the document file.
            file_content: File content as bytes or file-like object.
            filename: Name of the file (used when passing file_content).
            model: OCR model to use. Options: "ocr", "document-parse".
            output_format: Output format. Options: "text", "html", "markdown".
            **kwargs: Additional API parameters.

        Returns:
            Dictionary containing OCR results.

        Raises:
            ValueError: If neither file_path nor file_content is provided.
        """
        if file_path is None and file_content is None:
            raise ValueError("Either file_path or file_content must be provided")

        headers = {"Authorization": f"Bearer {self._api_key}"}
        data: dict[str, Any] = {"model": model}

        if output_format:
            data["output_format"] = output_format

        data.update(kwargs)

        async with httpx.AsyncClient() as client:
            if file_path:
                path = Path(file_path)
                with open(path, "rb") as f:
                    files = {"document": (path.name, f)}
                    response = await client.post(
                        self._base_url,
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=120.0,
                    )
            else:
                if isinstance(file_content, bytes):
                    files = {"document": (filename, file_content)}
                else:
                    files = {"document": (filename, file_content)}
                response = await client.post(
                    self._base_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=120.0,
                )

            response.raise_for_status()
            return response.json()

    def digitize_sync(
        self,
        file_path: str | Path | None = None,
        file_content: BinaryIO | bytes | None = None,
        filename: str = "document",
        model: str = "ocr",
        output_format: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Synchronous version of digitize.

        Args:
            file_path: Path to the document file.
            file_content: File content as bytes or file-like object.
            filename: Name of the file (used when passing file_content).
            model: OCR model to use.
            output_format: Output format.
            **kwargs: Additional API parameters.

        Returns:
            Dictionary containing OCR results.
        """
        import requests

        if file_path is None and file_content is None:
            raise ValueError("Either file_path or file_content must be provided")

        headers = {"Authorization": f"Bearer {self._api_key}"}
        data: dict[str, Any] = {"model": model}

        if output_format:
            data["output_format"] = output_format

        data.update(kwargs)

        if file_path:
            path = Path(file_path)
            with open(path, "rb") as f:
                files = {"document": (path.name, f)}
                response = requests.post(
                    self._base_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=120,
                )
        else:
            if isinstance(file_content, bytes):
                files = {"document": (filename, file_content)}
            else:
                files = {"document": (filename, file_content)}
            response = requests.post(
                self._base_url,
                headers=headers,
                files=files,
                data=data,
                timeout=120,
            )

        response.raise_for_status()
        return response.json()

    async def parse_document(
        self,
        file_path: str | Path | None = None,
        file_content: BinaryIO | bytes | None = None,
        filename: str = "document",
        output_format: str = "markdown",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Parse a complex document (uses document-parse model).

        This is useful for complex documents with tables, figures, etc.

        Args:
            file_path: Path to the document file.
            file_content: File content as bytes or file-like object.
            filename: Name of the file.
            output_format: Output format (markdown, html, text).
            **kwargs: Additional API parameters.

        Returns:
            Dictionary containing parsed document.
        """
        return await self.digitize(
            file_path=file_path,
            file_content=file_content,
            filename=filename,
            model="document-parse",
            output_format=output_format,
            **kwargs,
        )
