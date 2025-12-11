"""Tests for LLM provider abstraction layer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm.base import BaseLLMProvider, LLMResponse
from src.llm.anthropic import AnthropicProvider
from src.llm.upstage import UpstageProvider, UpstageOCR
from src.llm.factory import LLMProviderFactory, get_provider


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_create_response(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            content="Hello, world!",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason="stop",
        )
        assert response.content == "Hello, world!"
        assert response.model == "test-model"
        assert response.usage["input_tokens"] == 10
        assert response.finish_reason == "stop"

    def test_response_with_defaults(self):
        """Test response with default values."""
        response = LLMResponse(content="test", model="model")
        assert response.usage == {}
        assert response.finish_reason is None
        assert response.raw_response is None


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_provider_name(self):
        """Test provider name property."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider()
            assert provider.provider_name == "anthropic"

    def test_default_model(self):
        """Test default model property."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider()
            assert provider.default_model == "claude-haiku-4-5-20251001"

    def test_available_models(self):
        """Test available models list."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider()
            models = provider.get_available_models()
            assert "claude-haiku-4-5-20251001" in models
            assert "claude-sonnet-4-20250514" in models

    @pytest.mark.asyncio
    async def test_chat_call(self):
        """Test chat completion call."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider()

            # Mock the async client
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Hello!")]
            mock_response.model = "claude-haiku-4-5-20251001"
            mock_response.usage = MagicMock(input_tokens=5, output_tokens=3)
            mock_response.stop_reason = "end_turn"

            provider._async_client.messages.create = AsyncMock(
                return_value=mock_response
            )

            response = await provider.chat(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
            )

            assert response.content == "Hello!"
            assert response.model == "claude-haiku-4-5-20251001"


class TestUpstageProvider:
    """Tests for UpstageProvider."""

    def test_provider_name(self):
        """Test provider name property."""
        with patch.dict("os.environ", {"UPSTAGE_API_KEY": "test-key"}):
            provider = UpstageProvider()
            assert provider.provider_name == "upstage"

    def test_default_model(self):
        """Test default model property."""
        with patch.dict("os.environ", {"UPSTAGE_API_KEY": "test-key"}):
            provider = UpstageProvider()
            assert provider.default_model == "solar-pro2"

    def test_available_models(self):
        """Test available models list."""
        with patch.dict("os.environ", {"UPSTAGE_API_KEY": "test-key"}):
            provider = UpstageProvider()
            models = provider.get_available_models()
            assert "solar-pro2" in models
            assert "solar-pro" in models
            assert "solar-mini" in models

    def test_base_url(self):
        """Test base URL configuration."""
        with patch.dict("os.environ", {"UPSTAGE_API_KEY": "test-key"}):
            provider = UpstageProvider()
            assert provider._base_url == "https://api.upstage.ai/v1"

    @pytest.mark.asyncio
    async def test_chat_call(self):
        """Test chat completion call."""
        with patch.dict("os.environ", {"UPSTAGE_API_KEY": "test-key"}):
            provider = UpstageProvider()

            # Mock the async client
            mock_choice = MagicMock()
            mock_choice.message.content = "안녕하세요!"
            mock_choice.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.model = "solar-pro2"
            mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=3)

            provider._async_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            response = await provider.chat(
                messages=[{"role": "user", "content": "안녕"}],
                max_tokens=100,
            )

            assert response.content == "안녕하세요!"
            assert response.model == "solar-pro2"


class TestUpstageOCR:
    """Tests for UpstageOCR service."""

    def test_initialization(self):
        """Test OCR service initialization."""
        with patch.dict("os.environ", {"UPSTAGE_API_KEY": "test-key"}):
            ocr = UpstageOCR()
            assert ocr._api_key == "test-key"
            assert ocr._base_url == "https://api.upstage.ai/v1/document-digitization"

    def test_custom_api_key(self):
        """Test OCR with custom API key."""
        ocr = UpstageOCR(api_key="custom-key")
        assert ocr._api_key == "custom-key"

    @pytest.mark.asyncio
    async def test_digitize_requires_file(self):
        """Test that digitize requires file input."""
        ocr = UpstageOCR(api_key="test-key")
        with pytest.raises(ValueError, match="Either file_path or file_content"):
            await ocr.digitize()


class TestLLMProviderFactory:
    """Tests for LLMProviderFactory."""

    def test_list_providers(self):
        """Test listing available providers."""
        providers = LLMProviderFactory.list_providers()
        assert "anthropic" in providers
        assert "upstage" in providers

    def test_list_models(self):
        """Test listing available models."""
        models = LLMProviderFactory.list_models()
        assert "claude-haiku-4-5-20251001" in models
        assert "solar-pro2" in models

    def test_list_models_by_provider(self):
        """Test listing models filtered by provider."""
        anthropic_models = LLMProviderFactory.list_models("anthropic")
        assert "claude-haiku-4-5-20251001" in anthropic_models
        assert "solar-pro2" not in anthropic_models

        upstage_models = LLMProviderFactory.list_models("upstage")
        assert "solar-pro2" in upstage_models
        assert "claude-haiku-4-5-20251001" not in upstage_models

    def test_get_provider_for_model(self):
        """Test getting provider for a model."""
        assert LLMProviderFactory.get_provider_for_model("claude-haiku-4-5-20251001") == "anthropic"
        assert LLMProviderFactory.get_provider_for_model("solar-pro2") == "upstage"

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = LLMProviderFactory.create(provider="anthropic")
            assert isinstance(provider, AnthropicProvider)

    def test_create_upstage_provider(self):
        """Test creating Upstage provider."""
        with patch.dict("os.environ", {"UPSTAGE_API_KEY": "test-key"}):
            provider = LLMProviderFactory.create(provider="upstage")
            assert isinstance(provider, UpstageProvider)

    def test_create_provider_from_model(self):
        """Test creating provider from model name."""
        with patch.dict("os.environ", {"UPSTAGE_API_KEY": "test-key"}):
            provider = LLMProviderFactory.create(model="solar-pro2")
            assert isinstance(provider, UpstageProvider)

    def test_create_unknown_provider_raises(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMProviderFactory.create(provider="unknown")

    def test_register_provider(self):
        """Test registering a custom provider."""

        class CustomProvider(BaseLLMProvider):
            @property
            def provider_name(self) -> str:
                return "custom"

            @property
            def default_model(self) -> str:
                return "custom-model"

            async def chat(self, messages, **kwargs):
                return LLMResponse(content="", model="custom-model")

            async def chat_stream(self, messages, **kwargs):
                yield ""

        LLMProviderFactory.register_provider("custom", CustomProvider)
        assert "custom" in LLMProviderFactory.list_providers()

        # Cleanup
        del LLMProviderFactory._providers["custom"]


class TestGetProvider:
    """Tests for get_provider convenience function."""

    def test_get_provider_anthropic(self):
        """Test getting Anthropic provider."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = get_provider(provider="anthropic")
            assert isinstance(provider, AnthropicProvider)

    def test_get_provider_upstage(self):
        """Test getting Upstage provider."""
        with patch.dict("os.environ", {"UPSTAGE_API_KEY": "test-key"}):
            provider = get_provider(provider="upstage")
            assert isinstance(provider, UpstageProvider)

    def test_get_provider_by_model(self):
        """Test getting provider by model name."""
        with patch.dict("os.environ", {"UPSTAGE_API_KEY": "test-key"}):
            provider = get_provider(model="solar-pro2")
            assert isinstance(provider, UpstageProvider)
