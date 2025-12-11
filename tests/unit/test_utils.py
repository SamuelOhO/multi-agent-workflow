"""Unit tests for utility modules.

Tests for config, logging, exceptions, error_handlers, and observability.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from src.utils.config import (
    AgentDefaults,
    AppConfig,
    AppSettings,
    Environment,
    LogFormat,
    LoggingConfig,
    TimeoutConfig,
    get_config,
    init_config,
    reset_config,
)
from src.utils.exceptions import (
    AgentOrchestratorError,
    AgentTimeoutError,
    BadRequestError,
    ConflictError,
    ConversationTimeoutError,
    ForbiddenError,
    InternalServerError,
    InvalidConfigurationError,
    LLMAPIError,
    LLMRateLimitError,
    LLMTimeoutError,
    MissingConfigurationError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    UnauthorizedError,
    ValidationError,
)
from src.utils.logging import (
    LoggerAdapter,
    clear_correlation_id,
    get_agent_logger,
    get_api_logger,
    get_conversation_logger,
    get_correlation_id,
    get_logger,
    set_correlation_id,
    setup_logging,
)
from src.utils.observability import (
    LangfuseClient,
    get_observability_client,
    init_observability,
    reset_observability,
)

# ============================================================================
# Config Tests
# ============================================================================


class TestAppSettings:
    """Tests for AppSettings model."""

    def test_default_values(self):
        """Test default values are set correctly."""
        settings = AppSettings()
        assert settings.env == Environment.DEVELOPMENT
        assert settings.debug is False
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000

    def test_custom_values(self):
        """Test custom values are accepted."""
        settings = AppSettings(
            env=Environment.PRODUCTION,
            debug=True,
            host="127.0.0.1",
            port=3000,
        )
        assert settings.env == Environment.PRODUCTION
        assert settings.debug is True
        assert settings.host == "127.0.0.1"
        assert settings.port == 3000

    def test_invalid_port(self):
        """Test invalid port raises error."""
        with pytest.raises(ValueError):
            AppSettings(port=0)
        with pytest.raises(ValueError):
            AppSettings(port=70000)


class TestLoggingConfig:
    """Tests for LoggingConfig model."""

    def test_default_values(self):
        """Test default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == LogFormat.JSON

    def test_valid_levels(self):
        """Test valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LoggingConfig(level=level)
            assert config.level == level

    def test_invalid_level(self):
        """Test invalid log level raises error."""
        with pytest.raises(ValueError):
            LoggingConfig(level="INVALID")


class TestAgentDefaults:
    """Tests for AgentDefaults model."""

    def test_default_values(self):
        """Test default values."""
        defaults = AgentDefaults()
        assert defaults.model == "claude-haiku-4-5-20251001"
        assert defaults.max_tokens == 4096
        assert defaults.temperature == 0.7

    def test_temperature_validation(self):
        """Test temperature must be between 0.0 and 2.0."""
        AgentDefaults(temperature=0.0)
        AgentDefaults(temperature=2.0)
        with pytest.raises(ValueError):
            AgentDefaults(temperature=-0.1)
        with pytest.raises(ValueError):
            AgentDefaults(temperature=2.1)

    def test_max_tokens_validation(self):
        """Test max_tokens must be positive."""
        with pytest.raises(ValueError):
            AgentDefaults(max_tokens=0)
        with pytest.raises(ValueError):
            AgentDefaults(max_tokens=-1)


class TestTimeoutConfig:
    """Tests for TimeoutConfig model."""

    def test_default_values(self):
        """Test default values."""
        config = TimeoutConfig()
        assert config.agent == 120
        assert config.conversation == 300

    def test_invalid_timeout(self):
        """Test invalid timeout raises error."""
        with pytest.raises(ValueError):
            TimeoutConfig(agent=0)
        with pytest.raises(ValueError):
            TimeoutConfig(conversation=-1)


class TestAppConfig:
    """Tests for AppConfig model."""

    def test_default_config(self):
        """Test default configuration."""
        config = AppConfig()
        assert config.app.env == Environment.DEVELOPMENT
        assert config.logging.level == "INFO"
        assert config.agent_defaults.model == "claude-haiku-4-5-20251001"

    def test_from_yaml(self):
        """Test loading from YAML file."""
        yaml_content = """
app:
  env: production
  debug: false
  port: 9000
logging:
  level: WARNING
  format: console
agent_defaults:
  model: claude-3-opus
  temperature: 0.5
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = os.path.join(tmpdir, "test_config.yaml")
            with open(yaml_path, "w") as f:
                f.write(yaml_content)
            config = AppConfig.from_yaml(yaml_path)
            assert config.app.env == Environment.PRODUCTION
            assert config.app.port == 9000
            assert config.logging.level == "WARNING"
            assert config.agent_defaults.temperature == 0.5

    def test_from_yaml_file_not_found(self):
        """Test FileNotFoundError for missing YAML."""
        with pytest.raises(FileNotFoundError):
            AppConfig.from_yaml("/nonexistent/path.yaml")

    def test_from_env(self):
        """Test loading from environment variables."""
        with patch.dict(
            os.environ,
            {
                "APP_ENV": "staging",
                "APP_DEBUG": "true",
                "APP_PORT": "5000",
                "LOG_LEVEL": "DEBUG",
                "DEFAULT_TEMPERATURE": "0.9",
            },
        ):
            config = AppConfig.from_env()
            assert config.app.env == Environment.STAGING
            assert config.app.debug is True
            assert config.app.port == 5000
            assert config.logging.level == "DEBUG"
            assert config.agent_defaults.temperature == 0.9


class TestGlobalConfig:
    """Tests for global config functions."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_get_config_not_initialized(self):
        """Test get_config raises error when not initialized."""
        with pytest.raises(RuntimeError):
            get_config()

    def test_init_config(self):
        """Test init_config creates global config."""
        config = init_config()
        assert config is not None
        assert get_config() is config

    def test_reset_config(self):
        """Test reset_config clears global config."""
        init_config()
        reset_config()
        with pytest.raises(RuntimeError):
            get_config()


# ============================================================================
# Logging Tests
# ============================================================================


class TestLogging:
    """Tests for logging module."""

    def test_setup_logging_json(self):
        """Test JSON logging setup."""
        setup_logging(level="DEBUG", json_format=True)
        logger = get_logger("test")
        assert logger is not None

    def test_setup_logging_console(self):
        """Test console logging setup."""
        setup_logging(level="INFO", json_format=False)
        logger = get_logger("test")
        assert logger is not None

    def test_correlation_id(self):
        """Test correlation ID context."""
        assert get_correlation_id() is None
        cid = set_correlation_id()
        assert cid is not None
        assert get_correlation_id() == cid
        clear_correlation_id()
        assert get_correlation_id() is None

    def test_set_specific_correlation_id(self):
        """Test setting a specific correlation ID."""
        cid = set_correlation_id("my-correlation-id")
        assert cid == "my-correlation-id"
        assert get_correlation_id() == "my-correlation-id"
        clear_correlation_id()


class TestLoggerAdapter:
    """Tests for LoggerAdapter class."""

    def setup_method(self):
        """Setup logging before tests."""
        setup_logging(level="DEBUG", json_format=True)

    def test_create_adapter(self):
        """Test creating a logger adapter."""
        adapter = LoggerAdapter("test")
        assert adapter is not None

    def test_adapter_with_context(self):
        """Test adapter with initial context."""
        adapter = LoggerAdapter("test", user_id="123", request_id="abc")
        assert adapter._context == {"user_id": "123", "request_id": "abc"}

    def test_bind_context(self):
        """Test binding additional context."""
        adapter = LoggerAdapter("test", user_id="123")
        new_adapter = adapter.bind(session_id="xyz")
        assert "session_id" in new_adapter._context
        assert new_adapter._context["user_id"] == "123"

    def test_unbind_context(self):
        """Test unbinding context."""
        adapter = LoggerAdapter("test", user_id="123", request_id="abc")
        new_adapter = adapter.unbind("request_id")
        assert "request_id" not in new_adapter._context
        assert "user_id" in new_adapter._context


class TestSpecializedLoggers:
    """Tests for specialized logger functions."""

    def setup_method(self):
        """Setup logging before tests."""
        setup_logging(level="DEBUG", json_format=True)

    def test_get_agent_logger(self):
        """Test agent logger creation."""
        logger = get_agent_logger("agent_001", "Test Agent")
        assert logger._context["agent_id"] == "agent_001"
        assert logger._context["agent_name"] == "Test Agent"

    def test_get_conversation_logger(self):
        """Test conversation logger creation."""
        logger = get_conversation_logger("conv_001")
        assert logger._context["conversation_id"] == "conv_001"

    def test_get_api_logger(self):
        """Test API logger creation."""
        logger = get_api_logger()
        assert logger is not None


# ============================================================================
# Exception Tests
# ============================================================================


class TestBaseException:
    """Tests for AgentOrchestratorError."""

    def test_basic_exception(self):
        """Test basic exception creation."""
        exc = AgentOrchestratorError("Test error")
        assert exc.message == "Test error"
        assert exc.details == {}
        assert exc.cause is None

    def test_exception_with_details(self):
        """Test exception with details."""
        exc = AgentOrchestratorError(
            "Test error",
            details={"key": "value"},
        )
        assert exc.details == {"key": "value"}

    def test_exception_with_cause(self):
        """Test exception with cause."""
        cause = ValueError("Original error")
        exc = AgentOrchestratorError("Test error", cause=cause)
        assert exc.cause is cause

    def test_to_dict(self):
        """Test exception to dict conversion."""
        exc = AgentOrchestratorError(
            "Test error",
            details={"key": "value"},
            cause=ValueError("cause"),
        )
        d = exc.to_dict()
        assert d["error"] == "AgentOrchestratorError"
        assert d["message"] == "Test error"
        assert d["details"] == {"key": "value"}
        assert "cause" in d


class TestConfigurationErrors:
    """Tests for configuration exception classes."""

    def test_missing_configuration_error(self):
        """Test MissingConfigurationError."""
        exc = MissingConfigurationError("API_KEY")
        assert "API_KEY" in exc.message
        assert exc.config_key == "API_KEY"

    def test_invalid_configuration_error(self):
        """Test InvalidConfigurationError."""
        exc = InvalidConfigurationError("PORT", "invalid")
        assert "PORT" in exc.message
        assert exc.config_key == "PORT"
        assert exc.value == "invalid"


class TestAPIErrors:
    """Tests for API exception classes."""

    def test_bad_request_error(self):
        """Test BadRequestError."""
        exc = BadRequestError("Invalid input")
        assert exc.status_code == 400

    def test_unauthorized_error(self):
        """Test UnauthorizedError."""
        exc = UnauthorizedError()
        assert exc.status_code == 401

    def test_forbidden_error(self):
        """Test ForbiddenError."""
        exc = ForbiddenError()
        assert exc.status_code == 403

    def test_not_found_error(self):
        """Test NotFoundError."""
        exc = NotFoundError("Agent", "agent_001")
        assert exc.status_code == 404
        assert exc.resource_type == "Agent"
        assert exc.resource_id == "agent_001"

    def test_conflict_error(self):
        """Test ConflictError."""
        exc = ConflictError("Resource already exists")
        assert exc.status_code == 409

    def test_validation_error(self):
        """Test ValidationError."""
        errors = [{"field": "name", "message": "required"}]
        exc = ValidationError("Validation failed", errors=errors)
        assert exc.status_code == 422
        assert exc.errors == errors

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        exc = RateLimitError(retry_after=60)
        assert exc.status_code == 429
        assert exc.retry_after == 60

    def test_internal_server_error(self):
        """Test InternalServerError."""
        exc = InternalServerError("Something went wrong")
        assert exc.status_code == 500

    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError."""
        exc = ServiceUnavailableError("Redis")
        assert exc.status_code == 503
        assert exc.service_name == "Redis"


class TestLLMErrors:
    """Tests for LLM exception classes."""

    def test_llm_api_error(self):
        """Test LLMAPIError."""
        exc = LLMAPIError("API call failed", model="claude-3-opus")
        assert exc.provider == "anthropic"
        assert exc.model == "claude-3-opus"

    def test_llm_rate_limit_error(self):
        """Test LLMRateLimitError."""
        exc = LLMRateLimitError(retry_after=30)
        assert exc.retry_after == 30

    def test_llm_timeout_error(self):
        """Test LLMTimeoutError."""
        exc = LLMTimeoutError(timeout_seconds=60)
        assert exc.timeout_seconds == 60


class TestTimeoutErrors:
    """Tests for timeout exception classes."""

    def test_agent_timeout_error(self):
        """Test AgentTimeoutError."""
        exc = AgentTimeoutError("agent_001", timeout_seconds=120)
        assert exc.agent_id == "agent_001"
        assert exc.timeout_seconds == 120

    def test_conversation_timeout_error(self):
        """Test ConversationTimeoutError."""
        exc = ConversationTimeoutError("conv_001", timeout_seconds=300)
        assert exc.conversation_id == "conv_001"
        assert exc.timeout_seconds == 300


# ============================================================================
# Observability Tests
# ============================================================================


class TestLangfuseClient:
    """Tests for LangfuseClient class."""

    def setup_method(self):
        """Reset observability before each test."""
        reset_observability()

    def teardown_method(self):
        """Reset observability after each test."""
        reset_observability()

    def test_disabled_client(self):
        """Test disabled client does nothing."""
        client = LangfuseClient(enabled=False)
        assert client.enabled is False
        assert client.start_trace("trace_1", "test") is None
        assert client.start_span("span_1", "trace_1", "test") is None

    def test_client_without_credentials(self):
        """Test client without credentials is disabled."""
        client = LangfuseClient(enabled=True, public_key="", secret_key="")
        assert client.enabled is False

    def test_trace_context_manager_disabled(self):
        """Test trace context manager when disabled."""
        client = LangfuseClient(enabled=False)
        with client.trace_context("trace_1", "test") as trace_id:
            assert trace_id is None

    def test_span_context_manager_disabled(self):
        """Test span context manager when disabled."""
        client = LangfuseClient(enabled=False)
        with client.span_context("span_1", "trace_1", "test") as span_id:
            assert span_id is None


class TestGlobalObservability:
    """Tests for global observability functions."""

    def setup_method(self):
        """Reset observability before each test."""
        reset_observability()

    def teardown_method(self):
        """Reset observability after each test."""
        reset_observability()

    def test_get_observability_client_not_initialized(self):
        """Test get_observability_client returns disabled client when not initialized."""
        client = get_observability_client()
        assert client is not None
        assert client.enabled is False

    def test_init_observability(self):
        """Test init_observability creates global client."""
        client = init_observability(enabled=False)
        assert client is not None
        assert get_observability_client() is client

    def test_reset_observability(self):
        """Test reset_observability clears global client."""
        init_observability(enabled=False)
        reset_observability()
        client = get_observability_client()
        # Should return a new disabled client
        assert client.enabled is False
