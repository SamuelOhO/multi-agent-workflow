"""Utility modules for Agent Orchestrator.

This package provides utility functions and classes for:
- Configuration management
- Structured logging
- Exception handling
- LLM observability (Langfuse)
"""

from .config import (
    AgentDefaults,
    AnthropicConfig,
    AppConfig,
    AppSettings,
    Environment,
    LangfuseConfig,
    LogFormat,
    LoggingConfig,
    RateLimitConfig,
    RedisConfig,
    SecurityConfig,
    TimeoutConfig,
    get_config,
    init_config,
    reset_config,
)
from .error_handlers import (
    create_error_response,
    register_error_handlers,
)
from .exceptions import (
    AgentOrchestratorError,
    AgentTimeoutError,
    APIError,
    BadRequestError,
    ConfigurationError,
    ConflictError,
    ConversationTimeoutError,
    ExternalServiceError,
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
from .logging import (
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
from .observability import (
    LangfuseClient,
    get_observability_client,
    init_observability,
    reset_observability,
    shutdown_observability,
)

__all__ = [
    # Config
    "AppConfig",
    "AppSettings",
    "AnthropicConfig",
    "LoggingConfig",
    "RedisConfig",
    "SecurityConfig",
    "RateLimitConfig",
    "AgentDefaults",
    "TimeoutConfig",
    "LangfuseConfig",
    "Environment",
    "LogFormat",
    "get_config",
    "init_config",
    "reset_config",
    # Logging
    "setup_logging",
    "get_logger",
    "LoggerAdapter",
    "get_agent_logger",
    "get_conversation_logger",
    "get_api_logger",
    "get_correlation_id",
    "set_correlation_id",
    "clear_correlation_id",
    # Exceptions
    "AgentOrchestratorError",
    "ConfigurationError",
    "MissingConfigurationError",
    "InvalidConfigurationError",
    "APIError",
    "BadRequestError",
    "UnauthorizedError",
    "ForbiddenError",
    "NotFoundError",
    "ConflictError",
    "ValidationError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
    "ExternalServiceError",
    "LLMAPIError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "AgentTimeoutError",
    "ConversationTimeoutError",
    # Error Handlers
    "register_error_handlers",
    "create_error_response",
    # Observability
    "LangfuseClient",
    "get_observability_client",
    "init_observability",
    "shutdown_observability",
    "reset_observability",
]
