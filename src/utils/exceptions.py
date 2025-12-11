"""Custom exception classes for the Agent Orchestrator system.

This module provides a unified exception hierarchy for the application.
Specific exception classes are defined in their respective modules,
but this module re-exports them and provides additional utility exceptions.
"""

from typing import Any


class AgentOrchestratorError(Exception):
    """Base exception for all Agent Orchestrator errors.

    All custom exceptions in this system should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        """Initialize the exception.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
            cause: Optional original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result: dict[str, Any] = {
            "error": self.__class__.__name__,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        if self.cause:
            result["cause"] = str(self.cause)
        return result


# ============================================================================
# Configuration Errors
# ============================================================================


class ConfigurationError(AgentOrchestratorError):
    """Raised when there's a configuration error."""

    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when a required configuration is missing."""

    def __init__(self, config_key: str, message: str | None = None):
        self.config_key = config_key
        msg = message or f"Missing required configuration: {config_key}"
        super().__init__(msg, details={"config_key": config_key})


class InvalidConfigurationError(ConfigurationError):
    """Raised when a configuration value is invalid."""

    def __init__(self, config_key: str, value: Any, message: str | None = None):
        self.config_key = config_key
        self.value = value
        msg = message or f"Invalid configuration value for {config_key}: {value}"
        super().__init__(msg, details={"config_key": config_key, "value": str(value)})


# ============================================================================
# API Errors
# ============================================================================


class APIError(AgentOrchestratorError):
    """Base class for API-related errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, details, cause)
        self.status_code = status_code


class BadRequestError(APIError):
    """Raised for bad request errors (400)."""

    def __init__(
        self,
        message: str = "Bad request",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, status_code=400, details=details)


class UnauthorizedError(APIError):
    """Raised for authentication errors (401)."""

    def __init__(
        self,
        message: str = "Unauthorized",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, status_code=401, details=details)


class ForbiddenError(APIError):
    """Raised for authorization errors (403)."""

    def __init__(
        self,
        message: str = "Forbidden",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, status_code=403, details=details)


class NotFoundError(APIError):
    """Raised when a resource is not found (404)."""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        message: str | None = None,
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id
        msg = message or f"{resource_type} not found: {resource_id}"
        super().__init__(
            msg,
            status_code=404,
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class ConflictError(APIError):
    """Raised for conflict errors (409)."""

    def __init__(
        self,
        message: str = "Resource conflict",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, status_code=409, details=details)


class ValidationError(APIError):
    """Raised for validation errors (422)."""

    def __init__(
        self,
        message: str = "Validation error",
        errors: list[dict[str, Any]] | None = None,
    ):
        details = {"validation_errors": errors} if errors else None
        super().__init__(message, status_code=422, details=details)
        self.errors = errors or []


class RateLimitError(APIError):
    """Raised when rate limit is exceeded (429)."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
    ):
        details = {"retry_after": retry_after} if retry_after else None
        super().__init__(message, status_code=429, details=details)
        self.retry_after = retry_after


class InternalServerError(APIError):
    """Raised for internal server errors (500)."""

    def __init__(
        self,
        message: str = "Internal server error",
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, status_code=500, details=details, cause=cause)


class ServiceUnavailableError(APIError):
    """Raised when a service is unavailable (503)."""

    def __init__(
        self,
        service_name: str,
        message: str | None = None,
    ):
        msg = message or f"Service unavailable: {service_name}"
        super().__init__(msg, status_code=503, details={"service": service_name})
        self.service_name = service_name


# ============================================================================
# LLM/External Service Errors
# ============================================================================


class ExternalServiceError(AgentOrchestratorError):
    """Base class for external service errors."""

    pass


class LLMAPIError(ExternalServiceError):
    """Raised when LLM API call fails."""

    def __init__(
        self,
        message: str,
        provider: str = "anthropic",
        model: str | None = None,
        cause: Exception | None = None,
    ):
        details = {"provider": provider}
        if model:
            details["model"] = model
        super().__init__(message, details=details, cause=cause)
        self.provider = provider
        self.model = model


class LLMRateLimitError(LLMAPIError):
    """Raised when LLM API rate limit is exceeded."""

    def __init__(
        self,
        retry_after: int | None = None,
        provider: str = "anthropic",
        cause: Exception | None = None,
    ):
        super().__init__(
            "LLM API rate limit exceeded",
            provider=provider,
            cause=cause,
        )
        self.retry_after = retry_after
        if retry_after:
            self.details["retry_after"] = retry_after


class LLMTimeoutError(LLMAPIError):
    """Raised when LLM API call times out."""

    def __init__(
        self,
        timeout_seconds: int,
        provider: str = "anthropic",
        cause: Exception | None = None,
    ):
        super().__init__(
            f"LLM API call timed out after {timeout_seconds}s",
            provider=provider,
            cause=cause,
        )
        self.timeout_seconds = timeout_seconds
        self.details["timeout_seconds"] = timeout_seconds


# ============================================================================
# Timeout Errors
# ============================================================================


class TimeoutError(AgentOrchestratorError):
    """Base class for timeout errors."""

    def __init__(
        self,
        message: str,
        timeout_seconds: int,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        details["timeout_seconds"] = timeout_seconds
        super().__init__(message, details=details)
        self.timeout_seconds = timeout_seconds


class AgentTimeoutError(TimeoutError):
    """Raised when an agent operation times out."""

    def __init__(self, agent_id: str, timeout_seconds: int):
        super().__init__(
            f"Agent {agent_id} timed out after {timeout_seconds}s",
            timeout_seconds=timeout_seconds,
            details={"agent_id": agent_id},
        )
        self.agent_id = agent_id


class ConversationTimeoutError(TimeoutError):
    """Raised when a conversation times out."""

    def __init__(self, conversation_id: str, timeout_seconds: int):
        super().__init__(
            f"Conversation {conversation_id} timed out after {timeout_seconds}s",
            timeout_seconds=timeout_seconds,
            details={"conversation_id": conversation_id},
        )
        self.conversation_id = conversation_id
