"""Structured logging configuration.

This module provides structured logging using structlog with JSON and console formatters.
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import structlog
from structlog.types import EventDict, Processor

# Context variable for correlation ID (request tracking across async calls)
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Set a correlation ID in context.

    Args:
        correlation_id: Optional correlation ID. If not provided, generates a new UUID.

    Returns:
        The correlation ID that was set.
    """
    if correlation_id is None:
        correlation_id = str(uuid4())
    correlation_id_var.set(correlation_id)
    return correlation_id


def clear_correlation_id() -> None:
    """Clear the correlation ID from context."""
    correlation_id_var.set(None)


def add_correlation_id(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Processor that adds correlation ID to log events."""
    correlation_id = get_correlation_id()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def add_app_context(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Processor that adds application context to log events."""
    event_dict.setdefault("app", "agent-orchestrator")
    return event_dict


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: str | None = None,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, use JSON format; otherwise use console format
        log_file: Optional file path to write logs to
    """
    # Convert string level to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Shared processors for all outputs
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_correlation_id,
        add_app_context,
    ]

    if json_format:
        # JSON format for production
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.rich_traceback,
            ),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Add file handler if specified
    handlers: list[logging.Handler] = [handler]
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
        force=True,
    )

    # Set levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Optional logger name. If not provided, uses the caller's module name.

    Returns:
        A configured structlog BoundLogger instance.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


class LoggerAdapter:
    """Adapter for logging with additional context.

    This class provides a convenient way to add persistent context to log messages.
    """

    def __init__(self, name: str | None = None, **initial_context: Any):
        """Initialize the logger adapter.

        Args:
            name: Logger name
            **initial_context: Initial context to bind to all log messages
        """
        self._logger = get_logger(name)
        self._context = initial_context

    def bind(self, **new_context: Any) -> "LoggerAdapter":
        """Create a new adapter with additional context.

        Args:
            **new_context: Additional context to bind

        Returns:
            New LoggerAdapter with merged context
        """
        merged = {**self._context, **new_context}
        adapter = LoggerAdapter.__new__(LoggerAdapter)
        adapter._logger = self._logger
        adapter._context = merged
        return adapter

    def unbind(self, *keys: str) -> "LoggerAdapter":
        """Create a new adapter with specified keys removed.

        Args:
            *keys: Keys to remove from context

        Returns:
            New LoggerAdapter with keys removed
        """
        new_context = {k: v for k, v in self._context.items() if k not in keys}
        adapter = LoggerAdapter.__new__(LoggerAdapter)
        adapter._logger = self._logger
        adapter._context = new_context
        return adapter

    def _log(self, level: str, event: str, **kwargs: Any) -> None:
        """Internal logging method."""
        merged = {**self._context, **kwargs}
        log_method = getattr(self._logger, level)
        log_method(event, **merged)

    def debug(self, event: str, **kwargs: Any) -> None:
        """Log a debug message."""
        self._log("debug", event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        """Log an info message."""
        self._log("info", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self._log("warning", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        """Log an error message."""
        self._log("error", event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        """Log a critical message."""
        self._log("critical", event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        """Log an exception with traceback."""
        self._log("exception", event, **kwargs)


# Convenience function for creating agent-specific loggers
def get_agent_logger(agent_id: str, agent_name: str | None = None) -> LoggerAdapter:
    """Get a logger configured for an agent.

    Args:
        agent_id: The agent's unique identifier
        agent_name: Optional agent name

    Returns:
        LoggerAdapter with agent context bound
    """
    context: dict[str, Any] = {"agent_id": agent_id}
    if agent_name:
        context["agent_name"] = agent_name
    return LoggerAdapter("agent", **context)


def get_conversation_logger(conversation_id: str) -> LoggerAdapter:
    """Get a logger configured for a conversation.

    Args:
        conversation_id: The conversation's unique identifier

    Returns:
        LoggerAdapter with conversation context bound
    """
    return LoggerAdapter("conversation", conversation_id=conversation_id)


def get_api_logger() -> LoggerAdapter:
    """Get a logger configured for API operations.

    Returns:
        LoggerAdapter for API logging
    """
    return LoggerAdapter("api")
