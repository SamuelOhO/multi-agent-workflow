"""LLM Observability with Langfuse.

This module provides integration with Langfuse for tracing LLM calls,
agent interactions, and conversation flows.
"""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from .logging import get_logger

logger = get_logger(__name__)

# Try to import langfuse, but make it optional
try:
    from langfuse import Langfuse

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None  # type: ignore[misc, assignment]


class LangfuseClient:
    """Wrapper for Langfuse client with graceful degradation.

    This class provides a unified interface for Langfuse operations,
    with graceful handling when Langfuse is not available or disabled.
    """

    def __init__(
        self,
        public_key: str = "",
        secret_key: str = "",
        host: str = "https://cloud.langfuse.com",
        enabled: bool = True,
    ):
        """Initialize the Langfuse client.

        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL
            enabled: Whether to enable Langfuse tracking
        """
        self.enabled = enabled and LANGFUSE_AVAILABLE
        self._client: Any = None
        self._traces: dict[str, Any] = {}
        self._spans: dict[str, Any] = {}

        if self.enabled and public_key and secret_key:
            try:
                self._client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                logger.info(
                    "Langfuse client initialized",
                    host=host,
                )
            except Exception as e:
                logger.warning(
                    "Failed to initialize Langfuse client",
                    error=str(e),
                )
                self.enabled = False
        elif enabled and not LANGFUSE_AVAILABLE:
            logger.warning(
                "Langfuse package not installed. Install with: pip install langfuse"
            )
            self.enabled = False
        elif enabled and (not public_key or not secret_key):
            logger.debug("Langfuse credentials not provided, tracking disabled")
            self.enabled = False

    def start_trace(
        self,
        trace_id: str,
        name: str,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> str | None:
        """Start a new trace for a conversation or task.

        Args:
            trace_id: Unique identifier for the trace (e.g., conversation_id)
            name: Name of the trace
            metadata: Optional metadata to attach
            user_id: Optional user identifier
            session_id: Optional session identifier
            tags: Optional tags for categorization

        Returns:
            The trace ID if successful, None otherwise
        """
        if not self.enabled or not self._client:
            return None

        try:
            trace = self._client.trace(
                id=trace_id,
                name=name,
                metadata=metadata or {},
                user_id=user_id,
                session_id=session_id,
                tags=tags or [],
            )
            self._traces[trace_id] = trace
            logger.debug("Started trace", trace_id=trace_id, name=name)
            return trace_id
        except Exception as e:
            logger.warning("Failed to start trace", error=str(e), trace_id=trace_id)
            return None

    def end_trace(
        self,
        trace_id: str,
        output: dict[str, Any] | None = None,
        status: str = "success",
    ) -> None:
        """End a trace and record the final output.

        Args:
            trace_id: The trace ID to end
            output: Optional final output
            status: Status of the trace (success, error)
        """
        if not self.enabled or trace_id not in self._traces:
            return

        try:
            trace = self._traces[trace_id]
            trace.update(
                output=output,
                metadata={"status": status},
            )
            del self._traces[trace_id]
            logger.debug("Ended trace", trace_id=trace_id, status=status)
        except Exception as e:
            logger.warning("Failed to end trace", error=str(e), trace_id=trace_id)

    def start_span(
        self,
        span_id: str,
        trace_id: str,
        name: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
    ) -> str | None:
        """Start a new span within a trace.

        Args:
            span_id: Unique identifier for the span
            trace_id: Parent trace ID
            name: Name of the span
            input_data: Input data for this span
            metadata: Optional metadata
            parent_span_id: Optional parent span ID for nested spans

        Returns:
            The span ID if successful, None otherwise
        """
        if not self.enabled or not self._client:
            return None

        try:
            # Get parent (trace or span)
            if parent_span_id and parent_span_id in self._spans:
                parent = self._spans[parent_span_id]
            elif trace_id in self._traces:
                parent = self._traces[trace_id]
            else:
                # Create trace if it doesn't exist
                self.start_trace(trace_id, name=f"trace_{trace_id}")
                parent = self._traces.get(trace_id)
                if not parent:
                    return None

            span = parent.span(
                id=span_id,
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
            self._spans[span_id] = span
            logger.debug(
                "Started span",
                span_id=span_id,
                trace_id=trace_id,
                name=name,
            )
            return span_id
        except Exception as e:
            logger.warning("Failed to start span", error=str(e), span_id=span_id)
            return None

    def end_span(
        self,
        span_id: str,
        output: dict[str, Any] | None = None,
        status: str = "success",
        level: str = "DEFAULT",
    ) -> None:
        """End a span and record the output.

        Args:
            span_id: The span ID to end
            output: Output data from this span
            status: Status of the span (success, error)
            level: Log level (DEBUG, DEFAULT, WARNING, ERROR)
        """
        if not self.enabled or span_id not in self._spans:
            return

        try:
            span = self._spans[span_id]
            span.update(
                output=output,
                level=level,
                metadata={"status": status},
            )
            span.end()
            del self._spans[span_id]
            logger.debug("Ended span", span_id=span_id, status=status)
        except Exception as e:
            logger.warning("Failed to end span", error=str(e), span_id=span_id)

    def log_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        input_messages: list[dict[str, Any]],
        output: str | dict[str, Any],
        model_parameters: dict[str, Any] | None = None,
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
    ) -> None:
        """Log an LLM generation event.

        Args:
            trace_id: Parent trace ID
            name: Name of the generation
            model: Model name (e.g., "claude-haiku-4-5-20251001")
            input_messages: Input messages to the LLM
            output: LLM output
            model_parameters: Optional model parameters (temperature, max_tokens, etc.)
            usage: Optional token usage (prompt_tokens, completion_tokens, total_tokens)
            metadata: Optional additional metadata
            parent_span_id: Optional parent span ID
        """
        if not self.enabled or not self._client:
            return

        try:
            # Get parent
            if parent_span_id and parent_span_id in self._spans:
                parent = self._spans[parent_span_id]
            elif trace_id in self._traces:
                parent = self._traces[trace_id]
            else:
                logger.warning(
                    "Cannot log generation: trace not found", trace_id=trace_id
                )
                return

            parent.generation(
                name=name,
                model=model,
                input=input_messages,
                output=output,
                model_parameters=model_parameters or {},
                usage=usage,
                metadata=metadata or {},
            )
            logger.debug(
                "Logged generation",
                trace_id=trace_id,
                model=model,
                name=name,
            )
        except Exception as e:
            logger.warning("Failed to log generation", error=str(e), trace_id=trace_id)

    def log_message(
        self,
        trace_id: str,
        message_id: str,
        sender_id: str,
        recipient_id: str | None,
        message_type: str,
        content: dict[str, Any],
        timestamp: str | None = None,
    ) -> None:
        """Log a message exchange between agents.

        Args:
            trace_id: Parent trace ID (conversation_id)
            message_id: Unique message ID
            sender_id: Sender agent ID
            recipient_id: Recipient agent ID (None for broadcast)
            message_type: Type of message
            content: Message content
            timestamp: Optional timestamp
        """
        if not self.enabled or trace_id not in self._traces:
            return

        try:
            trace = self._traces[trace_id]
            trace.event(
                name=f"message_{message_type}",
                metadata={
                    "message_id": message_id,
                    "sender_id": sender_id,
                    "recipient_id": recipient_id,
                    "message_type": message_type,
                    "timestamp": timestamp or datetime.utcnow().isoformat(),
                },
                input=content,
            )
            logger.debug(
                "Logged message",
                trace_id=trace_id,
                message_id=message_id,
                message_type=message_type,
            )
        except Exception as e:
            logger.warning("Failed to log message", error=str(e), message_id=message_id)

    def score(
        self,
        trace_id: str,
        name: str,
        value: float | int | str,
        comment: str | None = None,
    ) -> None:
        """Add a score to a trace.

        Args:
            trace_id: The trace ID to score
            name: Name of the score (e.g., "quality", "relevance")
            value: Score value
            comment: Optional comment
        """
        if not self.enabled or trace_id not in self._traces:
            return

        try:
            self._client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
            logger.debug(
                "Added score",
                trace_id=trace_id,
                name=name,
                value=value,
            )
        except Exception as e:
            logger.warning("Failed to add score", error=str(e), trace_id=trace_id)

    @contextmanager
    def trace_context(
        self,
        trace_id: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[str | None, None, None]:
        """Context manager for automatic trace lifecycle management.

        Args:
            trace_id: Unique identifier for the trace
            name: Name of the trace
            metadata: Optional metadata

        Yields:
            The trace ID if successful, None otherwise
        """
        trace_result = self.start_trace(trace_id, name, metadata)
        try:
            yield trace_result
        except Exception as e:
            self.end_trace(trace_id, output={"error": str(e)}, status="error")
            raise
        else:
            self.end_trace(trace_id, status="success")

    @contextmanager
    def span_context(
        self,
        span_id: str,
        trace_id: str,
        name: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[str | None, None, None]:
        """Context manager for automatic span lifecycle management.

        Args:
            span_id: Unique identifier for the span
            trace_id: Parent trace ID
            name: Name of the span
            input_data: Optional input data
            metadata: Optional metadata

        Yields:
            The span ID if successful, None otherwise
        """
        span_result = self.start_span(span_id, trace_id, name, input_data, metadata)
        try:
            yield span_result
        except Exception as e:
            self.end_span(
                span_id, output={"error": str(e)}, status="error", level="ERROR"
            )
            raise
        else:
            self.end_span(span_id, status="success")

    def flush(self) -> None:
        """Flush any pending events to Langfuse."""
        if self.enabled and self._client:
            try:
                self._client.flush()
                logger.debug("Flushed Langfuse events")
            except Exception as e:
                logger.warning("Failed to flush Langfuse events", error=str(e))

    def shutdown(self) -> None:
        """Shutdown the Langfuse client."""
        if self.enabled and self._client:
            try:
                self._client.shutdown()
                logger.info("Langfuse client shutdown")
            except Exception as e:
                logger.warning("Failed to shutdown Langfuse client", error=str(e))


# Global observability client instance
_observability_client: LangfuseClient | None = None


def get_observability_client() -> LangfuseClient:
    """Get the global observability client.

    Returns:
        The global LangfuseClient instance

    Raises:
        RuntimeError: If observability has not been initialized
    """
    global _observability_client
    if _observability_client is None:
        # Return a disabled client if not initialized
        return LangfuseClient(enabled=False)
    return _observability_client


def init_observability(
    public_key: str = "",
    secret_key: str = "",
    host: str = "https://cloud.langfuse.com",
    enabled: bool = True,
) -> LangfuseClient:
    """Initialize the global observability client.

    Args:
        public_key: Langfuse public key
        secret_key: Langfuse secret key
        host: Langfuse host URL
        enabled: Whether to enable observability

    Returns:
        The initialized LangfuseClient instance
    """
    global _observability_client
    _observability_client = LangfuseClient(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        enabled=enabled,
    )
    return _observability_client


def shutdown_observability() -> None:
    """Shutdown the global observability client."""
    global _observability_client
    if _observability_client:
        _observability_client.shutdown()
        _observability_client = None


def reset_observability() -> None:
    """Reset the global observability client (mainly for testing)."""
    global _observability_client
    _observability_client = None
