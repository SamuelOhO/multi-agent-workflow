"""Base Agent - Abstract base class for all agents.

This module defines the abstract base class that all agents must inherit from.
It provides common functionality including LLM API calls and health checking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.models import AgentConfig, AgentStatus, Message
from src.llm import BaseLLMProvider, LLMProviderFactory


class AgentError(Exception):
    """Base exception for agent-related errors."""

    pass


class LLMError(AgentError):
    """Raised when LLM API call fails."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


class BaseAgent(ABC):
    """Abstract base class for all agents.

    All agents must inherit from this class and implement the required
    abstract methods. Provides common functionality for LLM interaction
    and health checking.

    Attributes:
        config: Agent configuration.
        is_active: Whether the agent is currently active.
        status: Current agent status.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the agent.

        Args:
            config: Agent configuration containing ID, capabilities, model settings, etc.
        """
        self._config = config
        self._is_active = True
        self._status = AgentStatus.ACTIVE
        self._llm_provider: BaseLLMProvider | None = None

    @property
    def config(self) -> AgentConfig:
        """Get the agent configuration."""
        return self._config

    @property
    def is_active(self) -> bool:
        """Check if the agent is active."""
        return self._is_active

    @property
    def status(self) -> AgentStatus:
        """Get the current agent status."""
        return self._status

    def activate(self) -> None:
        """Activate the agent."""
        self._is_active = True
        self._status = AgentStatus.ACTIVE

    def deactivate(self) -> None:
        """Deactivate the agent."""
        self._is_active = False
        self._status = AgentStatus.INACTIVE

    def set_busy(self) -> None:
        """Mark the agent as busy."""
        self._status = AgentStatus.BUSY

    def set_error(self) -> None:
        """Mark the agent as in error state."""
        self._status = AgentStatus.ERROR
        self._is_active = False

    @abstractmethod
    async def process(self, message: Message) -> Message:
        """Process a message and generate a response.

        This is the main method that agents must implement to handle
        incoming messages and produce responses.

        Args:
            message: The incoming message to process.

        Returns:
            A response message.
        """
        pass

    def can_handle(self, capability: str) -> bool:
        """Check if the agent can handle a specific capability.

        Args:
            capability: The capability name to check.

        Returns:
            True if the agent can handle the capability, False otherwise.
        """
        return self._config.has_capability(capability)

    def get_capabilities(self) -> list[str]:
        """Get the list of capabilities this agent supports.

        Returns:
            List of capability names.
        """
        return self._config.get_capability_names()

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on the agent.

        Returns:
            Dictionary containing health status information.
        """
        return {
            "agent_id": self._config.agent_id,
            "name": self._config.name,
            "status": "healthy" if self._is_active else "unhealthy",
            "capabilities": self.get_capabilities(),
            "model": self._config.model,
        }

    def _get_llm_provider(self) -> BaseLLMProvider:
        """Get or create the LLM provider.

        Returns:
            BaseLLMProvider instance.

        Raises:
            LLMError: If provider cannot be created.
        """
        if self._llm_provider is None:
            try:
                self._llm_provider = LLMProviderFactory.create(
                    model=self._config.model
                )
            except Exception as e:
                raise LLMError(f"Failed to create LLM provider: {e}") from e
        return self._llm_provider

    def set_llm_provider(self, provider: BaseLLMProvider) -> None:
        """Set a custom LLM provider for this agent.

        Args:
            provider: The LLM provider to use.
        """
        self._llm_provider = provider

    async def _call_llm(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Call the LLM API with the given prompt.

        Args:
            prompt: The user prompt to send to the LLM.
            context: Optional context dictionary to include in the prompt.
            system_prompt: Optional system prompt override. If not provided,
                uses the agent's configured system prompt.

        Returns:
            The LLM response text.

        Raises:
            LLMError: If the API call fails.
        """
        try:
            provider = self._get_llm_provider()

            # Build the system prompt
            effective_system_prompt = system_prompt or self._config.system_prompt or ""

            # Build the user message with context
            user_message = str(prompt) if not isinstance(prompt, str) else prompt
            if context:
                context_parts = []
                for k, v in context.items():
                    if isinstance(v, (dict, list)):
                        import json
                        v_str = json.dumps(v, ensure_ascii=False, default=str)
                    else:
                        v_str = str(v)
                    context_parts.append(f"{k}: {v_str}")
                context_str = "\n".join(context_parts)
                user_message = f"Context:\n{context_str}\n\nTask:\n{user_message}"

            # Make the API call using the provider
            response = await provider.chat(
                messages=[{"role": "user", "content": user_message}],
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                system_prompt=effective_system_prompt,
            )

            return response.content

        except Exception as e:
            raise LLMError(
                f"Error calling LLM: {e}", original_error=e
            ) from e

    def _create_response(
        self,
        original_message: Message,
        content: dict[str, Any],
    ) -> Message:
        """Create a response message to the given message.

        Args:
            original_message: The message being responded to.
            content: The response content.

        Returns:
            A new response message.
        """
        return Message.create_response(
            sender_id=self._config.agent_id,
            recipient_id=original_message.sender_id,
            content=content,
            correlation_id=original_message.correlation_id,
            parent_message_id=original_message.id,
        )

    def _create_error_response(
        self,
        original_message: Message,
        error_message: str,
        error_code: str | None = None,
    ) -> Message:
        """Create an error response message.

        Args:
            original_message: The message that caused the error.
            error_message: Description of the error.
            error_code: Optional error code.

        Returns:
            An error message.
        """
        return Message.create_error(
            sender_id=self._config.agent_id,
            recipient_id=original_message.sender_id,
            error_message=error_message,
            error_code=error_code,
            correlation_id=original_message.correlation_id,
        )


class SimpleAgent(BaseAgent):
    """A simple agent implementation for basic use cases.

    This agent processes messages by sending the task content directly
    to the LLM and returning the response. Useful for straightforward
    tasks that don't require specialized processing.
    """

    async def process(self, message: Message) -> Message:
        """Process a message using the LLM.

        Args:
            message: The incoming message to process.

        Returns:
            A response message with the LLM's output.
        """
        try:
            self.set_busy()

            # Extract task from message content
            task = message.content.get("task", "")
            if isinstance(task, dict):
                # task가 dict인 경우 (예: debate 패턴의 proposal)
                import json
                task = json.dumps(task, ensure_ascii=False, default=str)
            if not task:
                task = message.content.get("query", "")
            if not task:
                import json
                task = json.dumps(message.content, ensure_ascii=False, default=str)

            # Build context from message
            context = {}
            if message.context.history:
                context["previous_messages"] = str(message.context.history)
            if message.context.metadata:
                context.update(message.context.metadata)

            # Call LLM
            response_text = await self._call_llm(task, context)

            self.activate()

            return self._create_response(
                original_message=message,
                content={"result": response_text},
            )

        except LLMError as e:
            self.set_error()
            return self._create_error_response(
                original_message=message,
                error_message=str(e),
                error_code="LLM_ERROR",
            )
        except Exception as e:
            self.set_error()
            return self._create_error_response(
                original_message=message,
                error_message=f"Unexpected error: {e}",
                error_code="AGENT_ERROR",
            )
