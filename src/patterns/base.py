"""Base pattern class for conversation patterns.

This module defines the abstract base class for all conversation patterns.
Patterns control how agents collaborate to complete tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.models import (
    Conversation,
    ConversationPattern,
    ConversationResult,
    ConversationStage,
    ConversationStatus,
    Message,
)

if TYPE_CHECKING:
    from src.agents.base import BaseAgent
    from src.core.conversation import ConversationManager
    from src.core.registry import AgentRegistry


class PatternError(Exception):
    """Base exception for pattern-related errors."""

    pass


class NoAgentAvailableError(PatternError):
    """Raised when no agent is available for a capability."""

    def __init__(self, capability: str):
        self.capability = capability
        super().__init__(f"No agent available for capability: {capability}")


class StageExecutionError(PatternError):
    """Raised when stage execution fails."""

    def __init__(self, stage_name: str, message: str):
        self.stage_name = stage_name
        super().__init__(f"Stage '{stage_name}' failed: {message}")


class PatternValidationError(PatternError):
    """Raised when pattern validation fails."""

    pass


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for agent compatibility with patterns."""

    @property
    def config(self) -> Any: ...

    async def process(self, message: Message) -> Message: ...

    def can_handle(self, capability: str) -> bool: ...


class BasePattern(ABC):
    """Abstract base class for conversation patterns.

    Patterns define how agents collaborate to complete tasks.
    Different patterns support different collaboration styles:
    - Sequential: Agents work one after another
    - Parallel: Agents work simultaneously
    - Debate: Agents discuss and reach consensus
    - Router: Task is routed to appropriate agent
    """

    ORCHESTRATOR_ID = "orchestrator"

    def __init__(
        self,
        registry: AgentRegistry,
        conversation_manager: ConversationManager,
    ):
        """Initialize the pattern.

        Args:
            registry: Agent registry for agent lookup.
            conversation_manager: Manager for conversation state.
        """
        self.registry = registry
        self.conversation_manager = conversation_manager

    @property
    @abstractmethod
    def pattern_type(self) -> ConversationPattern:
        """Return the pattern type."""
        pass

    @abstractmethod
    async def execute(self, conversation: Conversation) -> ConversationResult:
        """Execute the pattern on a conversation.

        Args:
            conversation: The conversation to execute.

        Returns:
            ConversationResult with the execution results.
        """
        pass

    def validate_conversation(self, conversation: Conversation) -> None:
        """Validate that the conversation is suitable for this pattern.

        Args:
            conversation: The conversation to validate.

        Raises:
            PatternValidationError: If validation fails.
        """
        if conversation.pattern != self.pattern_type:
            raise PatternValidationError(
                f"Conversation pattern {conversation.pattern} does not match "
                f"pattern type {self.pattern_type}"
            )

    async def _select_agent(self, stage: ConversationStage) -> BaseAgent:
        """Select an agent for a stage.

        Args:
            stage: The stage requiring an agent.

        Returns:
            Selected agent.

        Raises:
            NoAgentAvailableError: If no agent is available.
        """
        from src.core.registry import AgentNotFoundError

        # If specific agent ID is set, use that
        if stage.agent_id:
            try:
                return await self.registry.get(stage.agent_id)
            except AgentNotFoundError as e:
                raise NoAgentAvailableError(
                    f"Specified agent {stage.agent_id} not found"
                ) from e

        # Otherwise, find by capability
        agent = await self.registry.find_one_by_capability(stage.agent_capability)
        if agent is None:
            raise NoAgentAvailableError(stage.agent_capability)

        return agent

    async def _execute_single_stage(
        self,
        conversation: Conversation,
        stage: ConversationStage,
    ) -> dict[str, Any]:
        """Execute a single stage and return its output.

        Args:
            conversation: The conversation.
            stage: The stage to execute.

        Returns:
            Stage output dictionary.

        Raises:
            StageExecutionError: If stage execution fails.
        """
        try:
            agent = await self._select_agent(stage)

            message = Message.create_task(
                sender_id=self.ORCHESTRATOR_ID,
                recipient_id=agent.config.agent_id,
                content={
                    "task": stage.input_data,
                    "stage": stage.name,
                    "capability": stage.agent_capability,
                },
                correlation_id=conversation.id,
            )

            await self.conversation_manager.add_message(conversation.id, message)
            response = await agent.process(message)
            await self.conversation_manager.add_message(conversation.id, response)

            result = response.content.get("result", response.content)
            if isinstance(result, dict):
                return result
            return {"result": result}

        except NoAgentAvailableError:
            raise
        except Exception as e:
            raise StageExecutionError(stage.name, str(e)) from e

    async def _mark_stage_started(self, conversation_id: str, stage_name: str) -> None:
        """Mark a stage as started.

        Args:
            conversation_id: The conversation ID.
            stage_name: The stage name.
        """
        await self.conversation_manager.update_stage(
            conversation_id, stage_name, status=ConversationStatus.IN_PROGRESS
        )

    async def _mark_stage_completed(
        self,
        conversation_id: str,
        stage_name: str,
        output: dict[str, Any],
    ) -> None:
        """Mark a stage as completed.

        Args:
            conversation_id: The conversation ID.
            stage_name: The stage name.
            output: The stage output.
        """
        await self.conversation_manager.update_stage(
            conversation_id,
            stage_name,
            status=ConversationStatus.COMPLETED,
            output=output,
        )

    async def _mark_stage_failed(
        self,
        conversation_id: str,
        stage_name: str,
        error: str,
    ) -> None:
        """Mark a stage as failed.

        Args:
            conversation_id: The conversation ID.
            stage_name: The stage name.
            error: The error message.
        """
        await self.conversation_manager.update_stage(
            conversation_id,
            stage_name,
            status=ConversationStatus.FAILED,
            error=error,
        )


def get_pattern_class(pattern: ConversationPattern) -> type[BasePattern]:
    """Get the pattern class for a given pattern type.

    Args:
        pattern: The conversation pattern type.

    Returns:
        The pattern class.

    Raises:
        ValueError: If pattern type is not supported.
    """
    from .debate import DebatePattern
    from .hierarchical import HierarchicalPattern
    from .parallel import ParallelPattern
    from .router import RouterPattern
    from .sequential import SequentialPattern

    pattern_map: dict[ConversationPattern, type[BasePattern]] = {
        ConversationPattern.SEQUENTIAL: SequentialPattern,
        ConversationPattern.PARALLEL: ParallelPattern,
        ConversationPattern.ROUTER: RouterPattern,
        ConversationPattern.DEBATE: DebatePattern,
        ConversationPattern.HIERARCHICAL: HierarchicalPattern,
    }

    pattern_class = pattern_map.get(pattern)
    if pattern_class is None:
        raise ValueError(f"Unsupported pattern type: {pattern}")

    return pattern_class


def create_pattern(
    pattern: ConversationPattern,
    registry: AgentRegistry,
    conversation_manager: ConversationManager,
) -> BasePattern:
    """Create a pattern instance for the given pattern type.

    Args:
        pattern: The conversation pattern type.
        registry: Agent registry.
        conversation_manager: Conversation manager.

    Returns:
        Pattern instance.
    """
    pattern_class = get_pattern_class(pattern)
    return pattern_class(registry, conversation_manager)
