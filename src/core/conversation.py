"""Conversation Manager - Manages conversation lifecycle and state.

This module handles creation, tracking, and state management of conversations.
"""

import asyncio
from typing import Any

from src.models import (
    Conversation,
    ConversationPattern,
    ConversationResult,
    ConversationStage,
    ConversationStatus,
    Message,
)


class ConversationNotFoundError(Exception):
    """Raised when a conversation is not found."""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        super().__init__(f"Conversation not found: {conversation_id}")


class ConversationAlreadyExistsError(Exception):
    """Raised when trying to create a conversation that already exists."""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        super().__init__(f"Conversation already exists: {conversation_id}")


class ConversationStateError(Exception):
    """Raised when an operation is invalid for the current conversation state."""

    def __init__(self, conversation_id: str, current_state: str, operation: str):
        self.conversation_id = conversation_id
        self.current_state = current_state
        self.operation = operation
        super().__init__(
            f"Cannot {operation} conversation {conversation_id} in state {current_state}"
        )


class ConversationManager:
    """Manages conversation lifecycle and state.

    Responsible for creating, tracking, and updating conversations.
    Thread-safe for concurrent access.
    """

    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        name: str = "",
        description: str = "",
        pattern: ConversationPattern = ConversationPattern.SEQUENTIAL,
        stages: list[ConversationStage] | None = None,
        initial_input: dict[str, Any] | None = None,
        timeout_seconds: int = 300,
        max_iterations: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> Conversation:
        """Create a new conversation.

        Args:
            name: Name of the conversation.
            description: Description of the conversation.
            pattern: The conversation pattern to use.
            stages: List of conversation stages.
            initial_input: Initial input data from user.
            timeout_seconds: Timeout for the entire conversation.
            max_iterations: Maximum iterations for debate pattern.
            metadata: Additional metadata.

        Returns:
            The created Conversation object.
        """
        conversation = Conversation(
            name=name,
            description=description,
            pattern=pattern,
            stages=stages or [],
            initial_input=initial_input or {},
            timeout_seconds=timeout_seconds,
            max_iterations=max_iterations,
            metadata=metadata or {},
        )

        async with self._lock:
            if conversation.id in self._conversations:
                raise ConversationAlreadyExistsError(conversation.id)
            self._conversations[conversation.id] = conversation

        return conversation

    async def create_from_config(
        self,
        config: dict[str, Any],
        initial_input: dict[str, Any] | None = None,
    ) -> Conversation:
        """Create a conversation from a configuration dictionary.

        Args:
            config: Configuration dictionary (typically from YAML).
            initial_input: Initial input data from user.

        Returns:
            The created Conversation object.
        """
        stages = []
        for stage_config in config.get("stages", []):
            stages.append(
                ConversationStage(
                    name=stage_config["name"],
                    description=stage_config.get("description", ""),
                    agent_capability=stage_config["agent_capability"],
                    agent_id=stage_config.get("agent_id"),
                    parallel_with=stage_config.get("parallel_with"),
                )
            )

        pattern_str = config.get("pattern", "sequential")
        pattern = ConversationPattern(pattern_str)

        return await self.create(
            name=config.get("name", ""),
            description=config.get("description", ""),
            pattern=pattern,
            stages=stages,
            initial_input=initial_input,
            timeout_seconds=config.get("timeout_seconds", 300),
            max_iterations=config.get("max_iterations", 3),
            metadata=config.get("metadata", {}),
        )

    async def get(self, conversation_id: str) -> Conversation:
        """Get a conversation by ID.

        Args:
            conversation_id: The ID of the conversation.

        Returns:
            The Conversation object.

        Raises:
            ConversationNotFoundError: If the conversation is not found.
        """
        async with self._lock:
            if conversation_id not in self._conversations:
                raise ConversationNotFoundError(conversation_id)
            return self._conversations[conversation_id]

    async def get_messages(self, conversation_id: str) -> list[Message]:
        """Get all messages for a conversation.

        Args:
            conversation_id: The ID of the conversation.

        Returns:
            List of messages in the conversation.

        Raises:
            ConversationNotFoundError: If the conversation is not found.
        """
        conversation = await self.get(conversation_id)
        return list(conversation.messages)

    async def add_message(self, conversation_id: str, message: Message) -> None:
        """Add a message to a conversation.

        Args:
            conversation_id: The ID of the conversation.
            message: The message to add.

        Raises:
            ConversationNotFoundError: If the conversation is not found.
        """
        conversation = await self.get(conversation_id)
        async with self._lock:
            conversation.add_message(message)

    async def update_status(
        self, conversation_id: str, status: ConversationStatus
    ) -> Conversation:
        """Update the status of a conversation.

        Args:
            conversation_id: The ID of the conversation.
            status: The new status.

        Returns:
            The updated Conversation object.

        Raises:
            ConversationNotFoundError: If the conversation is not found.
        """
        conversation = await self.get(conversation_id)

        async with self._lock:
            if status == ConversationStatus.IN_PROGRESS:
                conversation.mark_started()
            elif status == ConversationStatus.COMPLETED:
                conversation.mark_completed()
            elif status == ConversationStatus.FAILED:
                conversation.mark_failed("Status set to failed")
            elif status == ConversationStatus.CANCELLED:
                conversation.status = ConversationStatus.CANCELLED
            else:
                conversation.status = status

        return conversation

    async def start(self, conversation_id: str) -> Conversation:
        """Start a conversation.

        Args:
            conversation_id: The ID of the conversation.

        Returns:
            The updated Conversation object.

        Raises:
            ConversationNotFoundError: If the conversation is not found.
            ConversationStateError: If the conversation cannot be started.
        """
        conversation = await self.get(conversation_id)

        if conversation.status != ConversationStatus.PENDING:
            raise ConversationStateError(
                conversation_id, conversation.status.value, "start"
            )

        async with self._lock:
            conversation.mark_started()

        return conversation

    async def complete(
        self, conversation_id: str, output: dict[str, Any] | None = None
    ) -> ConversationResult:
        """Mark a conversation as completed.

        Args:
            conversation_id: The ID of the conversation.
            output: Final output data.

        Returns:
            ConversationResult with the conversation results.

        Raises:
            ConversationNotFoundError: If the conversation is not found.
        """
        conversation = await self.get(conversation_id)

        async with self._lock:
            conversation.mark_completed(output)

        return ConversationResult.from_conversation(conversation)

    async def fail(self, conversation_id: str, error: str) -> ConversationResult:
        """Mark a conversation as failed.

        Args:
            conversation_id: The ID of the conversation.
            error: Error message.

        Returns:
            ConversationResult with the conversation results.

        Raises:
            ConversationNotFoundError: If the conversation is not found.
        """
        conversation = await self.get(conversation_id)

        async with self._lock:
            conversation.mark_failed(error)

        return ConversationResult.from_conversation(conversation)

    async def cancel(self, conversation_id: str) -> ConversationResult:
        """Cancel a conversation.

        Args:
            conversation_id: The ID of the conversation.

        Returns:
            ConversationResult with the conversation results.

        Raises:
            ConversationNotFoundError: If the conversation is not found.
            ConversationStateError: If the conversation cannot be cancelled.
        """
        conversation = await self.get(conversation_id)

        if conversation.is_finished():
            raise ConversationStateError(
                conversation_id, conversation.status.value, "cancel"
            )

        async with self._lock:
            conversation.status = ConversationStatus.CANCELLED

        return ConversationResult.from_conversation(conversation)

    async def update_stage(
        self,
        conversation_id: str,
        stage_name: str,
        status: ConversationStatus | None = None,
        output: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> ConversationStage:
        """Update a conversation stage.

        Args:
            conversation_id: The ID of the conversation.
            stage_name: The name of the stage to update.
            status: New status for the stage.
            output: Output data from the stage.
            error: Error message if the stage failed.

        Returns:
            The updated ConversationStage.

        Raises:
            ConversationNotFoundError: If the conversation is not found.
            ValueError: If the stage is not found.
        """
        conversation = await self.get(conversation_id)
        stage = conversation.get_stage_by_name(stage_name)

        if stage is None:
            raise ValueError(f"Stage not found: {stage_name}")

        async with self._lock:
            if status == ConversationStatus.IN_PROGRESS:
                stage.mark_started()
            elif status == ConversationStatus.COMPLETED:
                stage.mark_completed(output)
            elif status == ConversationStatus.FAILED and error:
                stage.mark_failed(error)
            elif status:
                stage.status = status

        return stage

    async def advance_stage(self, conversation_id: str) -> ConversationStage | None:
        """Advance to the next stage in the conversation.

        Args:
            conversation_id: The ID of the conversation.

        Returns:
            The next ConversationStage, or None if at the end.

        Raises:
            ConversationNotFoundError: If the conversation is not found.
        """
        conversation = await self.get(conversation_id)

        async with self._lock:
            if conversation.advance_stage():
                return conversation.get_current_stage()
            return None

    async def list_all(
        self, status: ConversationStatus | None = None
    ) -> list[Conversation]:
        """List all conversations, optionally filtered by status.

        Args:
            status: Optional status filter.

        Returns:
            List of conversations.
        """
        async with self._lock:
            if status:
                return [
                    conv
                    for conv in self._conversations.values()
                    if conv.status == status
                ]
            return list(self._conversations.values())

    async def list_active(self) -> list[Conversation]:
        """List all active (in-progress) conversations.

        Returns:
            List of active conversations.
        """
        return await self.list_all(status=ConversationStatus.IN_PROGRESS)

    async def delete(self, conversation_id: str) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: The ID of the conversation to delete.

        Returns:
            True if deleted, False if not found.
        """
        async with self._lock:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
                return True
            return False

    async def cleanup_finished(self, keep_count: int = 100) -> int:
        """Clean up old finished conversations.

        Args:
            keep_count: Number of finished conversations to keep.

        Returns:
            Number of conversations deleted.
        """
        async with self._lock:
            finished = [
                conv for conv in self._conversations.values() if conv.is_finished()
            ]

            # Sort by completion time, oldest first
            finished.sort(key=lambda c: c.completed_at or c.created_at)

            # Delete oldest beyond keep_count
            to_delete = finished[:-keep_count] if len(finished) > keep_count else []
            deleted_count = 0

            for conv in to_delete:
                del self._conversations[conv.id]
                deleted_count += 1

            return deleted_count

    def __len__(self) -> int:
        """Return the number of conversations."""
        return len(self._conversations)

    def __contains__(self, conversation_id: str) -> bool:
        """Check if a conversation exists."""
        return conversation_id in self._conversations
