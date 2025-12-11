"""Sequential pattern implementation.

This module implements the sequential conversation pattern where
agents work one after another, passing output as input to the next.
"""

from __future__ import annotations

from typing import Any

from src.models import (
    Conversation,
    ConversationPattern,
    ConversationResult,
)

from .base import BasePattern, StageExecutionError


class SequentialPattern(BasePattern):
    """Sequential execution pattern.

    Executes stages one after another. Each stage's output becomes
    the next stage's input.

    Flow:
        [Task] → [Agent A] → [Agent B] → [Agent C] → [Result]

    Use cases:
        - Pipeline processing (research → design → implement → review)
        - Sequential workflows
        - Tasks requiring ordered execution
    """

    @property
    def pattern_type(self) -> ConversationPattern:
        """Return the pattern type."""
        return ConversationPattern.SEQUENTIAL

    async def execute(self, conversation: Conversation) -> ConversationResult:
        """Execute stages sequentially.

        Each stage's output becomes the next stage's input.

        Args:
            conversation: The conversation to execute.

        Returns:
            ConversationResult with the execution results.
        """
        self.validate_conversation(conversation)

        current_input = conversation.initial_input
        final_output: dict[str, Any] = {}

        for stage in conversation.stages:
            # Update stage input and status
            stage.input_data = current_input
            await self._mark_stage_started(conversation.id, stage.name)

            try:
                # Execute stage
                stage_output = await self._execute_single_stage(conversation, stage)

                # Mark completed and update
                await self._mark_stage_completed(
                    conversation.id, stage.name, stage_output
                )

                # Pass output to next stage
                current_input = stage_output
                final_output = stage_output

            except StageExecutionError as e:
                await self._mark_stage_failed(conversation.id, stage.name, str(e))
                return await self.conversation_manager.fail(conversation.id, str(e))

            except Exception as e:
                error_msg = f"Unexpected error in stage '{stage.name}': {e}"
                await self._mark_stage_failed(conversation.id, stage.name, error_msg)
                return await self.conversation_manager.fail(conversation.id, error_msg)

        return await self.conversation_manager.complete(conversation.id, final_output)
