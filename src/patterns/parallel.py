"""Parallel pattern implementation.

This module implements the parallel conversation pattern where
multiple agents work simultaneously and results are merged.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.models import (
    Conversation,
    ConversationPattern,
    ConversationResult,
    ConversationStage,
)

from .base import BasePattern, PatternValidationError


class MergeStrategy:
    """Strategies for merging parallel results."""

    @staticmethod
    def dict_merge(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Merge results as a dictionary keyed by stage name.

        Args:
            results: Dictionary of stage_name -> output.

        Returns:
            Merged dictionary with all results.
        """
        return dict(results)

    @staticmethod
    def list_merge(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Merge results into a list.

        Args:
            results: Dictionary of stage_name -> output.

        Returns:
            Dictionary with 'results' key containing list of outputs.
        """
        return {
            "results": [
                {"stage": name, "output": output} for name, output in results.items()
            ]
        }

    @staticmethod
    def flatten_merge(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Flatten all results into a single dictionary.

        Args:
            results: Dictionary of stage_name -> output.

        Returns:
            Flattened dictionary with all key-value pairs.
        """
        merged: dict[str, Any] = {}
        for output in results.values():
            if isinstance(output, dict):
                merged.update(output)
        return merged


class ParallelPattern(BasePattern):
    """Parallel execution pattern.

    Executes all stages simultaneously using asyncio.gather.
    Results are merged based on the merge strategy.

    Flow:
              ┌─→ [Agent A] ─┐
        [Task]─┼─→ [Agent B] ─┼─→ [Merge] → [Result]
              └─→ [Agent C] ─┘

    Use cases:
        - Multi-source information gathering
        - Parallel analysis tasks
        - Independent subtask execution
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the parallel pattern.

        Args:
            *args: Arguments passed to BasePattern.
            **kwargs: Keyword arguments passed to BasePattern.
        """
        super().__init__(*args, **kwargs)
        self.merge_strategy = MergeStrategy.dict_merge

    def set_merge_strategy(self, strategy: str | None = None) -> None:
        """Set the merge strategy.

        Args:
            strategy: One of 'dict', 'list', or 'flatten'. Default is 'dict'.
        """
        strategies = {
            "dict": MergeStrategy.dict_merge,
            "list": MergeStrategy.list_merge,
            "flatten": MergeStrategy.flatten_merge,
        }
        self.merge_strategy = strategies.get(
            strategy or "dict", MergeStrategy.dict_merge
        )

    @property
    def pattern_type(self) -> ConversationPattern:
        """Return the pattern type."""
        return ConversationPattern.PARALLEL

    def validate_conversation(self, conversation: Conversation) -> None:
        """Validate that the conversation is suitable for parallel execution.

        Args:
            conversation: The conversation to validate.

        Raises:
            PatternValidationError: If validation fails.
        """
        super().validate_conversation(conversation)

        if len(conversation.stages) < 2:
            raise PatternValidationError("Parallel pattern requires at least 2 stages")

    async def execute(self, conversation: Conversation) -> ConversationResult:
        """Execute stages in parallel and merge results.

        Args:
            conversation: The conversation to execute.

        Returns:
            ConversationResult with merged results.
        """
        self.validate_conversation(conversation)

        # Check merge strategy from conversation metadata
        merge_strategy = conversation.metadata.get("merge_strategy")
        if merge_strategy:
            self.set_merge_strategy(merge_strategy)

        # Prepare all stages for parallel execution
        tasks: list[asyncio.Task[tuple[str, dict[str, Any] | BaseException]]] = []

        for stage in conversation.stages:
            stage.input_data = conversation.initial_input
            await self._mark_stage_started(conversation.id, stage.name)

            task = asyncio.create_task(
                self._execute_stage_with_name(conversation, stage)
            )
            tasks.append(task)

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_results: dict[str, dict[str, Any]] = {}
        errors: list[str] = []

        for result in results:
            if isinstance(result, BaseException):
                errors.append(str(result))
            else:
                stage_name, output = result
                if isinstance(output, BaseException):
                    errors.append(f"{stage_name}: {output!s}")
                    await self._mark_stage_failed(
                        conversation.id, stage_name, str(output)
                    )
                else:
                    successful_results[stage_name] = output
                    await self._mark_stage_completed(
                        conversation.id, stage_name, output
                    )

        # Handle errors
        if errors:
            error_msg = "; ".join(errors)
            return await self.conversation_manager.fail(conversation.id, error_msg)

        # Merge results
        merged_output = self.merge_strategy(successful_results)
        return await self.conversation_manager.complete(conversation.id, merged_output)

    async def _execute_stage_with_name(
        self,
        conversation: Conversation,
        stage: ConversationStage,
    ) -> tuple[str, dict[str, Any] | BaseException]:
        """Execute a stage and return its name with the result.

        Args:
            conversation: The conversation.
            stage: The stage to execute.

        Returns:
            Tuple of (stage_name, output_or_exception).
        """
        try:
            output = await self._execute_single_stage(conversation, stage)
            return (stage.name, output)
        except Exception as e:
            return (stage.name, e)
