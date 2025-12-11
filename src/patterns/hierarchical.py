"""Hierarchical pattern implementation.

This module implements the hierarchical conversation pattern where
a supervisor agent manages and coordinates worker agents.
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

from .base import BasePattern, PatternValidationError, StageExecutionError


class HierarchicalPattern(BasePattern):
    """Hierarchical execution pattern.

    A supervisor agent manages worker agents, delegating tasks
    and aggregating results.

    Flow:
                  [Supervisor]
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    [Worker A]    [Worker B]    [Worker C]

    The supervisor:
    1. Analyzes the initial task
    2. Delegates subtasks to workers
    3. Aggregates worker results
    4. Produces final output

    Use cases:
        - Large-scale task decomposition
        - Complex project management
        - Multi-step workflows with oversight
    """

    @property
    def pattern_type(self) -> ConversationPattern:
        """Return the pattern type."""
        return ConversationPattern.HIERARCHICAL

    def validate_conversation(self, conversation: Conversation) -> None:
        """Validate that the conversation is suitable for hierarchical execution.

        The first stage is treated as the supervisor, remaining stages as workers.

        Args:
            conversation: The conversation to validate.

        Raises:
            PatternValidationError: If validation fails.
        """
        super().validate_conversation(conversation)

        if len(conversation.stages) < 2:
            raise PatternValidationError(
                "Hierarchical pattern requires at least 2 stages "
                "(1 supervisor + 1 or more workers)"
            )

    async def execute(self, conversation: Conversation) -> ConversationResult:
        """Execute using hierarchical pattern.

        The first stage acts as supervisor, delegating to worker stages.

        Args:
            conversation: The conversation to execute.

        Returns:
            ConversationResult with the execution results.
        """
        self.validate_conversation(conversation)

        supervisor_stage = conversation.stages[0]
        worker_stages = conversation.stages[1:]

        # Phase 1: Supervisor analyzes task and creates delegation plan
        supervisor_stage.input_data = {
            "task": conversation.initial_input,
            "available_workers": [
                {
                    "name": w.name,
                    "capability": w.agent_capability,
                    "description": w.description,
                }
                for w in worker_stages
            ],
            "phase": "planning",
        }

        await self._mark_stage_started(conversation.id, supervisor_stage.name)

        try:
            plan_output = await self._execute_single_stage(
                conversation, supervisor_stage
            )
        except StageExecutionError as e:
            await self._mark_stage_failed(
                conversation.id, supervisor_stage.name, str(e)
            )
            return await self.conversation_manager.fail(conversation.id, str(e))

        # Extract delegation plan from supervisor
        delegations = self._extract_delegations(plan_output, worker_stages)

        if not delegations:
            # Supervisor decided no delegation needed, use its output directly
            await self._mark_stage_completed(
                conversation.id, supervisor_stage.name, plan_output
            )
            return await self.conversation_manager.complete(
                conversation.id, plan_output
            )

        # Phase 2: Execute delegated tasks (can be parallel or sequential)
        execution_mode = plan_output.get("execution_mode", "parallel")
        worker_results: dict[str, dict[str, Any]] = {}

        if execution_mode == "sequential":
            worker_results = await self._execute_workers_sequential(
                conversation, delegations
            )
        else:
            worker_results = await self._execute_workers_parallel(
                conversation, delegations
            )

        # Phase 3: Supervisor aggregates results
        supervisor_stage.input_data = {
            "task": conversation.initial_input,
            "phase": "aggregation",
            "worker_results": worker_results,
            "original_plan": plan_output,
        }

        try:
            final_output = await self._execute_single_stage(
                conversation, supervisor_stage
            )
            await self._mark_stage_completed(
                conversation.id, supervisor_stage.name, final_output
            )
        except StageExecutionError as e:
            await self._mark_stage_failed(
                conversation.id, supervisor_stage.name, str(e)
            )
            # Return worker results if supervisor aggregation fails
            final_output = {
                "supervisor_aggregation_failed": str(e),
                "worker_results": worker_results,
            }

        return await self.conversation_manager.complete(conversation.id, final_output)

    def _extract_delegations(
        self,
        plan_output: dict[str, Any],
        worker_stages: list[ConversationStage],
    ) -> dict[str, dict[str, Any]]:
        """Extract delegation assignments from supervisor's plan.

        Args:
            plan_output: The supervisor's planning output.
            worker_stages: Available worker stages.

        Returns:
            Dictionary mapping worker stage names to their assigned tasks.
        """
        delegations: dict[str, dict[str, Any]] = {}

        # Check for explicit delegations in output
        if "delegations" in plan_output:
            for delegation in plan_output["delegations"]:
                worker_name = delegation.get("worker")
                task = delegation.get("task")
                if worker_name and task:
                    # Verify worker exists
                    for stage in worker_stages:
                        if (
                            stage.name == worker_name
                            or stage.agent_capability == worker_name
                        ):
                            delegations[stage.name] = {"task": task}
                            break

        # If no explicit delegations, delegate to all workers with initial task
        if not delegations and plan_output.get("delegate_to_all", False):
            subtasks = plan_output.get("subtasks", {})
            for stage in worker_stages:
                task_for_worker = subtasks.get(
                    stage.name,
                    subtasks.get(
                        stage.agent_capability, plan_output.get("task_for_workers")
                    ),
                )
                if task_for_worker:
                    delegations[stage.name] = {"task": task_for_worker}

        return delegations

    async def _execute_workers_parallel(
        self,
        conversation: Conversation,
        delegations: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Execute worker tasks in parallel.

        Args:
            conversation: The conversation.
            delegations: Worker name to task mapping.

        Returns:
            Dictionary of worker results.
        """
        tasks: list[asyncio.Task[tuple[str, dict[str, Any] | BaseException]]] = []

        for stage in conversation.stages[1:]:  # Skip supervisor
            if stage.name not in delegations:
                continue

            stage.input_data = delegations[stage.name]
            await self._mark_stage_started(conversation.id, stage.name)

            task = asyncio.create_task(
                self._execute_worker_with_name(conversation, stage)
            )
            tasks.append(task)

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        worker_results: dict[str, dict[str, Any]] = {}
        for result in results:
            if isinstance(result, BaseException):
                continue
            stage_name, output = result
            if isinstance(output, BaseException):
                await self._mark_stage_failed(conversation.id, stage_name, str(output))
                worker_results[stage_name] = {"error": str(output)}
            else:
                await self._mark_stage_completed(conversation.id, stage_name, output)
                worker_results[stage_name] = output

        return worker_results

    async def _execute_workers_sequential(
        self,
        conversation: Conversation,
        delegations: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Execute worker tasks sequentially.

        Args:
            conversation: The conversation.
            delegations: Worker name to task mapping.

        Returns:
            Dictionary of worker results.
        """
        worker_results: dict[str, dict[str, Any]] = {}
        previous_output: dict[str, Any] = {}

        for stage in conversation.stages[1:]:  # Skip supervisor
            if stage.name not in delegations:
                continue

            # Include previous worker's output for chaining
            stage.input_data = {
                **delegations[stage.name],
                "previous_worker_output": previous_output,
            }
            await self._mark_stage_started(conversation.id, stage.name)

            try:
                output = await self._execute_single_stage(conversation, stage)
                await self._mark_stage_completed(conversation.id, stage.name, output)
                worker_results[stage.name] = output
                previous_output = output
            except StageExecutionError as e:
                await self._mark_stage_failed(conversation.id, stage.name, str(e))
                worker_results[stage.name] = {"error": str(e)}

        return worker_results

    async def _execute_worker_with_name(
        self,
        conversation: Conversation,
        stage: ConversationStage,
    ) -> tuple[str, dict[str, Any] | BaseException]:
        """Execute a worker stage and return its name with the result.

        Args:
            conversation: The conversation.
            stage: The worker stage to execute.

        Returns:
            Tuple of (stage_name, output_or_exception).
        """
        try:
            output = await self._execute_single_stage(conversation, stage)
            return (stage.name, output)
        except Exception as e:
            return (stage.name, e)
