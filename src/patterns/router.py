"""Router pattern implementation.

This module implements the router conversation pattern where
tasks are routed to the most appropriate agent based on conditions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.models import (
    Conversation,
    ConversationPattern,
    ConversationResult,
    ConversationStage,
)

from .base import BasePattern, PatternValidationError

# Type alias for routing condition functions
RoutingCondition = Callable[[dict[str, Any], ConversationStage], bool]


class RouterPattern(BasePattern):
    """Router execution pattern.

      Routes the task to the most appropriate agent based on
      task analysis or explicit routing conditions.

      Flow:
                [Router]
                   │
       ┌───────┼───────┐
       ▼       ▼       ▼
    [Agent A] [Agent B] [Agent C]
    (cond A)  (cond B)  (cond C)

      Use cases:
          - Question type classification
          - Domain-specific routing
          - Conditional execution
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the router pattern.

        Args:
            *args: Arguments passed to BasePattern.
            **kwargs: Keyword arguments passed to BasePattern.
        """
        super().__init__(*args, **kwargs)
        self._routing_conditions: dict[str, RoutingCondition] = {}
        self._default_route: str | None = None

    @property
    def pattern_type(self) -> ConversationPattern:
        """Return the pattern type."""
        return ConversationPattern.ROUTER

    def add_routing_condition(
        self,
        stage_name: str,
        condition: RoutingCondition,
    ) -> None:
        """Add a routing condition for a stage.

        Args:
            stage_name: The name of the stage this condition routes to.
            condition: A callable that takes (task, stage) and returns bool.
        """
        self._routing_conditions[stage_name] = condition

    def set_default_route(self, stage_name: str) -> None:
        """Set the default route when no conditions match.

        Args:
            stage_name: The name of the default stage.
        """
        self._default_route = stage_name

    def validate_conversation(self, conversation: Conversation) -> None:
        """Validate that the conversation is suitable for routing.

        Args:
            conversation: The conversation to validate.

        Raises:
            PatternValidationError: If validation fails.
        """
        super().validate_conversation(conversation)

        if len(conversation.stages) < 1:
            raise PatternValidationError("Router pattern requires at least 1 stage")

    async def execute(self, conversation: Conversation) -> ConversationResult:
        """Execute using router pattern.

        Routes to appropriate agent based on task analysis.

        Args:
            conversation: The conversation to execute.

        Returns:
            ConversationResult with the execution results.
        """
        self.validate_conversation(conversation)

        task = conversation.initial_input

        # Select the appropriate route
        selected_stage = await self._select_route(conversation, task)

        if selected_stage is None:
            return await self.conversation_manager.fail(
                conversation.id, "No suitable route found for task"
            )

        # Mark other stages as skipped (remain in PENDING status)
        # Only execute the selected stage
        selected_stage.input_data = task
        await self._mark_stage_started(conversation.id, selected_stage.name)

        try:
            output = await self._execute_single_stage(conversation, selected_stage)

            await self._mark_stage_completed(
                conversation.id, selected_stage.name, output
            )

            return await self.conversation_manager.complete(conversation.id, output)

        except Exception as e:
            error_msg = str(e)
            await self._mark_stage_failed(
                conversation.id, selected_stage.name, error_msg
            )
            return await self.conversation_manager.fail(conversation.id, error_msg)

    async def _select_route(
        self,
        conversation: Conversation,
        task: dict[str, Any],
    ) -> ConversationStage | None:
        """Select the appropriate stage/route for the task.

        Routing priority:
        1. Explicit 'route' field in task
        2. Custom routing conditions
        3. Keyword-based routing (based on stage metadata)
        4. Default route
        5. First stage with available agent

        Args:
            conversation: The conversation.
            task: The task to analyze.

        Returns:
            Selected stage, or None if no match.
        """
        # 1. Check for explicit route in task
        if "route" in task:
            route_name = task["route"]
            stage = conversation.get_stage_by_name(route_name)
            if stage:
                return stage

        # 2. Check custom routing conditions
        for stage in conversation.stages:
            condition = self._routing_conditions.get(stage.name)
            if condition and condition(task, stage):
                return stage

        # 3. Check keyword-based routing from stage metadata
        task_text = self._extract_task_text(task)
        for stage in conversation.stages:
            keywords = self._get_stage_keywords(stage)
            if keywords and self._matches_keywords(task_text, keywords):
                return stage

        # 4. Check default route
        if self._default_route:
            stage = conversation.get_stage_by_name(self._default_route)
            if stage:
                return stage

        # 5. Return first stage with available agent
        for stage in conversation.stages:
            agent = await self.registry.find_one_by_capability(stage.agent_capability)
            if agent:
                return stage

        return None

    def _extract_task_text(self, task: dict[str, Any]) -> str:
        """Extract text content from task for keyword matching.

        Args:
            task: The task dictionary.

        Returns:
            Combined text from task fields.
        """
        text_parts = []

        for key in ["query", "task", "text", "content", "message", "request"]:
            if key in task and isinstance(task[key], str):
                text_parts.append(task[key])

        if "name" in task and isinstance(task["name"], str):
            text_parts.append(task["name"])

        if "description" in task and isinstance(task["description"], str):
            text_parts.append(task["description"])

        return " ".join(text_parts).lower()

    def _get_stage_keywords(self, stage: ConversationStage) -> list[str]:
        """Get routing keywords for a stage.

        Keywords can be defined in:
        - stage.input_data['routing_keywords']
        - Or derived from capability name

        Args:
            stage: The stage to get keywords for.

        Returns:
            List of keywords.
        """
        # Check for explicit keywords in stage input data
        keywords = stage.input_data.get("routing_keywords", [])
        if keywords:
            return [k.lower() for k in keywords]

        # Derive from capability name
        capability_keywords = stage.agent_capability.replace("_", " ").split()
        return [k.lower() for k in capability_keywords]

    def _matches_keywords(self, text: str, keywords: list[str]) -> bool:
        """Check if text contains any of the keywords.

        Args:
            text: The text to search in.
            keywords: Keywords to match.

        Returns:
            True if any keyword is found in text.
        """
        return any(keyword in text for keyword in keywords)
