"""Orchestrator - Central coordinator for agent collaboration.

This module coordinates task distribution, agent selection, and result merging.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.models import (
    Conversation,
    ConversationPattern,
    ConversationResult,
    ConversationStage,
    ConversationStatus,
    Message,
)

from .conversation import ConversationManager
from .message_bus import MessageBus
from .registry import AgentNotFoundError, AgentRegistry

if TYPE_CHECKING:
    from src.agents.base import BaseAgent


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""

    pass


class NoAgentAvailableError(OrchestratorError):
    """Raised when no agent is available for a capability."""

    def __init__(self, capability: str):
        self.capability = capability
        super().__init__(f"No agent available for capability: {capability}")


class TaskExecutionError(OrchestratorError):
    """Raised when task execution fails."""

    def __init__(self, message: str, stage: str | None = None):
        self.stage = stage
        super().__init__(message)


class ConversationTimeoutError(OrchestratorError):
    """Raised when a conversation times out."""

    def __init__(self, conversation_id: str, timeout: int):
        self.conversation_id = conversation_id
        self.timeout = timeout
        super().__init__(
            f"Conversation {conversation_id} timed out after {timeout} seconds"
        )


class Orchestrator:
    """Central coordinator for agent collaboration.

    Manages task distribution, agent selection, pattern execution,
    and result aggregation.
    """

    ORCHESTRATOR_ID = "orchestrator"

    def __init__(
        self,
        registry: AgentRegistry,
        message_bus: MessageBus,
        conversation_manager: ConversationManager,
    ):
        """Initialize the orchestrator.

        Args:
            registry: Agent registry for agent lookup.
            message_bus: Message bus for agent communication.
            conversation_manager: Manager for conversation state.
        """
        self.registry = registry
        self.message_bus = message_bus
        self.conversation_manager = conversation_manager
        self._running_tasks: dict[str, asyncio.Task] = {}

    async def execute(
        self,
        task: dict[str, Any],
        pattern: ConversationPattern = ConversationPattern.SEQUENTIAL,
        stages: list[ConversationStage] | None = None,
        timeout_seconds: int = 300,
    ) -> ConversationResult:
        """Execute a task using the specified pattern.

        Args:
            task: The task to execute (initial input).
            pattern: The conversation pattern to use.
            stages: Predefined stages (optional).
            timeout_seconds: Timeout for the entire conversation.

        Returns:
            ConversationResult with the execution results.
        """
        # Create conversation
        conversation = await self.conversation_manager.create(
            name=task.get("name", "Task Execution"),
            description=task.get("description", ""),
            pattern=pattern,
            stages=stages or [],
            initial_input=task,
            timeout_seconds=timeout_seconds,
        )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_conversation(conversation),
                timeout=timeout_seconds,
            )
            return result

        except TimeoutError as e:
            await self.conversation_manager.fail(
                conversation.id, f"Timeout after {timeout_seconds} seconds"
            )
            raise ConversationTimeoutError(conversation.id, timeout_seconds) from e

        except Exception as e:
            await self.conversation_manager.fail(conversation.id, str(e))
            raise

    async def execute_conversation(self, conversation_id: str) -> ConversationResult:
        """Execute an existing conversation.

        Args:
            conversation_id: The ID of the conversation to execute.

        Returns:
            ConversationResult with the execution results.
        """
        conversation = await self.conversation_manager.get(conversation_id)

        try:
            result = await asyncio.wait_for(
                self._execute_conversation(conversation),
                timeout=conversation.timeout_seconds,
            )
            return result

        except TimeoutError as e:
            await self.conversation_manager.fail(
                conversation.id,
                f"Timeout after {conversation.timeout_seconds} seconds",
            )
            raise ConversationTimeoutError(
                conversation.id, conversation.timeout_seconds
            ) from e

        except Exception as e:
            await self.conversation_manager.fail(conversation.id, str(e))
            raise

    async def _execute_conversation(
        self, conversation: Conversation
    ) -> ConversationResult:
        """Internal method to execute a conversation.

        Args:
            conversation: The conversation to execute.

        Returns:
            ConversationResult with the execution results.
        """
        await self.conversation_manager.start(conversation.id)

        # Execute based on pattern
        if conversation.pattern == ConversationPattern.SEQUENTIAL:
            result = await self._execute_sequential(conversation)
        elif conversation.pattern == ConversationPattern.PARALLEL:
            result = await self._execute_parallel(conversation)
        elif conversation.pattern == ConversationPattern.ROUTER:
            result = await self._execute_router(conversation)
        elif conversation.pattern == ConversationPattern.DEBATE:
            result = await self._execute_debate(conversation)
        else:
            raise OrchestratorError(f"Unsupported pattern: {conversation.pattern}")

        return result

    async def _execute_sequential(
        self, conversation: Conversation
    ) -> ConversationResult:
        """Execute stages sequentially.

        Each stage's output becomes the next stage's input.
        """
        current_input = conversation.initial_input
        final_output = {}

        for stage in conversation.stages:
            # Update stage status
            await self.conversation_manager.update_stage(
                conversation.id, stage.name, status=ConversationStatus.IN_PROGRESS
            )
            stage.input_data = current_input

            try:
                # Find agent for this stage
                agent = await self._select_agent(stage)

                # Create and send task message
                message = Message.create_task(
                    sender_id=self.ORCHESTRATOR_ID,
                    recipient_id=agent.config.agent_id,
                    content={
                        "task": current_input,
                        "stage": stage.name,
                        "capability": stage.agent_capability,
                    },
                    correlation_id=conversation.id,
                )

                # Record message
                await self.conversation_manager.add_message(conversation.id, message)

                # Process through agent
                response = await agent.process(message)

                # Record response
                await self.conversation_manager.add_message(conversation.id, response)

                # Extract output
                stage_output = response.content.get("result", response.content)

                # Update stage as completed
                await self.conversation_manager.update_stage(
                    conversation.id,
                    stage.name,
                    status=ConversationStatus.COMPLETED,
                    output=stage_output,
                )

                # Pass output to next stage
                current_input = stage_output
                final_output = stage_output

            except Exception as e:
                await self.conversation_manager.update_stage(
                    conversation.id,
                    stage.name,
                    status=ConversationStatus.FAILED,
                    error=str(e),
                )
                raise TaskExecutionError(str(e), stage=stage.name) from e

        return await self.conversation_manager.complete(conversation.id, final_output)

    async def _execute_parallel(self, conversation: Conversation) -> ConversationResult:
        """Execute stages in parallel and merge results."""
        tasks = []
        stage_names = []

        for stage in conversation.stages:
            stage.input_data = conversation.initial_input
            await self.conversation_manager.update_stage(
                conversation.id, stage.name, status=ConversationStatus.IN_PROGRESS
            )

            task = asyncio.create_task(self._execute_single_stage(conversation, stage))
            tasks.append(task)
            stage_names.append(stage.name)

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        merged_output = {}
        errors = []

        for stage_name, result in zip(stage_names, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(f"{stage_name}: {result!s}")
                await self.conversation_manager.update_stage(
                    conversation.id,
                    stage_name,
                    status=ConversationStatus.FAILED,
                    error=str(result),
                )
            else:
                merged_output[stage_name] = result
                await self.conversation_manager.update_stage(
                    conversation.id,
                    stage_name,
                    status=ConversationStatus.COMPLETED,
                    output=result,
                )

        if errors:
            error_msg = "; ".join(errors)
            return await self.conversation_manager.fail(conversation.id, error_msg)

        return await self.conversation_manager.complete(conversation.id, merged_output)

    async def _execute_single_stage(
        self, conversation: Conversation, stage: ConversationStage
    ) -> dict[str, Any]:
        """Execute a single stage.

        Args:
            conversation: The conversation.
            stage: The stage to execute.

        Returns:
            Stage output.
        """
        print(f"[DEBUG] Executing stage: {stage.name}")
        agent = await self._select_agent(stage)
        print(f"[DEBUG] Selected agent: {agent.config.agent_id}")

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
        print(f"[DEBUG] Calling agent.process()...")
        response = await agent.process(message)
        print(f"[DEBUG] Got response: {response.content}")
        await self.conversation_manager.add_message(conversation.id, response)

        result = response.content.get("result", response.content)
        if isinstance(result, dict):
            return result
        return {"result": result}

    async def _execute_router(self, conversation: Conversation) -> ConversationResult:
        """Execute using router pattern.

        Routes to appropriate agent based on task analysis.
        """
        # Analyze task to determine best route
        task = conversation.initial_input
        selected_stage = await self._analyze_and_route(conversation, task)

        if selected_stage is None:
            return await self.conversation_manager.fail(
                conversation.id, "No suitable route found for task"
            )

        # Execute only the selected stage
        await self.conversation_manager.update_stage(
            conversation.id, selected_stage.name, status=ConversationStatus.IN_PROGRESS
        )
        selected_stage.input_data = task

        try:
            output = await self._execute_single_stage(conversation, selected_stage)

            await self.conversation_manager.update_stage(
                conversation.id,
                selected_stage.name,
                status=ConversationStatus.COMPLETED,
                output=output,
            )

            return await self.conversation_manager.complete(conversation.id, output)

        except Exception as e:
            await self.conversation_manager.update_stage(
                conversation.id,
                selected_stage.name,
                status=ConversationStatus.FAILED,
                error=str(e),
            )
            return await self.conversation_manager.fail(conversation.id, str(e))

    async def _analyze_and_route(
        self, conversation: Conversation, task: dict[str, Any]
    ) -> ConversationStage | None:
        """Analyze task and select appropriate stage/route.

        Args:
            conversation: The conversation.
            task: The task to analyze.

        Returns:
            Selected stage, or None if no match.
        """
        # Simple routing based on task keywords or explicit routing
        if "route" in task:
            route_name = task["route"]
            return conversation.get_stage_by_name(route_name)

        # Default: return first stage with available agent
        for stage in conversation.stages:
            agent = await self.registry.find_one_by_capability(stage.agent_capability)
            if agent:
                return stage

        return None

    async def _execute_debate(self, conversation: Conversation) -> ConversationResult:
        """Execute debate pattern.

        Agents discuss and iterate until consensus or max iterations.
        Then a judge provides the final conclusion.
        """
        if len(conversation.stages) < 2:
            return await self.conversation_manager.fail(
                conversation.id, "Debate pattern requires at least 2 stages/agents"
            )

        current_proposal = conversation.initial_input
        iteration = 0
        all_debate_history: list[dict[str, Any]] = []

        while iteration < conversation.max_iterations:
            iteration += 1
            debate_round: dict[str, Any] = {"iteration": iteration, "responses": []}

            for stage in conversation.stages:
                stage.input_data = {
                    "proposal": current_proposal,
                    "iteration": iteration,
                    "previous_responses": debate_round["responses"],
                }

                try:
                    output = await self._execute_single_stage(conversation, stage)
                    debate_round["responses"].append(
                        {
                            "agent": stage.name,
                            "response": output.get("result", output),
                        }
                    )

                    # Check for consensus signal
                    if output.get("consensus", False):
                        final_output = {
                            "consensus": True,
                            "final_proposal": output.get("proposal", current_proposal),
                            "iterations": iteration,
                            "history": debate_round,
                        }
                        return await self.conversation_manager.complete(
                            conversation.id, final_output
                        )

                    # Update proposal for next round
                    if "proposal" in output:
                        current_proposal = output["proposal"]

                except Exception as e:
                    await self.conversation_manager.update_stage(
                        conversation.id,
                        stage.name,
                        status=ConversationStatus.FAILED,
                        error=str(e),
                    )

            all_debate_history.append(debate_round)

        # Get final conclusion from judge
        conclusion = await self._get_judge_conclusion(
            conversation, current_proposal, all_debate_history
        )

        final_output = {
            "consensus": False,
            "topic": current_proposal.get("topic", str(current_proposal)),
            "iterations": conversation.max_iterations,
            "debate_history": all_debate_history,
            "conclusion": conclusion,
        }
        return await self.conversation_manager.complete(conversation.id, final_output)

    async def _get_judge_conclusion(
        self,
        conversation: Conversation,
        topic: dict[str, Any],
        debate_history: list[dict[str, Any]],
    ) -> str:
        """Get final conclusion from judge agent.

        Args:
            conversation: The conversation.
            topic: The debate topic.
            debate_history: All debate rounds.

        Returns:
            Judge's conclusion.
        """
        # Try to find judge agent
        judge_agent = None
        try:
            judge_agent = await self.registry.get("judge")
        except AgentNotFoundError:
            # Try to find by capability
            judge_agent = await self.registry.find_one_by_capability("judge")

        if judge_agent is None:
            # No judge available, generate summary from debate
            return self._generate_debate_summary(debate_history)

        # Prepare debate summary for judge
        debate_summary = self._format_debate_for_judge(topic, debate_history)

        # Create judge stage
        judge_stage = ConversationStage(
            name="judge",
            agent_capability="judge",
            agent_id="judge",
        )
        judge_stage.input_data = {
            "task": f"다음 토론을 분석하고 최종 결론을 도출해주세요:\n\n{debate_summary}",
        }

        print("[DEBUG] Executing judge stage...")
        try:
            output = await self._execute_single_stage(conversation, judge_stage)
            return output.get("result", str(output))
        except Exception as e:
            print(f"[DEBUG] Judge failed: {e}")
            return self._generate_debate_summary(debate_history)

    def _format_debate_for_judge(
        self, topic: dict[str, Any], debate_history: list[dict[str, Any]]
    ) -> str:
        """Format debate history for judge analysis."""
        topic_str = topic.get("topic", str(topic)) if isinstance(topic, dict) else str(topic)
        lines = [f"## 토론 주제\n{topic_str}\n"]

        for round_data in debate_history:
            lines.append(f"\n### 라운드 {round_data['iteration']}")
            for resp in round_data["responses"]:
                agent_name = resp["agent"]
                response = resp["response"]
                if isinstance(response, dict):
                    response = response.get("result", str(response))
                lines.append(f"\n**{agent_name}**:\n{response}")

        return "\n".join(lines)

    def _generate_debate_summary(self, debate_history: list[dict[str, Any]]) -> str:
        """Generate a simple summary when no judge is available."""
        if not debate_history:
            return "토론 내용이 없습니다."

        last_round = debate_history[-1]
        summary_parts = ["## 최종 라운드 요약\n"]

        for resp in last_round.get("responses", []):
            agent_name = resp.get("agent", "Unknown")
            response = resp.get("response", "")
            if isinstance(response, dict):
                response = response.get("result", str(response))
            # Truncate if too long
            if len(str(response)) > 500:
                response = str(response)[:500] + "..."
            summary_parts.append(f"**{agent_name}**: {response}")

        return "\n\n".join(summary_parts)

    async def _select_agent(self, stage: ConversationStage) -> BaseAgent:
        """Select an agent for a stage.

        Args:
            stage: The stage requiring an agent.

        Returns:
            Selected agent.

        Raises:
            NoAgentAvailableError: If no agent is available.
        """
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

    async def cancel(self, conversation_id: str) -> ConversationResult:
        """Cancel a running conversation.

        Args:
            conversation_id: The ID of the conversation to cancel.

        Returns:
            ConversationResult with cancelled status.
        """
        # Cancel any running task
        if conversation_id in self._running_tasks:
            self._running_tasks[conversation_id].cancel()
            del self._running_tasks[conversation_id]

        return await self.conversation_manager.cancel(conversation_id)

    async def get_status(self, conversation_id: str) -> dict[str, Any]:
        """Get the status of a conversation.

        Args:
            conversation_id: The ID of the conversation.

        Returns:
            Status dictionary.
        """
        conversation = await self.conversation_manager.get(conversation_id)

        current_stage = conversation.get_current_stage()
        return {
            "conversation_id": conversation.id,
            "status": conversation.status.value,
            "pattern": conversation.pattern.value,
            "current_stage": current_stage.name if current_stage else None,
            "completed_stages": len(conversation.get_completed_stages()),
            "total_stages": len(conversation.stages),
            "messages_count": len(conversation.messages),
        }
