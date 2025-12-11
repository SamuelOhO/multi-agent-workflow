"""Debate pattern implementation.

This module implements the debate/consensus conversation pattern where
agents discuss and iterate until they reach consensus or max iterations.
"""

from __future__ import annotations

from typing import Any

from src.models import (
    Conversation,
    ConversationPattern,
    ConversationResult,
    ConversationStage,
)

from .base import BasePattern, PatternValidationError


class DebateRound:
    """Represents a single round of debate."""

    def __init__(self, iteration: int):
        """Initialize a debate round.

        Args:
            iteration: The iteration number (1-indexed).
        """
        self.iteration = iteration
        self.responses: list[dict[str, Any]] = []
        self.consensus_reached = False
        self.final_proposal: dict[str, Any] | None = None

    def add_response(
        self,
        agent_capability: str,
        response: dict[str, Any],
    ) -> None:
        """Add a response to this round.

        Args:
            agent_capability: The capability of the responding agent.
            response: The agent's response.
        """
        self.responses.append(
            {
                "agent": agent_capability,
                "response": response,
            }
        )

        # Check for consensus signal
        if response.get("consensus", False):
            self.consensus_reached = True
            self.final_proposal = response.get("proposal")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the round.
        """
        return {
            "iteration": self.iteration,
            "responses": self.responses,
            "consensus_reached": self.consensus_reached,
        }


class DebatePattern(BasePattern):
    """Debate/consensus execution pattern.

    Agents discuss and iterate on proposals until consensus is
    reached or max iterations are exhausted.

    Flow:
        [Task] → [Agent A proposes] → [Agent B critiques]
                        ↓
           [Agent A responds] → [Agent B responds]
                        ↓
                  [Consensus] → [Result]

    Use cases:
        - Decision making
        - Code review discussions
        - Document review and approval
        - Multi-perspective analysis
    """

    @property
    def pattern_type(self) -> ConversationPattern:
        """Return the pattern type."""
        return ConversationPattern.DEBATE

    def validate_conversation(self, conversation: Conversation) -> None:
        """Validate that the conversation is suitable for debate.

        Args:
            conversation: The conversation to validate.

        Raises:
            PatternValidationError: If validation fails.
        """
        super().validate_conversation(conversation)

        if len(conversation.stages) < 2:
            raise PatternValidationError(
                "Debate pattern requires at least 2 stages/agents"
            )

        if conversation.max_iterations < 1:
            raise PatternValidationError("Debate pattern requires at least 1 iteration")

    async def execute(self, conversation: Conversation) -> ConversationResult:
        """Execute debate pattern.

        Agents discuss and iterate until consensus or max iterations.

        Args:
            conversation: The conversation to execute.

        Returns:
            ConversationResult with consensus or final state.
        """
        self.validate_conversation(conversation)

        current_proposal = conversation.initial_input
        debate_history: list[dict[str, Any]] = []
        iteration = 0

        while iteration < conversation.max_iterations:
            iteration += 1
            debate_round = DebateRound(iteration)

            for stage in conversation.stages:
                # Prepare input for this stage
                stage.input_data = {
                    "proposal": current_proposal,
                    "iteration": iteration,
                    "previous_responses": debate_round.responses.copy(),
                    "debate_history": debate_history,
                }

                try:
                    output = await self._execute_single_stage(conversation, stage)

                    debate_round.add_response(stage.agent_capability, output)

                    # Check for consensus
                    if debate_round.consensus_reached:
                        final_output = self._create_consensus_output(
                            consensus=True,
                            proposal=debate_round.final_proposal or current_proposal,
                            iterations=iteration,
                            history=debate_history + [debate_round.to_dict()],
                        )
                        return await self.conversation_manager.complete(
                            conversation.id, final_output
                        )

                    # Update proposal if agent provided one
                    if "proposal" in output:
                        current_proposal = output["proposal"]

                except Exception as e:
                    await self._mark_stage_failed(conversation.id, stage.name, str(e))
                    # Continue debate with other agents even if one fails

            debate_history.append(debate_round.to_dict())

        # Max iterations reached without consensus
        final_output = self._create_consensus_output(
            consensus=False,
            proposal=current_proposal,
            iterations=conversation.max_iterations,
            history=debate_history,
            message="Max iterations reached without consensus",
        )
        return await self.conversation_manager.complete(conversation.id, final_output)

    def _create_consensus_output(
        self,
        consensus: bool,
        proposal: dict[str, Any] | Any,
        iterations: int,
        history: list[dict[str, Any]],
        message: str | None = None,
    ) -> dict[str, Any]:
        """Create the final output for a debate.

        Args:
            consensus: Whether consensus was reached.
            proposal: The final proposal.
            iterations: Number of iterations completed.
            history: Full debate history.
            message: Optional message (e.g., for timeout).

        Returns:
            Final output dictionary.
        """
        output: dict[str, Any] = {
            "consensus": consensus,
            "final_proposal": proposal,
            "iterations": iterations,
            "history": history,
        }
        if message:
            output["message"] = message
        return output

    async def execute_with_moderator(
        self,
        conversation: Conversation,
        moderator_stage: ConversationStage,
    ) -> ConversationResult:
        """Execute debate with a moderator agent.

        The moderator decides when consensus is reached and can
        guide the discussion.

        Args:
            conversation: The conversation to execute.
            moderator_stage: Stage representing the moderator agent.

        Returns:
            ConversationResult with moderated consensus.
        """
        self.validate_conversation(conversation)

        current_proposal = conversation.initial_input
        debate_history: list[dict[str, Any]] = []
        iteration = 0

        while iteration < conversation.max_iterations:
            iteration += 1
            debate_round = DebateRound(iteration)

            # Regular agents discuss
            for stage in conversation.stages:
                stage.input_data = {
                    "proposal": current_proposal,
                    "iteration": iteration,
                    "previous_responses": debate_round.responses.copy(),
                }

                try:
                    output = await self._execute_single_stage(conversation, stage)
                    debate_round.add_response(stage.agent_capability, output)

                    if "proposal" in output:
                        current_proposal = output["proposal"]

                except Exception:
                    continue  # Continue with other agents

            debate_history.append(debate_round.to_dict())

            # Moderator evaluates the round
            moderator_stage.input_data = {
                "proposal": current_proposal,
                "iteration": iteration,
                "round_responses": debate_round.responses,
                "debate_history": debate_history,
            }

            try:
                moderator_output = await self._execute_single_stage(
                    conversation, moderator_stage
                )

                # Moderator decides if consensus is reached
                if moderator_output.get("consensus", False):
                    final_output = self._create_consensus_output(
                        consensus=True,
                        proposal=moderator_output.get("proposal", current_proposal),
                        iterations=iteration,
                        history=debate_history,
                    )
                    return await self.conversation_manager.complete(
                        conversation.id, final_output
                    )

                # Moderator can suggest direction for next round
                if "guidance" in moderator_output:
                    current_proposal["moderator_guidance"] = moderator_output[
                        "guidance"
                    ]

            except Exception:
                pass  # Continue without moderator input

        # Max iterations
        final_output = self._create_consensus_output(
            consensus=False,
            proposal=current_proposal,
            iterations=conversation.max_iterations,
            history=debate_history,
            message="Max iterations reached without consensus",
        )
        return await self.conversation_manager.complete(conversation.id, final_output)
