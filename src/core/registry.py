"""Agent Registry - Agent registration and discovery.

This module manages agent lifecycle and provides capability-based agent lookup.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.models import AgentConfig, AgentInfo, AgentStatus, Message

if TYPE_CHECKING:
    from src.agents.base import BaseAgent


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol defining the required interface for agents."""

    @property
    def config(self) -> AgentConfig:
        """Agent configuration."""
        ...

    @property
    def is_active(self) -> bool:
        """Whether the agent is active."""
        ...

    async def process(self, message: Message) -> Message:
        """Process a message and return response."""
        ...

    def can_handle(self, capability: str) -> bool:
        """Check if agent can handle a capability."""
        ...

    def get_capabilities(self) -> list[str]:
        """Get list of capabilities."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        ...


class AgentNotFoundError(Exception):
    """Raised when an agent is not found in the registry."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__(f"Agent not found: {agent_id}")


class AgentAlreadyExistsError(Exception):
    """Raised when trying to register an agent that already exists."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__(f"Agent already exists: {agent_id}")


class AgentRegistry:
    """Registry for managing agents.

    Provides registration, discovery, and health monitoring of agents.
    Thread-safe for concurrent access.
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._lock = asyncio.Lock()

    async def register(self, agent: BaseAgent) -> AgentInfo:
        """Register an agent.

        Args:
            agent: The agent to register.

        Returns:
            AgentInfo with the registered agent's information.

        Raises:
            AgentAlreadyExistsError: If an agent with the same ID already exists.
        """
        async with self._lock:
            agent_id = agent.config.agent_id

            if agent_id in self._agents:
                raise AgentAlreadyExistsError(agent_id)

            self._agents[agent_id] = agent

            return AgentInfo.from_config(agent.config, status=AgentStatus.ACTIVE)

    async def unregister(self, agent_id: str) -> bool:
        """Unregister an agent.

        Args:
            agent_id: The ID of the agent to unregister.

        Returns:
            True if the agent was unregistered, False if not found.
        """
        async with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                return True
            return False

    async def get(self, agent_id: str) -> BaseAgent:
        """Get an agent by ID.

        Args:
            agent_id: The ID of the agent to retrieve.

        Returns:
            The agent instance.

        Raises:
            AgentNotFoundError: If the agent is not found.
        """
        async with self._lock:
            if agent_id not in self._agents:
                raise AgentNotFoundError(agent_id)
            return self._agents[agent_id]

    async def get_info(self, agent_id: str) -> AgentInfo:
        """Get agent info by ID.

        Args:
            agent_id: The ID of the agent.

        Returns:
            AgentInfo for the agent.

        Raises:
            AgentNotFoundError: If the agent is not found.
        """
        agent = await self.get(agent_id)
        status = AgentStatus.ACTIVE if agent.is_active else AgentStatus.INACTIVE
        return AgentInfo.from_config(agent.config, status=status)

    async def list_all(self) -> list[AgentInfo]:
        """List all registered agents.

        Returns:
            List of AgentInfo for all registered agents.
        """
        async with self._lock:
            result = []
            for agent in self._agents.values():
                status = AgentStatus.ACTIVE if agent.is_active else AgentStatus.INACTIVE
                result.append(AgentInfo.from_config(agent.config, status=status))
            return result

    async def find_by_capability(self, capability: str) -> list[BaseAgent]:
        """Find agents that can handle a specific capability.

        Args:
            capability: The capability to search for.

        Returns:
            List of agents that can handle the capability.
        """
        async with self._lock:
            return [
                agent
                for agent in self._agents.values()
                if agent.can_handle(capability) and agent.is_active
            ]

    async def find_one_by_capability(self, capability: str) -> BaseAgent | None:
        """Find a single agent that can handle a capability.

        Args:
            capability: The capability to search for.

        Returns:
            An agent that can handle the capability, or None if not found.
        """
        agents = await self.find_by_capability(capability)
        return agents[0] if agents else None

    async def health_check(self, agent_id: str) -> dict[str, Any]:
        """Perform health check on a specific agent.

        Args:
            agent_id: The ID of the agent to check.

        Returns:
            Health check result dictionary.

        Raises:
            AgentNotFoundError: If the agent is not found.
        """
        agent = await self.get(agent_id)
        result: dict[str, Any] = await agent.health_check()
        return result

    async def health_check_all(self) -> dict[str, dict[str, Any]]:
        """Perform health check on all registered agents.

        Returns:
            Dictionary mapping agent IDs to their health check results.
        """
        async with self._lock:
            results: dict[str, dict[str, Any]] = {}
            for agent_id, agent in self._agents.items():
                try:
                    results[agent_id] = await agent.health_check()
                except Exception as e:
                    results[agent_id] = {
                        "agent_id": agent_id,
                        "status": "error",
                        "error": str(e),
                    }
            return results

    def __len__(self) -> int:
        """Return the number of registered agents."""
        return len(self._agents)

    def __contains__(self, agent_id: str) -> bool:
        """Check if an agent is registered."""
        return agent_id in self._agents
