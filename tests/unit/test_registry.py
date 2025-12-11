"""Agent Registry unit tests."""

import pytest

from src.core.registry import (
    AgentAlreadyExistsError,
    AgentNotFoundError,
    AgentRegistry,
)
from src.models import AgentCapability, AgentConfig, AgentStatus


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, config: AgentConfig, active: bool = True):
        self.config = config
        self.is_active = active

    def can_handle(self, capability: str) -> bool:
        return capability in [cap.name for cap in self.config.capabilities]

    def get_capabilities(self) -> list[str]:
        return [cap.name for cap in self.config.capabilities]

    async def health_check(self) -> dict:
        return {
            "agent_id": self.config.agent_id,
            "status": "healthy" if self.is_active else "unhealthy",
        }

    async def process(self, message):
        pass


def create_mock_agent(
    agent_id: str,
    name: str = "Test Agent",
    capabilities: list[str] | None = None,
    active: bool = True,
) -> MockAgent:
    """Helper to create mock agents."""
    caps = [
        AgentCapability(name=cap, description=f"{cap} capability")
        for cap in (capabilities or ["test"])
    ]
    config = AgentConfig(
        agent_id=agent_id,
        name=name,
        capabilities=caps,
    )
    return MockAgent(config, active)


class TestAgentRegistry:
    """Test AgentRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return AgentRegistry()

    @pytest.mark.asyncio
    async def test_register_agent(self, registry):
        """Test registering an agent."""
        agent = create_mock_agent("agent_001", "Test Agent")

        info = await registry.register(agent)

        assert info.agent_id == "agent_001"
        assert info.name == "Test Agent"
        assert info.status == AgentStatus.ACTIVE
        assert len(registry) == 1

    @pytest.mark.asyncio
    async def test_register_duplicate_agent(self, registry):
        """Test registering duplicate agent raises error."""
        agent1 = create_mock_agent("agent_001")
        agent2 = create_mock_agent("agent_001")

        await registry.register(agent1)

        with pytest.raises(AgentAlreadyExistsError) as exc_info:
            await registry.register(agent2)

        assert exc_info.value.agent_id == "agent_001"

    @pytest.mark.asyncio
    async def test_unregister_agent(self, registry):
        """Test unregistering an agent."""
        agent = create_mock_agent("agent_001")
        await registry.register(agent)

        result = await registry.unregister("agent_001")

        assert result is True
        assert len(registry) == 0

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_agent(self, registry):
        """Test unregistering nonexistent agent returns False."""
        result = await registry.unregister("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_agent(self, registry):
        """Test getting an agent by ID."""
        agent = create_mock_agent("agent_001")
        await registry.register(agent)

        retrieved = await registry.get("agent_001")

        assert retrieved == agent

    @pytest.mark.asyncio
    async def test_get_nonexistent_agent(self, registry):
        """Test getting nonexistent agent raises error."""
        with pytest.raises(AgentNotFoundError) as exc_info:
            await registry.get("nonexistent")

        assert exc_info.value.agent_id == "nonexistent"

    @pytest.mark.asyncio
    async def test_get_info(self, registry):
        """Test getting agent info."""
        agent = create_mock_agent("agent_001", "Test Agent", ["web_search"])
        await registry.register(agent)

        info = await registry.get_info("agent_001")

        assert info.agent_id == "agent_001"
        assert info.status == AgentStatus.ACTIVE
        assert "web_search" in info.capabilities

    @pytest.mark.asyncio
    async def test_list_all(self, registry):
        """Test listing all agents."""
        await registry.register(create_mock_agent("agent_001"))
        await registry.register(create_mock_agent("agent_002"))
        await registry.register(create_mock_agent("agent_003"))

        all_agents = await registry.list_all()

        assert len(all_agents) == 3
        agent_ids = [a.agent_id for a in all_agents]
        assert "agent_001" in agent_ids
        assert "agent_002" in agent_ids
        assert "agent_003" in agent_ids

    @pytest.mark.asyncio
    async def test_find_by_capability(self, registry):
        """Test finding agents by capability."""
        await registry.register(
            create_mock_agent("researcher", capabilities=["web_search", "summarize"])
        )
        await registry.register(
            create_mock_agent("coder", capabilities=["code_gen", "code_review"])
        )
        await registry.register(
            create_mock_agent("reviewer", capabilities=["code_review"])
        )

        search_agents = await registry.find_by_capability("web_search")
        review_agents = await registry.find_by_capability("code_review")

        assert len(search_agents) == 1
        assert search_agents[0].config.agent_id == "researcher"
        assert len(review_agents) == 2

    @pytest.mark.asyncio
    async def test_find_by_capability_excludes_inactive(self, registry):
        """Test that inactive agents are excluded from capability search."""
        await registry.register(
            create_mock_agent("active", capabilities=["test"], active=True)
        )
        await registry.register(
            create_mock_agent("inactive", capabilities=["test"], active=False)
        )

        agents = await registry.find_by_capability("test")

        assert len(agents) == 1
        assert agents[0].config.agent_id == "active"

    @pytest.mark.asyncio
    async def test_find_one_by_capability(self, registry):
        """Test finding single agent by capability."""
        await registry.register(create_mock_agent("agent", capabilities=["unique_cap"]))

        agent = await registry.find_one_by_capability("unique_cap")
        no_agent = await registry.find_one_by_capability("nonexistent")

        assert agent is not None
        assert agent.config.agent_id == "agent"
        assert no_agent is None

    @pytest.mark.asyncio
    async def test_health_check_single(self, registry):
        """Test health check for single agent."""
        await registry.register(create_mock_agent("agent_001"))

        health = await registry.health_check("agent_001")

        assert health["agent_id"] == "agent_001"
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_all(self, registry):
        """Test health check for all agents."""
        await registry.register(create_mock_agent("agent_001", active=True))
        await registry.register(create_mock_agent("agent_002", active=False))

        health_results = await registry.health_check_all()

        assert len(health_results) == 2
        assert health_results["agent_001"]["status"] == "healthy"
        assert health_results["agent_002"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_contains(self, registry):
        """Test __contains__ method."""
        await registry.register(create_mock_agent("agent_001"))

        assert "agent_001" in registry
        assert "nonexistent" not in registry

    @pytest.mark.asyncio
    async def test_len(self, registry):
        """Test __len__ method."""
        assert len(registry) == 0

        await registry.register(create_mock_agent("agent_001"))
        assert len(registry) == 1

        await registry.register(create_mock_agent("agent_002"))
        assert len(registry) == 2
