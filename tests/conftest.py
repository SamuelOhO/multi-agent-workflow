"""테스트 공통 설정 및 fixtures."""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from src.agents.base import SimpleAgent
from src.core.conversation import ConversationManager
from src.core.message_bus import InMemoryMessageBus
from src.core.orchestrator import Orchestrator
from src.core.registry import AgentRegistry
from src.models import AgentCapability, AgentConfig


@pytest.fixture
def agent_config() -> AgentConfig:
    """기본 Agent 설정 fixture."""
    return AgentConfig(
        agent_id="test_agent_001",
        name="Test Agent",
        description="테스트용 Agent",
        capabilities=[
            AgentCapability(
                name="test_capability",
                description="테스트 기능",
            )
        ],
        model="claude-haiku-4-5-20251001",
    )


@pytest.fixture
def simple_agent(agent_config: AgentConfig) -> SimpleAgent:
    """SimpleAgent fixture."""
    return SimpleAgent(agent_config)


@pytest_asyncio.fixture
async def registry() -> AsyncGenerator[AgentRegistry, None]:
    """AgentRegistry fixture."""
    reg = AgentRegistry()
    yield reg


@pytest_asyncio.fixture
async def message_bus() -> AsyncGenerator[InMemoryMessageBus, None]:
    """MessageBus fixture."""
    bus = InMemoryMessageBus()
    yield bus


@pytest_asyncio.fixture
async def conversation_manager() -> AsyncGenerator[ConversationManager, None]:
    """ConversationManager fixture."""
    manager = ConversationManager()
    yield manager


@pytest_asyncio.fixture
async def orchestrator(
    registry: AgentRegistry,
    message_bus: InMemoryMessageBus,
    conversation_manager: ConversationManager,
) -> AsyncGenerator[Orchestrator, None]:
    """Orchestrator fixture."""
    orch = Orchestrator(
        registry=registry,
        message_bus=message_bus,
        conversation_manager=conversation_manager,
    )
    yield orch


@pytest_asyncio.fixture
async def registered_agent(
    registry: AgentRegistry,
    simple_agent: SimpleAgent,
) -> AsyncGenerator[SimpleAgent, None]:
    """등록된 Agent fixture."""
    await registry.register(simple_agent)
    yield simple_agent
