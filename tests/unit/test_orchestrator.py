"""Orchestrator unit tests."""

import pytest

from src.core import (
    AgentRegistry,
    ConversationManager,
    InMemoryMessageBus,
    Orchestrator,
)
from src.models import (
    AgentCapability,
    AgentConfig,
    ConversationPattern,
    ConversationStage,
    ConversationStatus,
    Message,
)


class MockAgent:
    """Mock agent for testing orchestrator."""

    def __init__(self, config: AgentConfig, response_content: dict | None = None):
        self.config = config
        self.is_active = True
        self.response_content = response_content or {"result": "success"}
        self.received_messages: list[Message] = []

    def can_handle(self, capability: str) -> bool:
        return capability in [cap.name for cap in self.config.capabilities]

    def get_capabilities(self) -> list[str]:
        return [cap.name for cap in self.config.capabilities]

    async def health_check(self) -> dict:
        return {"agent_id": self.config.agent_id, "status": "healthy"}

    async def process(self, message: Message) -> Message:
        self.received_messages.append(message)
        return Message.create_response(
            sender_id=self.config.agent_id,
            recipient_id=message.sender_id,
            content=self.response_content,
            correlation_id=message.correlation_id,
            parent_message_id=message.id,
        )


def create_mock_agent(
    agent_id: str,
    capabilities: list[str],
    response: dict | None = None,
) -> MockAgent:
    """Helper to create mock agents."""
    caps = [
        AgentCapability(name=cap, description=f"{cap} capability")
        for cap in capabilities
    ]
    config = AgentConfig(agent_id=agent_id, name=f"Mock {agent_id}", capabilities=caps)
    return MockAgent(config, response)


class TestOrchestrator:
    """Test Orchestrator class."""

    @pytest.fixture
    def registry(self):
        return AgentRegistry()

    @pytest.fixture
    def message_bus(self):
        return InMemoryMessageBus()

    @pytest.fixture
    def conversation_manager(self):
        return ConversationManager()

    @pytest.fixture
    def orchestrator(self, registry, message_bus, conversation_manager):
        return Orchestrator(registry, message_bus, conversation_manager)

    @pytest.fixture
    async def setup_agents(self, registry):
        """Setup common test agents."""
        researcher = create_mock_agent(
            "researcher",
            ["web_search", "summarize"],
            {"result": {"research": "AI trends data"}},
        )
        coder = create_mock_agent(
            "coder",
            ["code_gen", "code_review"],
            {"result": {"code": "print('hello')"}},
        )
        reviewer = create_mock_agent(
            "reviewer",
            ["code_review"],
            {"result": {"review": "LGTM"}},
        )

        await registry.register(researcher)
        await registry.register(coder)
        await registry.register(reviewer)

        return {"researcher": researcher, "coder": coder, "reviewer": reviewer}

    @pytest.mark.asyncio
    async def test_execute_sequential(self, orchestrator, setup_agents):
        """Test sequential pattern execution."""
        stages = [
            ConversationStage(
                name="research",
                description="Research",
                agent_capability="web_search",
            ),
            ConversationStage(
                name="code",
                description="Code",
                agent_capability="code_gen",
            ),
        ]

        result = await orchestrator.execute(
            task={"request": "build a web scraper"},
            pattern=ConversationPattern.SEQUENTIAL,
            stages=stages,
        )

        assert result.status == ConversationStatus.COMPLETED
        assert result.stages_completed == 2

    @pytest.mark.asyncio
    async def test_execute_parallel(self, orchestrator, setup_agents):
        """Test parallel pattern execution."""
        stages = [
            ConversationStage(
                name="research",
                description="Research",
                agent_capability="web_search",
            ),
            ConversationStage(
                name="code",
                description="Code",
                agent_capability="code_gen",
            ),
        ]

        result = await orchestrator.execute(
            task={"request": "parallel task"},
            pattern=ConversationPattern.PARALLEL,
            stages=stages,
        )

        assert result.status == ConversationStatus.COMPLETED
        # Parallel merges results by stage name
        assert "research" in result.output or "code" in result.output

    @pytest.mark.asyncio
    async def test_execute_router(self, orchestrator, setup_agents):
        """Test router pattern execution."""
        stages = [
            ConversationStage(
                name="research_route",
                description="Research route",
                agent_capability="web_search",
            ),
            ConversationStage(
                name="code_route",
                description="Code route",
                agent_capability="code_gen",
            ),
        ]

        # Explicit route selection
        result = await orchestrator.execute(
            task={"request": "do research", "route": "research_route"},
            pattern=ConversationPattern.ROUTER,
            stages=stages,
        )

        assert result.status == ConversationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_no_agent_available(self, orchestrator):
        """Test error when no agent is available for capability."""
        from src.core.orchestrator import TaskExecutionError

        stages = [
            ConversationStage(
                name="unknown",
                description="Unknown",
                agent_capability="nonexistent_capability",
            ),
        ]

        with pytest.raises(TaskExecutionError) as exc_info:
            await orchestrator.execute(
                task={"request": "test"},
                pattern=ConversationPattern.SEQUENTIAL,
                stages=stages,
            )

        assert "nonexistent_capability" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_status(self, orchestrator, conversation_manager):
        """Test getting conversation status."""
        conv = await conversation_manager.create(
            name="Test",
            stages=[
                ConversationStage(
                    name="stage1", description="S1", agent_capability="test"
                ),
                ConversationStage(
                    name="stage2", description="S2", agent_capability="test"
                ),
            ],
        )
        await conversation_manager.start(conv.id)

        status = await orchestrator.get_status(conv.id)

        assert status["conversation_id"] == conv.id
        assert status["status"] == "in_progress"
        assert status["total_stages"] == 2

    @pytest.mark.asyncio
    async def test_cancel(self, orchestrator, conversation_manager):
        """Test cancelling a conversation."""
        conv = await conversation_manager.create(name="Test")
        await conversation_manager.start(conv.id)

        result = await orchestrator.cancel(conv.id)

        assert result.status == ConversationStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_execute_with_specific_agent_id(self, orchestrator, registry):
        """Test execution with specific agent ID in stage."""
        agent = create_mock_agent(
            "specific_agent",
            ["special"],
            {"result": {"data": "from specific agent"}},  # Must be dict
        )
        await registry.register(agent)

        stages = [
            ConversationStage(
                name="stage1",
                description="Stage with specific agent",
                agent_capability="special",
                agent_id="specific_agent",  # Specific agent
            ),
        ]

        result = await orchestrator.execute(
            task={"request": "test"},
            pattern=ConversationPattern.SEQUENTIAL,
            stages=stages,
        )

        assert result.status == ConversationStatus.COMPLETED
        assert len(agent.received_messages) == 1

    @pytest.mark.asyncio
    async def test_messages_recorded(
        self, orchestrator, setup_agents, conversation_manager
    ):
        """Test that messages are recorded in conversation."""
        stages = [
            ConversationStage(
                name="research",
                description="Research",
                agent_capability="web_search",
            ),
        ]

        result = await orchestrator.execute(
            task={"request": "test"},
            pattern=ConversationPattern.SEQUENTIAL,
            stages=stages,
        )

        # Get the conversation and check messages
        conv = await conversation_manager.get(result.conversation_id)
        assert len(conv.messages) >= 2  # At least task + response


class TestOrchestratorDebatePattern:
    """Test debate pattern specifically."""

    @pytest.fixture
    def registry(self):
        return AgentRegistry()

    @pytest.fixture
    def message_bus(self):
        return InMemoryMessageBus()

    @pytest.fixture
    def conversation_manager(self):
        return ConversationManager()

    @pytest.fixture
    def orchestrator(self, registry, message_bus, conversation_manager):
        return Orchestrator(registry, message_bus, conversation_manager)

    @pytest.mark.asyncio
    async def test_debate_reaches_consensus(self, orchestrator, registry):
        """Test debate pattern reaching consensus."""
        # Agents that signal consensus after a few calls
        call_count = {"value": 0}

        class ConsensusAgent(MockAgent):
            async def process(self, message: Message) -> Message:
                call_count["value"] += 1
                # _execute_single_stage extracts content.get("result", content)
                # so we need to put consensus/proposal inside "result" dict
                content = {
                    "result": {
                        "response": f"Response {call_count['value']}",
                        "consensus": call_count["value"] >= 2,
                        "proposal": "final proposal",
                    }
                }
                return Message.create_response(
                    sender_id=self.config.agent_id,
                    recipient_id=message.sender_id,
                    content=content,
                    correlation_id=message.correlation_id,
                )

        agent1_config = AgentConfig(
            agent_id="debater1",
            name="Debater 1",
            capabilities=[AgentCapability(name="debate1", description="Debate")],
        )
        agent2_config = AgentConfig(
            agent_id="debater2",
            name="Debater 2",
            capabilities=[AgentCapability(name="debate2", description="Debate")],
        )

        await registry.register(ConsensusAgent(agent1_config))
        await registry.register(ConsensusAgent(agent2_config))

        # Use different capabilities to ensure correct agent mapping
        stages = [
            ConversationStage(
                name="debater1", description="D1", agent_capability="debate1"
            ),
            ConversationStage(
                name="debater2", description="D2", agent_capability="debate2"
            ),
        ]

        result = await orchestrator.execute(
            task={"proposal": "initial idea"},
            pattern=ConversationPattern.DEBATE,
            stages=stages,
        )

        assert result.status == ConversationStatus.COMPLETED
        assert result.output.get("consensus") is True

    @pytest.mark.asyncio
    async def test_debate_max_iterations(self, orchestrator, registry):
        """Test debate pattern reaching max iterations."""
        # Agent that never reaches consensus
        stubborn_agent = create_mock_agent(
            "stubborn", ["debate"], {"result": "disagree", "consensus": False}
        )
        stubborn_agent2 = create_mock_agent(
            "stubborn2", ["debate"], {"result": "also disagree", "consensus": False}
        )

        await registry.register(stubborn_agent)
        await registry.register(stubborn_agent2)

        stages = [
            ConversationStage(
                name="stage1", description="S1", agent_capability="debate"
            ),
            ConversationStage(
                name="stage2", description="S2", agent_capability="debate"
            ),
        ]

        result = await orchestrator.execute(
            task={"proposal": "controversial topic"},
            pattern=ConversationPattern.DEBATE,
            stages=stages,
        )

        assert result.status == ConversationStatus.COMPLETED
        assert result.output.get("consensus") is False
        assert "Max iterations" in result.output.get("message", "")

    @pytest.mark.asyncio
    async def test_debate_requires_multiple_agents(self, orchestrator, registry):
        """Test that debate pattern fails with single stage."""
        agent = create_mock_agent("single", ["test"])
        await registry.register(agent)

        stages = [
            ConversationStage(
                name="only_one", description="Single", agent_capability="test"
            ),
        ]

        result = await orchestrator.execute(
            task={"proposal": "test"},
            pattern=ConversationPattern.DEBATE,
            stages=stages,
        )

        assert result.status == ConversationStatus.FAILED
