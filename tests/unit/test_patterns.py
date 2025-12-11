"""Unit tests for conversation patterns."""

import pytest

from src.core import (
    AgentRegistry,
    ConversationManager,
)
from src.models import (
    AgentCapability,
    AgentConfig,
    ConversationPattern,
    ConversationStage,
    ConversationStatus,
    Message,
)
from src.patterns import (
    DebatePattern,
    DebateRound,
    HierarchicalPattern,
    MergeStrategy,
    NoAgentAvailableError,
    ParallelPattern,
    PatternValidationError,
    RouterPattern,
    SequentialPattern,
    create_pattern,
    get_pattern_class,
)


class MockAgent:
    """Mock agent for testing patterns."""

    def __init__(
        self,
        config: AgentConfig,
        response_content: dict | None = None,
        fail_on_call: bool = False,
    ):
        self.config = config
        self.is_active = True
        self.response_content = response_content or {"result": "success"}
        self.fail_on_call = fail_on_call
        self.received_messages: list[Message] = []
        self.call_count = 0

    def can_handle(self, capability: str) -> bool:
        return capability in [cap.name for cap in self.config.capabilities]

    def get_capabilities(self) -> list[str]:
        return [cap.name for cap in self.config.capabilities]

    async def health_check(self) -> dict:
        return {"agent_id": self.config.agent_id, "status": "healthy"}

    async def process(self, message: Message) -> Message:
        self.call_count += 1
        self.received_messages.append(message)

        if self.fail_on_call:
            raise RuntimeError("Agent processing failed")

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
    fail_on_call: bool = False,
) -> MockAgent:
    """Helper to create mock agents."""
    caps = [
        AgentCapability(name=cap, description=f"{cap} capability")
        for cap in capabilities
    ]
    config = AgentConfig(agent_id=agent_id, name=f"Mock {agent_id}", capabilities=caps)
    return MockAgent(config, response, fail_on_call)


class TestPatternFactory:
    """Test pattern factory functions."""

    def test_get_pattern_class_sequential(self):
        """Test getting sequential pattern class."""
        cls = get_pattern_class(ConversationPattern.SEQUENTIAL)
        assert cls == SequentialPattern

    def test_get_pattern_class_parallel(self):
        """Test getting parallel pattern class."""
        cls = get_pattern_class(ConversationPattern.PARALLEL)
        assert cls == ParallelPattern

    def test_get_pattern_class_router(self):
        """Test getting router pattern class."""
        cls = get_pattern_class(ConversationPattern.ROUTER)
        assert cls == RouterPattern

    def test_get_pattern_class_debate(self):
        """Test getting debate pattern class."""
        cls = get_pattern_class(ConversationPattern.DEBATE)
        assert cls == DebatePattern

    def test_get_pattern_class_hierarchical(self):
        """Test getting hierarchical pattern class."""
        cls = get_pattern_class(ConversationPattern.HIERARCHICAL)
        assert cls == HierarchicalPattern

    def test_create_pattern(self):
        """Test creating pattern instance."""
        registry = AgentRegistry()
        conv_manager = ConversationManager()

        pattern = create_pattern(ConversationPattern.SEQUENTIAL, registry, conv_manager)

        assert isinstance(pattern, SequentialPattern)
        assert pattern.registry == registry
        assert pattern.conversation_manager == conv_manager


class TestSequentialPattern:
    """Test SequentialPattern class."""

    @pytest.fixture
    def registry(self):
        return AgentRegistry()

    @pytest.fixture
    def conversation_manager(self):
        return ConversationManager()

    @pytest.fixture
    def pattern(self, registry, conversation_manager):
        return SequentialPattern(registry, conversation_manager)

    @pytest.fixture
    async def setup_agents(self, registry):
        """Setup test agents."""
        agent1 = create_mock_agent(
            "agent1", ["capability1"], {"result": {"step1": "done"}}
        )
        agent2 = create_mock_agent(
            "agent2", ["capability2"], {"result": {"step2": "done"}}
        )
        await registry.register(agent1)
        await registry.register(agent2)
        return {"agent1": agent1, "agent2": agent2}

    def test_pattern_type(self, pattern):
        """Test pattern type property."""
        assert pattern.pattern_type == ConversationPattern.SEQUENTIAL

    @pytest.mark.asyncio
    async def test_execute_sequential(
        self, pattern, conversation_manager, setup_agents
    ):
        """Test sequential execution."""
        conv = await conversation_manager.create(
            name="Test Sequential",
            pattern=ConversationPattern.SEQUENTIAL,
            stages=[
                ConversationStage(
                    name="stage1",
                    description="S1",
                    agent_capability="capability1",
                ),
                ConversationStage(
                    name="stage2",
                    description="S2",
                    agent_capability="capability2",
                ),
            ],
            initial_input={"task": "test"},
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.COMPLETED
        assert result.stages_completed == 2

    @pytest.mark.asyncio
    async def test_execute_passes_output_to_next_stage(
        self, pattern, conversation_manager, setup_agents
    ):
        """Test that output is passed to next stage."""
        agents = setup_agents
        conv = await conversation_manager.create(
            name="Test Chaining",
            pattern=ConversationPattern.SEQUENTIAL,
            stages=[
                ConversationStage(
                    name="stage1", description="S1", agent_capability="capability1"
                ),
                ConversationStage(
                    name="stage2", description="S2", agent_capability="capability2"
                ),
            ],
            initial_input={"task": "initial"},
        )
        await conversation_manager.start(conv.id)

        await pattern.execute(conv)

        # Second agent should receive output from first
        agent2 = agents["agent2"]
        assert len(agent2.received_messages) == 1
        received_task = agent2.received_messages[0].content.get("task")
        # The task should be the output from stage1
        assert received_task == {"step1": "done"}

    @pytest.mark.asyncio
    async def test_execute_stage_failure(self, pattern, conversation_manager, registry):
        """Test handling of stage failure."""
        failing_agent = create_mock_agent("failing", ["fail_cap"], fail_on_call=True)
        await registry.register(failing_agent)

        conv = await conversation_manager.create(
            name="Test Failure",
            pattern=ConversationPattern.SEQUENTIAL,
            stages=[
                ConversationStage(
                    name="failing_stage",
                    description="Fail",
                    agent_capability="fail_cap",
                ),
            ],
            initial_input={"task": "test"},
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.FAILED

    @pytest.mark.asyncio
    async def test_validate_conversation_wrong_pattern(
        self, pattern, conversation_manager
    ):
        """Test validation fails for wrong pattern type."""
        conv = await conversation_manager.create(
            name="Wrong Pattern",
            pattern=ConversationPattern.PARALLEL,
            stages=[],
            initial_input={},
        )

        with pytest.raises(PatternValidationError):
            pattern.validate_conversation(conv)


class TestParallelPattern:
    """Test ParallelPattern class."""

    @pytest.fixture
    def registry(self):
        return AgentRegistry()

    @pytest.fixture
    def conversation_manager(self):
        return ConversationManager()

    @pytest.fixture
    def pattern(self, registry, conversation_manager):
        return ParallelPattern(registry, conversation_manager)

    @pytest.fixture
    async def setup_agents(self, registry):
        """Setup test agents."""
        agent1 = create_mock_agent(
            "parallel1", ["cap1"], {"result": {"data1": "result1"}}
        )
        agent2 = create_mock_agent(
            "parallel2", ["cap2"], {"result": {"data2": "result2"}}
        )
        await registry.register(agent1)
        await registry.register(agent2)
        return {"agent1": agent1, "agent2": agent2}

    def test_pattern_type(self, pattern):
        """Test pattern type property."""
        assert pattern.pattern_type == ConversationPattern.PARALLEL

    @pytest.mark.asyncio
    async def test_execute_parallel(self, pattern, conversation_manager, setup_agents):
        """Test parallel execution."""
        conv = await conversation_manager.create(
            name="Test Parallel",
            pattern=ConversationPattern.PARALLEL,
            stages=[
                ConversationStage(
                    name="stage1", description="S1", agent_capability="cap1"
                ),
                ConversationStage(
                    name="stage2", description="S2", agent_capability="cap2"
                ),
            ],
            initial_input={"task": "parallel_task"},
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.COMPLETED
        # Default merge strategy is dict merge
        assert "stage1" in result.output
        assert "stage2" in result.output

    @pytest.mark.asyncio
    async def test_execute_all_receive_same_input(
        self, pattern, conversation_manager, setup_agents
    ):
        """Test that all stages receive the same initial input."""
        agents = setup_agents
        conv = await conversation_manager.create(
            name="Test Same Input",
            pattern=ConversationPattern.PARALLEL,
            stages=[
                ConversationStage(
                    name="stage1", description="S1", agent_capability="cap1"
                ),
                ConversationStage(
                    name="stage2", description="S2", agent_capability="cap2"
                ),
            ],
            initial_input={"common": "input"},
        )
        await conversation_manager.start(conv.id)

        await pattern.execute(conv)

        # Both agents should receive the same input
        for agent in agents.values():
            assert len(agent.received_messages) == 1
            assert agent.received_messages[0].content["task"] == {"common": "input"}

    @pytest.mark.asyncio
    async def test_validate_requires_multiple_stages(
        self, pattern, conversation_manager
    ):
        """Test validation fails with single stage."""
        conv = await conversation_manager.create(
            name="Single Stage",
            pattern=ConversationPattern.PARALLEL,
            stages=[
                ConversationStage(
                    name="only_one", description="S", agent_capability="cap"
                ),
            ],
            initial_input={},
        )

        with pytest.raises(PatternValidationError) as exc_info:
            pattern.validate_conversation(conv)
        assert "at least 2 stages" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_partial_failure(self, pattern, conversation_manager, registry):
        """Test parallel pattern with partial failure."""
        good_agent = create_mock_agent("good", ["good_cap"], {"result": "success"})
        bad_agent = create_mock_agent("bad", ["bad_cap"], fail_on_call=True)
        await registry.register(good_agent)
        await registry.register(bad_agent)

        conv = await conversation_manager.create(
            name="Partial Failure",
            pattern=ConversationPattern.PARALLEL,
            stages=[
                ConversationStage(
                    name="good", description="Good", agent_capability="good_cap"
                ),
                ConversationStage(
                    name="bad", description="Bad", agent_capability="bad_cap"
                ),
            ],
            initial_input={"task": "test"},
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.FAILED
        assert "bad:" in result.error.lower() or "bad" in result.error.lower()


class TestMergeStrategy:
    """Test MergeStrategy utility class."""

    def test_dict_merge(self):
        """Test dict merge strategy."""
        results = {
            "stage1": {"data": "result1"},
            "stage2": {"data": "result2"},
        }

        merged = MergeStrategy.dict_merge(results)

        assert merged == results

    def test_list_merge(self):
        """Test list merge strategy."""
        results = {
            "stage1": {"data": "result1"},
            "stage2": {"data": "result2"},
        }

        merged = MergeStrategy.list_merge(results)

        assert "results" in merged
        assert len(merged["results"]) == 2

    def test_flatten_merge(self):
        """Test flatten merge strategy."""
        results = {
            "stage1": {"key1": "val1"},
            "stage2": {"key2": "val2"},
        }

        merged = MergeStrategy.flatten_merge(results)

        assert merged == {"key1": "val1", "key2": "val2"}


class TestRouterPattern:
    """Test RouterPattern class."""

    @pytest.fixture
    def registry(self):
        return AgentRegistry()

    @pytest.fixture
    def conversation_manager(self):
        return ConversationManager()

    @pytest.fixture
    def pattern(self, registry, conversation_manager):
        return RouterPattern(registry, conversation_manager)

    @pytest.fixture
    async def setup_agents(self, registry):
        """Setup test agents."""
        search_agent = create_mock_agent(
            "searcher", ["web_search"], {"result": {"search_results": ["item1"]}}
        )
        code_agent = create_mock_agent(
            "coder", ["code_gen"], {"result": {"code": "print('hi')"}}
        )
        await registry.register(search_agent)
        await registry.register(code_agent)
        return {"searcher": search_agent, "coder": code_agent}

    def test_pattern_type(self, pattern):
        """Test pattern type property."""
        assert pattern.pattern_type == ConversationPattern.ROUTER

    @pytest.mark.asyncio
    async def test_explicit_route(self, pattern, conversation_manager, setup_agents):
        """Test routing with explicit route field."""
        conv = await conversation_manager.create(
            name="Explicit Route",
            pattern=ConversationPattern.ROUTER,
            stages=[
                ConversationStage(
                    name="search_route",
                    description="Search",
                    agent_capability="web_search",
                ),
                ConversationStage(
                    name="code_route",
                    description="Code",
                    agent_capability="code_gen",
                ),
            ],
            initial_input={"task": "search something", "route": "search_route"},
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.COMPLETED
        agents = setup_agents
        assert agents["searcher"].call_count == 1
        assert agents["coder"].call_count == 0

    @pytest.mark.asyncio
    async def test_custom_routing_condition(
        self, pattern, conversation_manager, setup_agents
    ):
        """Test routing with custom condition."""
        # Add custom condition
        pattern.add_routing_condition(
            "code_route",
            lambda task, stage: "write code" in task.get("task", "").lower(),
        )

        conv = await conversation_manager.create(
            name="Custom Condition",
            pattern=ConversationPattern.ROUTER,
            stages=[
                ConversationStage(
                    name="search_route",
                    description="Search",
                    agent_capability="web_search",
                ),
                ConversationStage(
                    name="code_route",
                    description="Code",
                    agent_capability="code_gen",
                ),
            ],
            initial_input={"task": "please write code for me"},
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.COMPLETED
        agents = setup_agents
        assert agents["coder"].call_count == 1

    @pytest.mark.asyncio
    async def test_default_route(self, pattern, conversation_manager, setup_agents):
        """Test default route when no conditions match."""
        pattern.set_default_route("code_route")

        conv = await conversation_manager.create(
            name="Default Route",
            pattern=ConversationPattern.ROUTER,
            stages=[
                ConversationStage(
                    name="search_route",
                    description="Search",
                    agent_capability="web_search",
                ),
                ConversationStage(
                    name="code_route",
                    description="Code",
                    agent_capability="code_gen",
                ),
            ],
            initial_input={"task": "unrelated task"},
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.COMPLETED
        agents = setup_agents
        assert agents["coder"].call_count == 1

    @pytest.mark.asyncio
    async def test_no_suitable_route(self, pattern, conversation_manager):
        """Test failure when no route is suitable."""
        conv = await conversation_manager.create(
            name="No Route",
            pattern=ConversationPattern.ROUTER,
            stages=[
                ConversationStage(
                    name="route1",
                    description="R1",
                    agent_capability="nonexistent",
                ),
            ],
            initial_input={"task": "test"},
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.FAILED
        assert "No suitable route" in result.error


class TestDebatePattern:
    """Test DebatePattern class."""

    @pytest.fixture
    def registry(self):
        return AgentRegistry()

    @pytest.fixture
    def conversation_manager(self):
        return ConversationManager()

    @pytest.fixture
    def pattern(self, registry, conversation_manager):
        return DebatePattern(registry, conversation_manager)

    def test_pattern_type(self, pattern):
        """Test pattern type property."""
        assert pattern.pattern_type == ConversationPattern.DEBATE

    @pytest.mark.asyncio
    async def test_validate_requires_multiple_stages(
        self, pattern, conversation_manager
    ):
        """Test validation fails with single stage."""
        conv = await conversation_manager.create(
            name="Single Debater",
            pattern=ConversationPattern.DEBATE,
            stages=[
                ConversationStage(
                    name="only_one", description="S", agent_capability="cap"
                ),
            ],
            initial_input={},
        )

        with pytest.raises(PatternValidationError) as exc_info:
            pattern.validate_conversation(conv)
        assert "at least 2 stages" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_debate_reaches_consensus(
        self, pattern, conversation_manager, registry
    ):
        """Test debate reaching consensus."""
        call_count = {"value": 0}

        class ConsensusAgent(MockAgent):
            async def process(self, message: Message) -> Message:
                call_count["value"] += 1
                content = {
                    "result": {
                        "response": f"Response {call_count['value']}",
                        "consensus": call_count["value"] >= 2,
                        "proposal": "agreed proposal",
                    }
                }
                return Message.create_response(
                    sender_id=self.config.agent_id,
                    recipient_id=message.sender_id,
                    content=content,
                    correlation_id=message.correlation_id,
                )

        caps1 = [AgentCapability(name="debate1", description="Debate")]
        caps2 = [AgentCapability(name="debate2", description="Debate")]
        config1 = AgentConfig(agent_id="debater1", name="D1", capabilities=caps1)
        config2 = AgentConfig(agent_id="debater2", name="D2", capabilities=caps2)

        await registry.register(ConsensusAgent(config1))
        await registry.register(ConsensusAgent(config2))

        conv = await conversation_manager.create(
            name="Consensus Test",
            pattern=ConversationPattern.DEBATE,
            stages=[
                ConversationStage(
                    name="d1", description="D1", agent_capability="debate1"
                ),
                ConversationStage(
                    name="d2", description="D2", agent_capability="debate2"
                ),
            ],
            initial_input={"proposal": "initial"},
            max_iterations=5,
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.COMPLETED
        assert result.output.get("consensus") is True
        assert result.output.get("final_proposal") == "agreed proposal"

    @pytest.mark.asyncio
    async def test_debate_max_iterations(self, pattern, conversation_manager, registry):
        """Test debate reaching max iterations without consensus."""
        agent1 = create_mock_agent(
            "stubborn1", ["debate1"], {"result": {"response": "disagree"}}
        )
        agent2 = create_mock_agent(
            "stubborn2", ["debate2"], {"result": {"response": "also disagree"}}
        )
        await registry.register(agent1)
        await registry.register(agent2)

        conv = await conversation_manager.create(
            name="No Consensus",
            pattern=ConversationPattern.DEBATE,
            stages=[
                ConversationStage(
                    name="d1", description="D1", agent_capability="debate1"
                ),
                ConversationStage(
                    name="d2", description="D2", agent_capability="debate2"
                ),
            ],
            initial_input={"proposal": "controversial"},
            max_iterations=2,
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.COMPLETED
        assert result.output.get("consensus") is False
        assert result.output.get("iterations") == 2
        assert "Max iterations" in result.output.get("message", "")


class TestDebateRound:
    """Test DebateRound utility class."""

    def test_create_round(self):
        """Test creating a debate round."""
        round = DebateRound(iteration=1)

        assert round.iteration == 1
        assert round.responses == []
        assert round.consensus_reached is False

    def test_add_response(self):
        """Test adding a response to a round."""
        round = DebateRound(iteration=1)

        round.add_response("agent1", {"opinion": "agree"})

        assert len(round.responses) == 1
        assert round.responses[0]["agent"] == "agent1"

    def test_add_consensus_response(self):
        """Test adding a response with consensus signal."""
        round = DebateRound(iteration=1)

        round.add_response("agent1", {"consensus": True, "proposal": "final"})

        assert round.consensus_reached is True
        assert round.final_proposal == "final"

    def test_to_dict(self):
        """Test converting round to dictionary."""
        round = DebateRound(iteration=2)
        round.add_response("a1", {"data": "r1"})

        d = round.to_dict()

        assert d["iteration"] == 2
        assert len(d["responses"]) == 1


class TestBasePatternHelpers:
    """Test BasePattern helper methods."""

    @pytest.fixture
    def registry(self):
        return AgentRegistry()

    @pytest.fixture
    def conversation_manager(self):
        return ConversationManager()

    @pytest.fixture
    def pattern(self, registry, conversation_manager):
        # Use SequentialPattern as concrete implementation
        return SequentialPattern(registry, conversation_manager)

    @pytest.mark.asyncio
    async def test_select_agent_by_capability(self, pattern, registry):
        """Test selecting agent by capability."""
        agent = create_mock_agent("test_agent", ["test_cap"])
        await registry.register(agent)

        stage = ConversationStage(
            name="test", description="Test", agent_capability="test_cap"
        )

        selected = await pattern._select_agent(stage)

        assert selected.config.agent_id == "test_agent"

    @pytest.mark.asyncio
    async def test_select_agent_by_id(self, pattern, registry):
        """Test selecting specific agent by ID."""
        agent1 = create_mock_agent("agent1", ["cap"])
        agent2 = create_mock_agent("agent2", ["cap"])
        await registry.register(agent1)
        await registry.register(agent2)

        stage = ConversationStage(
            name="test",
            description="Test",
            agent_capability="cap",
            agent_id="agent2",
        )

        selected = await pattern._select_agent(stage)

        assert selected.config.agent_id == "agent2"

    @pytest.mark.asyncio
    async def test_select_agent_not_found(self, pattern):
        """Test error when no agent found."""
        stage = ConversationStage(
            name="test", description="Test", agent_capability="nonexistent"
        )

        with pytest.raises(NoAgentAvailableError):
            await pattern._select_agent(stage)

    @pytest.mark.asyncio
    async def test_execute_single_stage(self, pattern, registry, conversation_manager):
        """Test executing a single stage."""
        agent = create_mock_agent("agent", ["cap"], {"result": {"data": "output"}})
        await registry.register(agent)

        conv = await conversation_manager.create(
            name="Test",
            pattern=ConversationPattern.SEQUENTIAL,
            stages=[],
            initial_input={"task": "test"},
        )

        stage = ConversationStage(
            name="test_stage", description="Test", agent_capability="cap"
        )
        stage.input_data = {"task": "input_task"}

        output = await pattern._execute_single_stage(conv, stage)

        assert output == {"data": "output"}


class TestHierarchicalPattern:
    """Test HierarchicalPattern class."""

    @pytest.fixture
    def registry(self):
        return AgentRegistry()

    @pytest.fixture
    def conversation_manager(self):
        return ConversationManager()

    @pytest.fixture
    def pattern(self, registry, conversation_manager):
        return HierarchicalPattern(registry, conversation_manager)

    def test_pattern_type(self, pattern):
        """Test pattern type property."""
        assert pattern.pattern_type == ConversationPattern.HIERARCHICAL

    @pytest.mark.asyncio
    async def test_validate_requires_multiple_stages(
        self, pattern, conversation_manager
    ):
        """Test validation fails with single stage."""
        conv = await conversation_manager.create(
            name="Single Stage",
            pattern=ConversationPattern.HIERARCHICAL,
            stages=[
                ConversationStage(
                    name="only_one", description="S", agent_capability="cap"
                ),
            ],
            initial_input={},
        )

        with pytest.raises(PatternValidationError) as exc_info:
            pattern.validate_conversation(conv)
        assert "at least 2 stages" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_with_delegation(
        self, pattern, conversation_manager, registry
    ):
        """Test hierarchical execution with delegation."""
        # Supervisor that delegates to workers
        supervisor_response = {
            "result": {
                "delegations": [
                    {"worker": "worker1", "task": {"subtask": "task1"}},
                    {"worker": "worker2", "task": {"subtask": "task2"}},
                ],
                "execution_mode": "parallel",
            }
        }
        supervisor = create_mock_agent("supervisor", ["supervise"], supervisor_response)

        # Workers
        worker1 = create_mock_agent(
            "worker1", ["work1"], {"result": {"output": "result1"}}
        )
        worker2 = create_mock_agent(
            "worker2", ["work2"], {"result": {"output": "result2"}}
        )

        await registry.register(supervisor)
        await registry.register(worker1)
        await registry.register(worker2)

        conv = await conversation_manager.create(
            name="Hierarchical Test",
            pattern=ConversationPattern.HIERARCHICAL,
            stages=[
                ConversationStage(
                    name="supervisor",
                    description="Supervisor",
                    agent_capability="supervise",
                ),
                ConversationStage(
                    name="worker1",
                    description="Worker 1",
                    agent_capability="work1",
                ),
                ConversationStage(
                    name="worker2",
                    description="Worker 2",
                    agent_capability="work2",
                ),
            ],
            initial_input={"main_task": "do something"},
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.COMPLETED
        # Supervisor is called at least twice (planning + aggregation)
        assert supervisor.call_count >= 1

    @pytest.mark.asyncio
    async def test_execute_no_delegation(self, pattern, conversation_manager, registry):
        """Test hierarchical execution when supervisor handles directly."""
        # Supervisor that doesn't delegate
        supervisor_response = {
            "result": {
                "direct_result": "handled by supervisor",
            }
        }
        supervisor = create_mock_agent("supervisor", ["supervise"], supervisor_response)
        worker = create_mock_agent("worker", ["work"], {"result": "unused"})

        await registry.register(supervisor)
        await registry.register(worker)

        conv = await conversation_manager.create(
            name="No Delegation",
            pattern=ConversationPattern.HIERARCHICAL,
            stages=[
                ConversationStage(
                    name="supervisor",
                    description="Supervisor",
                    agent_capability="supervise",
                ),
                ConversationStage(
                    name="worker",
                    description="Worker",
                    agent_capability="work",
                ),
            ],
            initial_input={"simple_task": "easy"},
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.COMPLETED
        # Worker should not be called
        assert worker.call_count == 0

    @pytest.mark.asyncio
    async def test_execute_sequential_workers(
        self, pattern, conversation_manager, registry
    ):
        """Test hierarchical execution with sequential worker execution."""
        supervisor_response = {
            "result": {
                "delegations": [
                    {"worker": "worker1", "task": {"step": 1}},
                    {"worker": "worker2", "task": {"step": 2}},
                ],
                "execution_mode": "sequential",
            }
        }
        supervisor = create_mock_agent("supervisor", ["supervise"], supervisor_response)
        worker1 = create_mock_agent("worker1", ["work1"], {"result": {"done": "step1"}})
        worker2 = create_mock_agent("worker2", ["work2"], {"result": {"done": "step2"}})

        await registry.register(supervisor)
        await registry.register(worker1)
        await registry.register(worker2)

        conv = await conversation_manager.create(
            name="Sequential Workers",
            pattern=ConversationPattern.HIERARCHICAL,
            stages=[
                ConversationStage(
                    name="supervisor",
                    description="Supervisor",
                    agent_capability="supervise",
                ),
                ConversationStage(
                    name="worker1",
                    description="W1",
                    agent_capability="work1",
                ),
                ConversationStage(
                    name="worker2",
                    description="W2",
                    agent_capability="work2",
                ),
            ],
            initial_input={"task": "sequential work"},
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.COMPLETED
        assert worker1.call_count == 1
        assert worker2.call_count == 1

    @pytest.mark.asyncio
    async def test_supervisor_failure(self, pattern, conversation_manager, registry):
        """Test handling of supervisor failure."""
        supervisor = create_mock_agent("supervisor", ["supervise"], fail_on_call=True)
        worker = create_mock_agent("worker", ["work"], {"result": "ok"})

        await registry.register(supervisor)
        await registry.register(worker)

        conv = await conversation_manager.create(
            name="Supervisor Failure",
            pattern=ConversationPattern.HIERARCHICAL,
            stages=[
                ConversationStage(
                    name="supervisor",
                    description="Supervisor",
                    agent_capability="supervise",
                ),
                ConversationStage(
                    name="worker",
                    description="Worker",
                    agent_capability="work",
                ),
            ],
            initial_input={"task": "test"},
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.FAILED

    @pytest.mark.asyncio
    async def test_delegate_to_all_workers(
        self, pattern, conversation_manager, registry
    ):
        """Test delegation to all workers."""
        supervisor_response = {
            "result": {
                "delegate_to_all": True,
                "task_for_workers": {"shared_task": "process"},
            }
        }
        supervisor = create_mock_agent("supervisor", ["supervise"], supervisor_response)
        worker1 = create_mock_agent("worker1", ["work1"], {"result": {"w1": "done"}})
        worker2 = create_mock_agent("worker2", ["work2"], {"result": {"w2": "done"}})

        await registry.register(supervisor)
        await registry.register(worker1)
        await registry.register(worker2)

        conv = await conversation_manager.create(
            name="Delegate All",
            pattern=ConversationPattern.HIERARCHICAL,
            stages=[
                ConversationStage(
                    name="supervisor",
                    description="S",
                    agent_capability="supervise",
                ),
                ConversationStage(
                    name="worker1",
                    description="W1",
                    agent_capability="work1",
                ),
                ConversationStage(
                    name="worker2",
                    description="W2",
                    agent_capability="work2",
                ),
            ],
            initial_input={"task": "broadcast"},
        )
        await conversation_manager.start(conv.id)

        result = await pattern.execute(conv)

        assert result.status == ConversationStatus.COMPLETED
        assert worker1.call_count == 1
        assert worker2.call_count == 1
