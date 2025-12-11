"""Unit tests for agent implementations (ResearchAgent, CoderAgent, ReviewerAgent)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.implementations.coder import CoderAgent
from src.agents.implementations.researcher import ResearchAgent
from src.agents.implementations.reviewer import ReviewerAgent
from src.models import (
    AgentCapability,
    AgentConfig,
    AgentStatus,
    Message,
    MessageType,
)


@pytest.fixture
def research_config() -> AgentConfig:
    """Create a research agent configuration."""
    return AgentConfig(
        agent_id="researcher_001",
        name="Research Agent",
        description="Research agent for testing",
        capabilities=[
            AgentCapability(name="web_search", description="Search the web"),
            AgentCapability(name="summarization", description="Summarize text"),
            AgentCapability(name="data_extraction", description="Extract data"),
        ],
    )


@pytest.fixture
def coder_config() -> AgentConfig:
    """Create a coder agent configuration."""
    return AgentConfig(
        agent_id="coder_001",
        name="Coder Agent",
        description="Coder agent for testing",
        capabilities=[
            AgentCapability(name="code_generation", description="Generate code"),
            AgentCapability(name="code_modification", description="Modify code"),
            AgentCapability(name="code_explanation", description="Explain code"),
            AgentCapability(name="debugging", description="Debug code"),
        ],
    )


@pytest.fixture
def reviewer_config() -> AgentConfig:
    """Create a reviewer agent configuration."""
    return AgentConfig(
        agent_id="reviewer_001",
        name="Reviewer Agent",
        description="Reviewer agent for testing",
        capabilities=[
            AgentCapability(name="code_review", description="Review code"),
            AgentCapability(name="security_review", description="Security review"),
            AgentCapability(
                name="performance_review", description="Performance review"
            ),
            AgentCapability(name="style_review", description="Style review"),
        ],
    )


@pytest.fixture
def researcher(research_config: AgentConfig) -> ResearchAgent:
    """Create a ResearchAgent instance."""
    return ResearchAgent(research_config)


@pytest.fixture
def coder(coder_config: AgentConfig) -> CoderAgent:
    """Create a CoderAgent instance."""
    return CoderAgent(coder_config)


@pytest.fixture
def reviewer(reviewer_config: AgentConfig) -> ReviewerAgent:
    """Create a ReviewerAgent instance."""
    return ReviewerAgent(reviewer_config)


class TestResearchAgent:
    """Tests for ResearchAgent."""

    def test_init_sets_default_system_prompt(
        self, research_config: AgentConfig
    ) -> None:
        """Test that init sets default system prompt if not provided."""
        agent = ResearchAgent(research_config)
        assert agent.config.system_prompt is not None
        assert "Research Agent" in agent.config.system_prompt

    def test_init_preserves_custom_system_prompt(self) -> None:
        """Test that init preserves custom system prompt."""
        config = AgentConfig(
            agent_id="researcher_custom",
            name="Custom Researcher",
            system_prompt="Custom prompt",
        )
        agent = ResearchAgent(config)
        assert agent.config.system_prompt == "Custom prompt"

    @pytest.mark.asyncio
    async def test_process_web_search(self, researcher: ResearchAgent) -> None:
        """Test processing web search capability."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="researcher_001",
            content={
                "capability": "web_search",
                "query": "AI trends 2025",
                "max_results": 5,
            },
        )

        with patch.object(researcher, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Search results for AI trends 2025..."

            response = await researcher.process(message)

            assert response.message_type == MessageType.RESPONSE
            assert response.content["capability"] == "web_search"
            assert response.content["query"] == "AI trends 2025"
            assert "result" in response.content

    @pytest.mark.asyncio
    async def test_process_summarization(self, researcher: ResearchAgent) -> None:
        """Test processing summarization capability."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="researcher_001",
            content={
                "capability": "summarization",
                "text": "Long text to summarize...",
                "max_length": 100,
            },
        )

        with patch.object(researcher, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Summary of the text..."

            response = await researcher.process(message)

            assert response.message_type == MessageType.RESPONSE
            assert response.content["capability"] == "summarization"
            assert "summary" in response.content

    @pytest.mark.asyncio
    async def test_process_data_extraction(self, researcher: ResearchAgent) -> None:
        """Test processing data extraction capability."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="researcher_001",
            content={
                "capability": "data_extraction",
                "text": "John Smith works at Acme Corp since 2020.",
                "fields": ["name", "company", "year"],
            },
        )

        with patch.object(researcher, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (
                "Extracted: name=John Smith, company=Acme Corp, year=2020"
            )

            response = await researcher.process(message)

            assert response.message_type == MessageType.RESPONSE
            assert response.content["capability"] == "data_extraction"
            assert "extracted_data" in response.content

    @pytest.mark.asyncio
    async def test_process_general_research(self, researcher: ResearchAgent) -> None:
        """Test processing general research task."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="researcher_001",
            content={
                "capability": "general_research",
                "task": "Research quantum computing basics",
            },
        )

        with patch.object(researcher, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Quantum computing overview..."

            response = await researcher.process(message)

            assert response.message_type == MessageType.RESPONSE
            assert response.content["capability"] == "general_research"

    @pytest.mark.asyncio
    async def test_process_error_handling(self, researcher: ResearchAgent) -> None:
        """Test error handling in process method."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="researcher_001",
            content={"task": "Research something"},
        )

        with patch.object(researcher, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("Research failed")

            response = await researcher.process(message)

            assert response.message_type == MessageType.ERROR
            assert "RESEARCH_ERROR" in response.content.get("error_code", "")


class TestCoderAgent:
    """Tests for CoderAgent."""

    def test_init_sets_default_system_prompt(self, coder_config: AgentConfig) -> None:
        """Test that init sets default system prompt if not provided."""
        agent = CoderAgent(coder_config)
        assert agent.config.system_prompt is not None
        assert "Software Engineer" in agent.config.system_prompt

    @pytest.mark.asyncio
    async def test_process_code_generation(self, coder: CoderAgent) -> None:
        """Test processing code generation capability."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="coder_001",
            content={
                "capability": "code_generation",
                "requirements": "Create a function to calculate factorial",
                "language": "python",
            },
        )

        with patch.object(coder, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (
                "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
            )

            response = await coder.process(message)

            assert response.message_type == MessageType.RESPONSE
            assert response.content["capability"] == "code_generation"
            assert response.content["language"] == "python"
            assert "code" in response.content

    @pytest.mark.asyncio
    async def test_process_code_modification(self, coder: CoderAgent) -> None:
        """Test processing code modification capability."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="coder_001",
            content={
                "capability": "code_modification",
                "code": "def add(a, b): return a + b",
                "modification": "Add type hints",
                "language": "python",
            },
        )

        with patch.object(coder, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "def add(a: int, b: int) -> int: return a + b"

            response = await coder.process(message)

            assert response.message_type == MessageType.RESPONSE
            assert response.content["capability"] == "code_modification"
            assert "modified_code" in response.content

    @pytest.mark.asyncio
    async def test_process_code_explanation(self, coder: CoderAgent) -> None:
        """Test processing code explanation capability."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="coder_001",
            content={
                "capability": "code_explanation",
                "code": "x = [i**2 for i in range(10)]",
                "language": "python",
                "detail_level": "comprehensive",
            },
        )

        with patch.object(coder, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (
                "This is a list comprehension that creates squares..."
            )

            response = await coder.process(message)

            assert response.message_type == MessageType.RESPONSE
            assert response.content["capability"] == "code_explanation"
            assert "explanation" in response.content

    @pytest.mark.asyncio
    async def test_process_debugging(self, coder: CoderAgent) -> None:
        """Test processing debugging capability."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="coder_001",
            content={
                "capability": "debugging",
                "code": "def divide(a, b): return a / b",
                "error": "ZeroDivisionError when b is 0",
                "language": "python",
            },
        )

        with patch.object(coder, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Add check for zero: if b == 0: raise ValueError"

            response = await coder.process(message)

            assert response.message_type == MessageType.RESPONSE
            assert response.content["capability"] == "debugging"
            assert "debug_result" in response.content

    @pytest.mark.asyncio
    async def test_process_general_coding(self, coder: CoderAgent) -> None:
        """Test processing general coding task."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="coder_001",
            content={
                "capability": "general_coding",
                "task": "Write a hello world program",
            },
        )

        with patch.object(coder, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "print('Hello, World!')"

            response = await coder.process(message)

            assert response.message_type == MessageType.RESPONSE
            assert response.content["capability"] == "general_coding"


class TestReviewerAgent:
    """Tests for ReviewerAgent."""

    def test_init_sets_default_system_prompt(
        self, reviewer_config: AgentConfig
    ) -> None:
        """Test that init sets default system prompt if not provided."""
        agent = ReviewerAgent(reviewer_config)
        assert agent.config.system_prompt is not None
        assert "Code Reviewer" in agent.config.system_prompt

    @pytest.mark.asyncio
    async def test_process_code_review(self, reviewer: ReviewerAgent) -> None:
        """Test processing code review capability."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="reviewer_001",
            content={
                "capability": "code_review",
                "code": "def foo(x): return x * 2",
                "language": "python",
                "context": "Utility function",
            },
        )

        with patch.object(reviewer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Code review: Function is simple and correct..."

            response = await reviewer.process(message)

            assert response.message_type == MessageType.RESPONSE
            assert response.content["capability"] == "code_review"
            assert "review" in response.content

    @pytest.mark.asyncio
    async def test_process_security_review(self, reviewer: ReviewerAgent) -> None:
        """Test processing security review capability."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="reviewer_001",
            content={
                "capability": "security_review",
                "code": "query = f'SELECT * FROM users WHERE id = {user_id}'",
                "language": "python",
            },
        )

        with patch.object(reviewer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "CRITICAL: SQL Injection vulnerability detected..."

            response = await reviewer.process(message)

            assert response.message_type == MessageType.RESPONSE
            assert response.content["capability"] == "security_review"
            assert "security_review" in response.content

    @pytest.mark.asyncio
    async def test_process_performance_review(self, reviewer: ReviewerAgent) -> None:
        """Test processing performance review capability."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="reviewer_001",
            content={
                "capability": "performance_review",
                "code": "for i in range(len(arr)): for j in range(len(arr)): pass",
                "language": "python",
            },
        )

        with patch.object(reviewer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Time complexity: O(n^2), consider optimization..."

            response = await reviewer.process(message)

            assert response.message_type == MessageType.RESPONSE
            assert response.content["capability"] == "performance_review"
            assert "performance_review" in response.content

    @pytest.mark.asyncio
    async def test_process_style_review(self, reviewer: ReviewerAgent) -> None:
        """Test processing style review capability."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="reviewer_001",
            content={
                "capability": "style_review",
                "code": "def MyFunction(X,Y): return X+Y",
                "language": "python",
                "style_guide": "PEP 8",
            },
        )

        with patch.object(reviewer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Style issues: Use snake_case for function names..."

            response = await reviewer.process(message)

            assert response.message_type == MessageType.RESPONSE
            assert response.content["capability"] == "style_review"
            assert "style_review" in response.content
            assert response.content["style_guide"] == "PEP 8"

    @pytest.mark.asyncio
    async def test_process_default_code_review(self, reviewer: ReviewerAgent) -> None:
        """Test that unknown capability defaults to code review."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="reviewer_001",
            content={
                "capability": "unknown_review_type",
                "code": "print('test')",
            },
        )

        with patch.object(reviewer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "General code review..."

            response = await reviewer.process(message)

            assert response.message_type == MessageType.RESPONSE
            assert response.content["capability"] == "code_review"


class TestAgentStatusManagement:
    """Test agent status management across implementations."""

    @pytest.mark.asyncio
    async def test_research_agent_status_on_success(
        self, researcher: ResearchAgent
    ) -> None:
        """Test researcher status is ACTIVE after successful processing."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="researcher_001",
            content={"task": "Research something"},
        )

        with patch.object(researcher, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Research results"

            await researcher.process(message)

            assert researcher.status == AgentStatus.ACTIVE
            assert researcher.is_active is True

    @pytest.mark.asyncio
    async def test_coder_agent_status_on_error(self, coder: CoderAgent) -> None:
        """Test coder status is ERROR after failed processing."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="coder_001",
            content={"task": "Generate code"},
        )

        with patch.object(coder, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM Error")

            await coder.process(message)

            assert coder.status == AgentStatus.ERROR
            assert coder.is_active is False

    @pytest.mark.asyncio
    async def test_reviewer_agent_status_during_processing(
        self, reviewer: ReviewerAgent
    ) -> None:
        """Test reviewer status is BUSY during processing."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="reviewer_001",
            content={"code": "print('test')"},
        )

        status_during_processing = None

        async def capture_status(*args, **kwargs):
            nonlocal status_during_processing
            status_during_processing = reviewer.status
            return "Review results"

        with patch.object(reviewer, "_call_llm", side_effect=capture_status):
            await reviewer.process(message)

            assert status_during_processing == AgentStatus.BUSY
