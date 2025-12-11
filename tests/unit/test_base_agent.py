"""Unit tests for BaseAgent and SimpleAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base import (
    LLMError,
    SimpleAgent,
)
from src.models import (
    AgentCapability,
    AgentConfig,
    AgentStatus,
    Message,
    MessageType,
)


@pytest.fixture
def sample_config() -> AgentConfig:
    """Create a sample agent configuration."""
    return AgentConfig(
        agent_id="test_agent_001",
        name="Test Agent",
        description="A test agent",
        capabilities=[
            AgentCapability(
                name="test_capability",
                description="A test capability",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            ),
            AgentCapability(
                name="another_capability",
                description="Another capability",
            ),
        ],
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        temperature=0.7,
        system_prompt="You are a test agent.",
    )


@pytest.fixture
def simple_agent(sample_config: AgentConfig) -> SimpleAgent:
    """Create a SimpleAgent instance."""
    return SimpleAgent(sample_config)


@pytest.fixture
def sample_message() -> Message:
    """Create a sample message for testing."""
    return Message.create_task(
        sender_id="orchestrator",
        recipient_id="test_agent_001",
        content={"task": "Do something useful"},
        correlation_id="corr_001",
    )


class TestBaseAgentProperties:
    """Test BaseAgent property accessors."""

    def test_config_property(
        self, simple_agent: SimpleAgent, sample_config: AgentConfig
    ) -> None:
        """Test config property returns correct configuration."""
        assert simple_agent.config == sample_config
        assert simple_agent.config.agent_id == "test_agent_001"
        assert simple_agent.config.name == "Test Agent"

    def test_is_active_default(self, simple_agent: SimpleAgent) -> None:
        """Test agent is active by default."""
        assert simple_agent.is_active is True

    def test_status_default(self, simple_agent: SimpleAgent) -> None:
        """Test agent status is ACTIVE by default."""
        assert simple_agent.status == AgentStatus.ACTIVE


class TestBaseAgentStatusMethods:
    """Test BaseAgent status management methods."""

    def test_deactivate(self, simple_agent: SimpleAgent) -> None:
        """Test deactivating an agent."""
        simple_agent.deactivate()
        assert simple_agent.is_active is False
        assert simple_agent.status == AgentStatus.INACTIVE

    def test_activate(self, simple_agent: SimpleAgent) -> None:
        """Test activating an agent."""
        simple_agent.deactivate()
        simple_agent.activate()
        assert simple_agent.is_active is True
        assert simple_agent.status == AgentStatus.ACTIVE

    def test_set_busy(self, simple_agent: SimpleAgent) -> None:
        """Test setting agent to busy status."""
        simple_agent.set_busy()
        assert simple_agent.status == AgentStatus.BUSY
        # Agent is still active when busy
        assert simple_agent.is_active is True

    def test_set_error(self, simple_agent: SimpleAgent) -> None:
        """Test setting agent to error status."""
        simple_agent.set_error()
        assert simple_agent.status == AgentStatus.ERROR
        assert simple_agent.is_active is False


class TestBaseAgentCapabilities:
    """Test BaseAgent capability methods."""

    def test_can_handle_existing_capability(self, simple_agent: SimpleAgent) -> None:
        """Test can_handle returns True for existing capability."""
        assert simple_agent.can_handle("test_capability") is True
        assert simple_agent.can_handle("another_capability") is True

    def test_can_handle_nonexistent_capability(self, simple_agent: SimpleAgent) -> None:
        """Test can_handle returns False for non-existent capability."""
        assert simple_agent.can_handle("nonexistent") is False

    def test_get_capabilities(self, simple_agent: SimpleAgent) -> None:
        """Test get_capabilities returns all capability names."""
        capabilities = simple_agent.get_capabilities()
        assert len(capabilities) == 2
        assert "test_capability" in capabilities
        assert "another_capability" in capabilities


class TestBaseAgentHealthCheck:
    """Test BaseAgent health check."""

    @pytest.mark.asyncio
    async def test_health_check_active(self, simple_agent: SimpleAgent) -> None:
        """Test health check for active agent."""
        result = await simple_agent.health_check()

        assert result["agent_id"] == "test_agent_001"
        assert result["name"] == "Test Agent"
        assert result["status"] == "healthy"
        assert "test_capability" in result["capabilities"]
        assert result["model"] == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_health_check_inactive(self, simple_agent: SimpleAgent) -> None:
        """Test health check for inactive agent."""
        simple_agent.deactivate()
        result = await simple_agent.health_check()

        assert result["status"] == "unhealthy"


class TestLLMError:
    """Test LLMError exception."""

    def test_llm_error_message(self) -> None:
        """Test LLMError contains correct message."""
        error = LLMError("Test error message")
        assert str(error) == "Test error message"
        assert error.original_error is None

    def test_llm_error_with_original(self) -> None:
        """Test LLMError with original exception."""
        original = ValueError("Original error")
        error = LLMError("Wrapped error", original_error=original)
        assert error.original_error == original


class TestSimpleAgentProcess:
    """Test SimpleAgent process method."""

    @pytest.mark.asyncio
    async def test_process_success(
        self,
        simple_agent: SimpleAgent,
        sample_message: Message,
    ) -> None:
        """Test successful message processing."""
        with patch.object(
            simple_agent, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = "Test response from LLM"

            response = await simple_agent.process(sample_message)

            assert response.message_type == MessageType.RESPONSE
            assert response.sender_id == "test_agent_001"
            assert response.recipient_id == "orchestrator"
            assert response.correlation_id == sample_message.correlation_id
            assert response.content["result"] == "Test response from LLM"
            assert simple_agent.status == AgentStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_process_llm_error(
        self,
        simple_agent: SimpleAgent,
        sample_message: Message,
    ) -> None:
        """Test message processing with LLM error."""
        with patch.object(
            simple_agent, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.side_effect = LLMError("API call failed")

            response = await simple_agent.process(sample_message)

            assert response.message_type == MessageType.ERROR
            assert "API call failed" in response.content["error"]
            assert response.content.get("error_code") == "LLM_ERROR"
            assert simple_agent.status == AgentStatus.ERROR

    @pytest.mark.asyncio
    async def test_process_unexpected_error(
        self,
        simple_agent: SimpleAgent,
        sample_message: Message,
    ) -> None:
        """Test message processing with unexpected error."""
        with patch.object(
            simple_agent, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.side_effect = RuntimeError("Unexpected error")

            response = await simple_agent.process(sample_message)

            assert response.message_type == MessageType.ERROR
            assert "Unexpected error" in response.content["error"]
            assert response.content.get("error_code") == "AGENT_ERROR"

    @pytest.mark.asyncio
    async def test_process_extracts_task_from_query(
        self,
        simple_agent: SimpleAgent,
    ) -> None:
        """Test that process extracts task from 'query' field if 'task' is missing."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="test_agent_001",
            content={"query": "What is the answer?"},
        )

        with patch.object(
            simple_agent, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = "The answer is 42"

            await simple_agent.process(message)

            # Verify _call_llm was called with the query
            call_args = mock_llm.call_args
            assert "What is the answer?" in call_args[0][0]


class TestBaseAgentLLMCall:
    """Test BaseAgent LLM calling functionality."""

    @pytest.mark.asyncio
    async def test_get_llm_provider_creates_provider(
        self, simple_agent: SimpleAgent
    ) -> None:
        """Test that _get_llm_provider creates a provider based on model."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = simple_agent._get_llm_provider()
            assert provider is not None
            assert provider.provider_name == "anthropic"

    @pytest.mark.asyncio
    async def test_call_llm_with_context(self, simple_agent: SimpleAgent) -> None:
        """Test _call_llm includes context in prompt."""
        from src.llm.base import LLMResponse

        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(
            return_value=LLMResponse(content="Response text", model="test-model")
        )

        simple_agent.set_llm_provider(mock_provider)

        result = await simple_agent._call_llm(
            prompt="Test prompt",
            context={"key1": "value1", "key2": "value2"},
        )

        # Verify the call was made
        mock_provider.chat.assert_called_once()
        call_kwargs = mock_provider.chat.call_args[1]

        # Check the messages include context
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert "Context:" in messages[0]["content"]
        assert "key1: value1" in messages[0]["content"]
        assert "Test prompt" in messages[0]["content"]

        assert result == "Response text"

    @pytest.mark.asyncio
    async def test_call_llm_uses_config_settings(
        self, simple_agent: SimpleAgent
    ) -> None:
        """Test _call_llm uses configuration settings."""
        from src.llm.base import LLMResponse

        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(
            return_value=LLMResponse(content="Response", model="test-model")
        )

        simple_agent.set_llm_provider(mock_provider)

        await simple_agent._call_llm("Test prompt")

        call_kwargs = mock_provider.chat.call_args[1]

        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["system_prompt"] == "You are a test agent."

    @pytest.mark.asyncio
    async def test_call_llm_custom_system_prompt(
        self, simple_agent: SimpleAgent
    ) -> None:
        """Test _call_llm with custom system prompt override."""
        from src.llm.base import LLMResponse

        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(
            return_value=LLMResponse(content="Response", model="test-model")
        )

        simple_agent.set_llm_provider(mock_provider)

        await simple_agent._call_llm(
            "Test prompt",
            system_prompt="Custom system prompt",
        )

        call_kwargs = mock_provider.chat.call_args[1]
        assert call_kwargs["system_prompt"] == "Custom system prompt"

    @pytest.mark.asyncio
    async def test_call_llm_empty_response(self, simple_agent: SimpleAgent) -> None:
        """Test _call_llm handles empty response."""
        from src.llm.base import LLMResponse

        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(
            return_value=LLMResponse(content="", model="test-model")
        )

        simple_agent.set_llm_provider(mock_provider)

        result = await simple_agent._call_llm("Test prompt")
        assert result == ""


class TestBaseAgentResponseCreation:
    """Test BaseAgent response message creation."""

    def test_create_response(
        self,
        simple_agent: SimpleAgent,
        sample_message: Message,
    ) -> None:
        """Test creating a response message."""
        response = simple_agent._create_response(
            original_message=sample_message,
            content={"result": "Test result"},
        )

        assert response.sender_id == "test_agent_001"
        assert response.recipient_id == "orchestrator"
        assert response.message_type == MessageType.RESPONSE
        assert response.content == {"result": "Test result"}
        assert response.correlation_id == sample_message.correlation_id

    def test_create_error_response(
        self,
        simple_agent: SimpleAgent,
        sample_message: Message,
    ) -> None:
        """Test creating an error response message."""
        response = simple_agent._create_error_response(
            original_message=sample_message,
            error_message="Something went wrong",
            error_code="TEST_ERROR",
        )

        assert response.sender_id == "test_agent_001"
        assert response.recipient_id == "orchestrator"
        assert response.message_type == MessageType.ERROR
        assert response.content["error"] == "Something went wrong"
        assert response.content["error_code"] == "TEST_ERROR"
        assert response.correlation_id == sample_message.correlation_id
