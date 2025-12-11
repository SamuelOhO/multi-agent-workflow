"""Conversation Manager unit tests."""

import pytest

from src.core.conversation import (
    ConversationManager,
    ConversationNotFoundError,
    ConversationStateError,
)
from src.models import (
    ConversationPattern,
    ConversationStage,
    ConversationStatus,
    Message,
    MessageType,
)


class TestConversationManager:
    """Test ConversationManager class."""

    @pytest.fixture
    def manager(self):
        """Create a fresh conversation manager for each test."""
        return ConversationManager()

    @pytest.fixture
    def sample_stages(self):
        """Create sample stages for testing."""
        return [
            ConversationStage(
                name="research",
                description="Research phase",
                agent_capability="web_search",
            ),
            ConversationStage(
                name="code",
                description="Coding phase",
                agent_capability="code_gen",
            ),
            ConversationStage(
                name="review",
                description="Review phase",
                agent_capability="code_review",
            ),
        ]

    @pytest.mark.asyncio
    async def test_create_conversation(self, manager, sample_stages):
        """Test creating a conversation."""
        conv = await manager.create(
            name="Test Conversation",
            description="A test conversation",
            pattern=ConversationPattern.SEQUENTIAL,
            stages=sample_stages,
            initial_input={"task": "build something"},
        )

        assert conv.name == "Test Conversation"
        assert conv.pattern == ConversationPattern.SEQUENTIAL
        assert len(conv.stages) == 3
        assert conv.status == ConversationStatus.PENDING
        assert len(manager) == 1

    @pytest.mark.asyncio
    async def test_create_conversation_defaults(self, manager):
        """Test creating conversation with default values."""
        conv = await manager.create()

        assert conv.name == ""
        assert conv.pattern == ConversationPattern.SEQUENTIAL
        assert conv.stages == []
        assert conv.timeout_seconds == 300

    @pytest.mark.asyncio
    async def test_create_from_config(self, manager):
        """Test creating conversation from config dict."""
        config = {
            "name": "Dev Pipeline",
            "description": "Development workflow",
            "pattern": "sequential",
            "stages": [
                {"name": "research", "agent_capability": "web_search"},
                {"name": "code", "agent_capability": "code_gen"},
            ],
            "timeout_seconds": 600,
        }

        conv = await manager.create_from_config(config, initial_input={"task": "test"})

        assert conv.name == "Dev Pipeline"
        assert conv.pattern == ConversationPattern.SEQUENTIAL
        assert len(conv.stages) == 2
        assert conv.timeout_seconds == 600

    @pytest.mark.asyncio
    async def test_get_conversation(self, manager):
        """Test getting a conversation by ID."""
        created = await manager.create(name="Test")

        retrieved = await manager.get(created.id)

        assert retrieved.id == created.id
        assert retrieved.name == "Test"

    @pytest.mark.asyncio
    async def test_get_nonexistent_conversation(self, manager):
        """Test getting nonexistent conversation raises error."""
        with pytest.raises(ConversationNotFoundError) as exc_info:
            await manager.get("nonexistent")

        assert exc_info.value.conversation_id == "nonexistent"

    @pytest.mark.asyncio
    async def test_add_message(self, manager):
        """Test adding messages to conversation."""
        conv = await manager.create(name="Test")
        message = Message(
            sender_id="agent",
            recipient_id="orchestrator",
            message_type=MessageType.RESPONSE,
            content={"result": "done"},
        )

        await manager.add_message(conv.id, message)
        messages = await manager.get_messages(conv.id)

        assert len(messages) == 1
        assert messages[0].content["result"] == "done"

    @pytest.mark.asyncio
    async def test_start_conversation(self, manager):
        """Test starting a conversation."""
        conv = await manager.create(name="Test")

        started = await manager.start(conv.id)

        assert started.status == ConversationStatus.IN_PROGRESS
        assert started.started_at is not None

    @pytest.mark.asyncio
    async def test_start_non_pending_conversation(self, manager):
        """Test starting non-pending conversation raises error."""
        conv = await manager.create(name="Test")
        await manager.start(conv.id)

        with pytest.raises(ConversationStateError) as exc_info:
            await manager.start(conv.id)

        assert exc_info.value.operation == "start"

    @pytest.mark.asyncio
    async def test_complete_conversation(self, manager):
        """Test completing a conversation."""
        conv = await manager.create(name="Test")
        await manager.start(conv.id)

        result = await manager.complete(conv.id, {"output": "success"})

        assert result.status == ConversationStatus.COMPLETED
        assert result.output["output"] == "success"

    @pytest.mark.asyncio
    async def test_fail_conversation(self, manager):
        """Test failing a conversation."""
        conv = await manager.create(name="Test")
        await manager.start(conv.id)

        result = await manager.fail(conv.id, "Something went wrong")

        assert result.status == ConversationStatus.FAILED
        assert result.error == "Something went wrong"

    @pytest.mark.asyncio
    async def test_cancel_conversation(self, manager):
        """Test cancelling a conversation."""
        conv = await manager.create(name="Test")
        await manager.start(conv.id)

        result = await manager.cancel(conv.id)

        assert result.status == ConversationStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_finished_conversation(self, manager):
        """Test cancelling finished conversation raises error."""
        conv = await manager.create(name="Test")
        await manager.start(conv.id)
        await manager.complete(conv.id)

        with pytest.raises(ConversationStateError):
            await manager.cancel(conv.id)

    @pytest.mark.asyncio
    async def test_update_stage(self, manager, sample_stages):
        """Test updating a stage."""
        conv = await manager.create(name="Test", stages=sample_stages)

        stage = await manager.update_stage(
            conv.id,
            "research",
            status=ConversationStatus.COMPLETED,
            output={"data": "found"},
        )

        assert stage.status == ConversationStatus.COMPLETED
        assert stage.output_data["data"] == "found"

    @pytest.mark.asyncio
    async def test_update_nonexistent_stage(self, manager, sample_stages):
        """Test updating nonexistent stage raises error."""
        conv = await manager.create(name="Test", stages=sample_stages)

        with pytest.raises(ValueError):
            await manager.update_stage(conv.id, "nonexistent")

    @pytest.mark.asyncio
    async def test_advance_stage(self, manager, sample_stages):
        """Test advancing to next stage."""
        conv = await manager.create(name="Test", stages=sample_stages)

        next_stage = await manager.advance_stage(conv.id)

        assert next_stage is not None
        assert next_stage.name == "code"

        # Advance again
        next_stage = await manager.advance_stage(conv.id)
        assert next_stage.name == "review"

        # Try to advance past end
        next_stage = await manager.advance_stage(conv.id)
        assert next_stage is None

    @pytest.mark.asyncio
    async def test_list_all(self, manager):
        """Test listing all conversations."""
        await manager.create(name="Conv1")
        await manager.create(name="Conv2")
        await manager.create(name="Conv3")

        all_convs = await manager.list_all()

        assert len(all_convs) == 3

    @pytest.mark.asyncio
    async def test_list_by_status(self, manager):
        """Test listing conversations by status."""
        conv1 = await manager.create(name="Conv1")
        conv2 = await manager.create(name="Conv2")
        await manager.create(name="Conv3")

        await manager.start(conv1.id)
        await manager.start(conv2.id)
        await manager.complete(conv2.id)

        active = await manager.list_all(status=ConversationStatus.IN_PROGRESS)
        completed = await manager.list_all(status=ConversationStatus.COMPLETED)

        assert len(active) == 1
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_list_active(self, manager):
        """Test listing active conversations."""
        conv1 = await manager.create(name="Conv1")
        await manager.create(name="Conv2")

        await manager.start(conv1.id)

        active = await manager.list_active()

        assert len(active) == 1
        assert active[0].name == "Conv1"

    @pytest.mark.asyncio
    async def test_delete_conversation(self, manager):
        """Test deleting a conversation."""
        conv = await manager.create(name="Test")

        result = await manager.delete(conv.id)

        assert result is True
        assert len(manager) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, manager):
        """Test deleting nonexistent conversation returns False."""
        result = await manager.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_finished(self, manager):
        """Test cleaning up old finished conversations."""
        # Create and complete several conversations
        for i in range(5):
            conv = await manager.create(name=f"Conv{i}")
            await manager.start(conv.id)
            await manager.complete(conv.id)

        # Also create some pending ones
        await manager.create(name="Pending1")
        await manager.create(name="Pending2")

        deleted = await manager.cleanup_finished(keep_count=2)

        assert deleted == 3
        assert len(manager) == 4  # 2 finished + 2 pending

    @pytest.mark.asyncio
    async def test_contains(self, manager):
        """Test __contains__ method."""
        conv = await manager.create(name="Test")

        assert conv.id in manager
        assert "nonexistent" not in manager

    @pytest.mark.asyncio
    async def test_len(self, manager):
        """Test __len__ method."""
        assert len(manager) == 0

        await manager.create(name="Conv1")
        assert len(manager) == 1

        await manager.create(name="Conv2")
        assert len(manager) == 2
