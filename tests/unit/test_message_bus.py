"""Message Bus unit tests."""

import asyncio

import pytest

from src.core.message_bus import (
    InMemoryMessageBus,
    SubscriptionError,
)
from src.models import Message, MessageType


class TestInMemoryMessageBus:
    """Test InMemoryMessageBus class."""

    @pytest.fixture
    def bus(self):
        """Create a fresh message bus for each test."""
        return InMemoryMessageBus(max_history=100)

    @pytest.mark.asyncio
    async def test_subscribe(self, bus):
        """Test subscribing to the message bus."""
        received = []

        async def handler(msg):
            received.append(msg)

        await bus.subscribe("agent_001", handler)

        assert bus.subscriber_count == 1
        assert bus.is_subscribed("agent_001")

    @pytest.mark.asyncio
    async def test_subscribe_duplicate(self, bus):
        """Test subscribing same agent twice raises error."""

        async def handler(msg):
            pass

        await bus.subscribe("agent_001", handler)

        with pytest.raises(SubscriptionError):
            await bus.subscribe("agent_001", handler)

    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus):
        """Test unsubscribing from the message bus."""

        async def handler(msg):
            pass

        await bus.subscribe("agent_001", handler)
        await bus.unsubscribe("agent_001")

        assert bus.subscriber_count == 0
        assert not bus.is_subscribed("agent_001")

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self, bus):
        """Test unsubscribing nonexistent agent doesn't raise error."""
        await bus.unsubscribe("nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_publish_direct_message(self, bus):
        """Test publishing a direct message to specific recipient."""
        received = []

        async def handler(msg):
            received.append(msg)

        await bus.subscribe("recipient", handler)

        message = Message(
            sender_id="sender",
            recipient_id="recipient",
            message_type=MessageType.TASK,
            content={"test": "data"},
        )

        await bus.publish(message)
        await asyncio.sleep(0.01)  # Allow async delivery

        assert len(received) == 1
        assert received[0].content["test"] == "data"

    @pytest.mark.asyncio
    async def test_publish_broadcast(self, bus):
        """Test broadcasting a message to all subscribers."""
        received_a = []
        received_b = []

        async def handler_a(msg):
            received_a.append(msg)

        async def handler_b(msg):
            received_b.append(msg)

        await bus.subscribe("agent_a", handler_a)
        await bus.subscribe("agent_b", handler_b)

        message = Message(
            sender_id="broadcaster",
            recipient_id=None,  # Broadcast
            message_type=MessageType.NOTIFICATION,
            content={"notice": "hello"},
        )

        await bus.publish(message)
        await asyncio.sleep(0.01)

        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_broadcast_excludes_sender(self, bus):
        """Test that broadcast doesn't send to the sender."""
        received = []

        async def handler(msg):
            received.append(msg)

        await bus.subscribe("sender", handler)

        message = Message(
            sender_id="sender",
            recipient_id=None,
            message_type=MessageType.NOTIFICATION,
            content={},
        )

        await bus.publish(message)
        await asyncio.sleep(0.01)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_message_history(self, bus):
        """Test message history storage."""
        message1 = Message(
            sender_id="a",
            recipient_id="b",
            message_type=MessageType.TASK,
            content={"msg": 1},
        )
        message2 = Message(
            sender_id="b",
            recipient_id="a",
            message_type=MessageType.RESPONSE,
            content={"msg": 2},
        )

        await bus.publish(message1)
        await bus.publish(message2)

        history = await bus.get_history()

        assert len(history) == 2
        assert bus.message_count == 2

    @pytest.mark.asyncio
    async def test_history_by_correlation_id(self, bus):
        """Test filtering history by correlation ID."""
        correlation_id = "conv_001"

        msg1 = Message(
            sender_id="a",
            recipient_id="b",
            message_type=MessageType.TASK,
            content={},
            correlation_id=correlation_id,
        )
        msg2 = Message(
            sender_id="a",
            recipient_id="c",
            message_type=MessageType.TASK,
            content={},
            correlation_id="other",
        )
        msg3 = Message(
            sender_id="b",
            recipient_id="a",
            message_type=MessageType.RESPONSE,
            content={},
            correlation_id=correlation_id,
        )

        await bus.publish(msg1)
        await bus.publish(msg2)
        await bus.publish(msg3)

        filtered = await bus.get_history(correlation_id=correlation_id)

        assert len(filtered) == 2

    @pytest.mark.asyncio
    async def test_history_limit(self, bus):
        """Test history limit parameter."""
        for i in range(10):
            msg = Message(
                sender_id="a",
                recipient_id="b",
                message_type=MessageType.TASK,
                content={"i": i},
            )
            await bus.publish(msg)

        history = await bus.get_history(limit=5)

        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_max_history_trimming(self):
        """Test that history is trimmed when max is exceeded."""
        bus = InMemoryMessageBus(max_history=5)

        for i in range(10):
            msg = Message(
                sender_id="a",
                recipient_id="b",
                message_type=MessageType.TASK,
                content={"i": i},
            )
            await bus.publish(msg)

        assert bus.message_count == 5
        history = await bus.get_history()
        # Should have last 5 messages
        assert history[0].content["i"] == 9

    @pytest.mark.asyncio
    async def test_get_conversation_messages(self, bus):
        """Test getting all messages for a conversation."""
        conv_id = "conversation_001"

        for i in range(3):
            msg = Message(
                sender_id="a",
                recipient_id="b",
                message_type=MessageType.TASK,
                content={"i": i},
                correlation_id=conv_id,
            )
            await bus.publish(msg)

        messages = await bus.get_conversation_messages(conv_id)

        assert len(messages) == 3
        # Should be in chronological order
        assert messages[0].content["i"] == 0
        assert messages[2].content["i"] == 2

    @pytest.mark.asyncio
    async def test_clear_history(self, bus):
        """Test clearing message history."""
        for _ in range(5):
            msg = Message(
                sender_id="a",
                recipient_id="b",
                message_type=MessageType.TASK,
                content={},
            )
            await bus.publish(msg)

        assert bus.message_count == 5

        await bus.clear_history()

        assert bus.message_count == 0

    @pytest.mark.asyncio
    async def test_handler_error_doesnt_break_delivery(self, bus):
        """Test that handler errors don't prevent delivery to others."""
        received = []

        async def error_handler(msg):
            raise Exception("Handler error")

        async def good_handler(msg):
            received.append(msg)

        await bus.subscribe("error_agent", error_handler)
        await bus.subscribe("good_agent", good_handler)

        message = Message(
            sender_id="sender",
            recipient_id=None,  # Broadcast
            message_type=MessageType.NOTIFICATION,
            content={},
        )

        await bus.publish(message)
        await asyncio.sleep(0.01)

        # Good handler should still receive the message
        assert len(received) == 1
