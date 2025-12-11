"""Message Bus - Asynchronous communication channel between agents.

This module provides pub/sub messaging infrastructure for agent communication.
"""

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

from src.models import Message

# Type alias for message handlers
MessageHandler = Callable[[Message], Coroutine[Any, Any, None]]


class MessageBusError(Exception):
    """Base exception for message bus errors."""

    pass


class SubscriptionError(MessageBusError):
    """Raised when subscription operations fail."""

    pass


class PublishError(MessageBusError):
    """Raised when publish operations fail."""

    pass


class MessageBus(ABC):
    """Abstract base class for message bus implementations.

    Defines the interface for pub/sub messaging between agents.
    """

    @abstractmethod
    async def publish(self, message: Message) -> None:
        """Publish a message to the bus.

        Args:
            message: The message to publish.
        """
        pass

    @abstractmethod
    async def subscribe(self, agent_id: str, handler: MessageHandler) -> None:
        """Subscribe an agent to receive messages.

        Args:
            agent_id: The ID of the subscribing agent.
            handler: Async callback function to handle received messages.
        """
        pass

    @abstractmethod
    async def unsubscribe(self, agent_id: str) -> None:
        """Unsubscribe an agent from the bus.

        Args:
            agent_id: The ID of the agent to unsubscribe.
        """
        pass

    @abstractmethod
    async def get_history(
        self, correlation_id: str | None = None, limit: int = 100
    ) -> list[Message]:
        """Get message history.

        Args:
            correlation_id: Optional filter by correlation ID.
            limit: Maximum number of messages to return.

        Returns:
            List of messages matching the criteria.
        """
        pass


class InMemoryMessageBus(MessageBus):
    """In-memory implementation of MessageBus.

    Suitable for development and single-instance deployments.
    For production with multiple instances, use RedisMessageBus.
    """

    def __init__(self, max_history: int = 1000):
        """Initialize the message bus.

        Args:
            max_history: Maximum number of messages to keep in history.
        """
        self._subscribers: dict[str, MessageHandler] = {}
        self._history: list[Message] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()

        # Index for faster correlation_id lookups
        self._correlation_index: dict[str, list[Message]] = defaultdict(list)

    async def publish(self, message: Message) -> None:
        """Publish a message to subscribers.

        If recipient_id is set, only that agent receives the message.
        If recipient_id is None (broadcast), all subscribers receive it.

        Args:
            message: The message to publish.
        """
        async with self._lock:
            # Store in history
            self._history.append(message)
            self._correlation_index[message.correlation_id].append(message)

            # Trim history if needed
            if len(self._history) > self._max_history:
                removed = self._history.pop(0)
                # Clean up correlation index
                if removed.correlation_id in self._correlation_index:
                    corr_list = self._correlation_index[removed.correlation_id]
                    if removed in corr_list:
                        corr_list.remove(removed)
                    if not corr_list:
                        del self._correlation_index[removed.correlation_id]

        # Determine recipients
        if message.recipient_id:
            # Direct message
            recipients = [message.recipient_id]
        else:
            # Broadcast to all except sender
            recipients = [
                agent_id
                for agent_id in self._subscribers.keys()
                if agent_id != message.sender_id
            ]

        # Deliver messages (outside lock to prevent deadlocks)
        delivery_tasks = []
        for recipient_id in recipients:
            if recipient_id in self._subscribers:
                handler = self._subscribers[recipient_id]
                delivery_tasks.append(self._deliver(handler, message, recipient_id))

        if delivery_tasks:
            await asyncio.gather(*delivery_tasks, return_exceptions=True)

    async def _deliver(
        self, handler: MessageHandler, message: Message, recipient_id: str
    ) -> None:
        """Deliver a message to a handler with error handling.

        Args:
            handler: The message handler.
            message: The message to deliver.
            recipient_id: The recipient agent ID (for logging).
        """
        try:
            await handler(message)
        except Exception as e:
            # Log error but don't fail the publish
            # In production, this should use proper logging
            print(f"Error delivering message to {recipient_id}: {e}")

    async def subscribe(self, agent_id: str, handler: MessageHandler) -> None:
        """Subscribe an agent to receive messages.

        Args:
            agent_id: The ID of the subscribing agent.
            handler: Async callback function to handle received messages.

        Raises:
            SubscriptionError: If the agent is already subscribed.
        """
        async with self._lock:
            if agent_id in self._subscribers:
                raise SubscriptionError(f"Agent {agent_id} is already subscribed")
            self._subscribers[agent_id] = handler

    async def unsubscribe(self, agent_id: str) -> None:
        """Unsubscribe an agent from the bus.

        Args:
            agent_id: The ID of the agent to unsubscribe.
        """
        async with self._lock:
            if agent_id in self._subscribers:
                del self._subscribers[agent_id]

    async def get_history(
        self, correlation_id: str | None = None, limit: int = 100
    ) -> list[Message]:
        """Get message history.

        Args:
            correlation_id: Optional filter by correlation ID.
            limit: Maximum number of messages to return.

        Returns:
            List of messages matching the criteria, newest first.
        """
        async with self._lock:
            if correlation_id:
                messages = self._correlation_index.get(correlation_id, [])
            else:
                messages = self._history

            # Return newest first, limited
            return list(reversed(messages[-limit:]))

    async def get_conversation_messages(self, correlation_id: str) -> list[Message]:
        """Get all messages for a specific conversation.

        Args:
            correlation_id: The correlation ID of the conversation.

        Returns:
            List of messages in chronological order.
        """
        async with self._lock:
            return list(self._correlation_index.get(correlation_id, []))

    async def clear_history(self) -> None:
        """Clear all message history."""
        async with self._lock:
            self._history.clear()
            self._correlation_index.clear()

    @property
    def subscriber_count(self) -> int:
        """Return the number of subscribers."""
        return len(self._subscribers)

    @property
    def message_count(self) -> int:
        """Return the number of messages in history."""
        return len(self._history)

    def is_subscribed(self, agent_id: str) -> bool:
        """Check if an agent is subscribed.

        Args:
            agent_id: The ID of the agent to check.

        Returns:
            True if subscribed, False otherwise.
        """
        return agent_id in self._subscribers


# Future implementation placeholder
class RedisMessageBus(MessageBus):
    """Redis-based implementation of MessageBus.

    Suitable for production deployments with multiple instances.
    Provides message persistence and cross-instance communication.

    TODO: Implement in Phase 2 (Production Features)
    """

    def __init__(self, redis_url: str):
        raise NotImplementedError(
            "RedisMessageBus is not yet implemented. "
            "Use InMemoryMessageBus for development."
        )

    async def publish(self, message: Message) -> None:
        raise NotImplementedError

    async def subscribe(self, agent_id: str, handler: MessageHandler) -> None:
        raise NotImplementedError

    async def unsubscribe(self, agent_id: str) -> None:
        raise NotImplementedError

    async def get_history(
        self, correlation_id: str | None = None, limit: int = 100
    ) -> list[Message]:
        raise NotImplementedError
