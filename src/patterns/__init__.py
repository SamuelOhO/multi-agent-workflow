"""Conversation patterns for agent collaboration.

This module provides different patterns for how agents collaborate:
- Sequential: Agents work one after another
- Parallel: Agents work simultaneously
- Router: Task is routed to appropriate agent
- Debate: Agents discuss and reach consensus
- Hierarchical: Supervisor manages worker agents
"""

from .base import (
    AgentProtocol,
    BasePattern,
    NoAgentAvailableError,
    PatternError,
    PatternValidationError,
    StageExecutionError,
    create_pattern,
    get_pattern_class,
)
from .debate import DebatePattern, DebateRound
from .hierarchical import HierarchicalPattern
from .parallel import MergeStrategy, ParallelPattern
from .router import RouterPattern
from .sequential import SequentialPattern

__all__ = [
    # Base
    "BasePattern",
    "PatternError",
    "NoAgentAvailableError",
    "StageExecutionError",
    "PatternValidationError",
    "AgentProtocol",
    "get_pattern_class",
    "create_pattern",
    # Patterns
    "SequentialPattern",
    "ParallelPattern",
    "RouterPattern",
    "DebatePattern",
    "HierarchicalPattern",
    # Utilities
    "MergeStrategy",
    "DebateRound",
]
