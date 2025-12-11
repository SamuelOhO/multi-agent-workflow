"""Agent module - Base classes and implementations for agents.

This module provides the foundation for creating agents including:
- BaseAgent: Abstract base class for all agents
- SimpleAgent: Basic agent implementation
- AgentLoader: Dynamic agent loading from YAML configurations
- Specialized agent implementations (Researcher, Coder, Reviewer)
"""

from src.agents.base import (
    AgentError,
    BaseAgent,
    LLMError,
    SimpleAgent,
)
from src.agents.loader import (
    AgentConfigError,
    AgentLoader,
    AgentLoadError,
    get_agent_types,
    register_agent_type,
    validate_yaml_schema,
)

__all__ = [
    # Base
    "AgentError",
    "BaseAgent",
    "LLMError",
    "SimpleAgent",
    # Loader
    "AgentConfigError",
    "AgentLoader",
    "AgentLoadError",
    "get_agent_types",
    "register_agent_type",
    "validate_yaml_schema",
]
