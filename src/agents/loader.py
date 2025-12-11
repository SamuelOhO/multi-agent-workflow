"""Agent Loader - Dynamic agent loading from YAML configuration.

This module provides functionality to load agents from YAML configuration files,
enabling dynamic agent creation and hot-reloading.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.agents.base import BaseAgent, SimpleAgent
from src.models import AgentCapability, AgentConfig


class AgentLoadError(Exception):
    """Raised when agent loading fails."""

    def __init__(self, message: str, path: str | None = None):
        self.path = path
        super().__init__(f"{message}" + (f" (path: {path})" if path else ""))


class AgentConfigError(AgentLoadError):
    """Raised when agent configuration is invalid."""

    pass


# Registry of available agent implementations
_AGENT_IMPLEMENTATIONS: dict[str, type[BaseAgent]] = {
    "simple": SimpleAgent,
}


def register_agent_type(name: str, agent_class: type[BaseAgent]) -> None:
    """Register a new agent type.

    Args:
        name: The type name to register.
        agent_class: The agent class to register.
    """
    _AGENT_IMPLEMENTATIONS[name] = agent_class


def get_agent_types() -> list[str]:
    """Get list of registered agent types.

    Returns:
        List of registered agent type names.
    """
    return list(_AGENT_IMPLEMENTATIONS.keys())


class AgentLoader:
    """Loader for creating agents from YAML configuration files.

    Supports loading individual agent configurations or all agents
    from a directory. Also supports custom agent implementations.
    """

    def __init__(self) -> None:
        """Initialize the agent loader."""
        self._loaded_agents: dict[str, BaseAgent] = {}

    def load_from_yaml(self, path: str | Path) -> BaseAgent:
        """Load an agent from a YAML configuration file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            The loaded agent instance.

        Raises:
            AgentLoadError: If the file cannot be read.
            AgentConfigError: If the configuration is invalid.
        """
        path = Path(path)

        if not path.exists():
            raise AgentLoadError("Configuration file not found", str(path))

        if not path.is_file():
            raise AgentLoadError("Path is not a file", str(path))

        try:
            with open(path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise AgentLoadError(f"Invalid YAML: {e}", str(path)) from e
        except OSError as e:
            raise AgentLoadError(f"Cannot read file: {e}", str(path)) from e

        if not config_data:
            raise AgentConfigError("Empty configuration file", str(path))

        return self.create_agent(config_data, source_path=str(path))

    def load_all_from_directory(self, dir_path: str | Path) -> list[BaseAgent]:
        """Load all agents from YAML files in a directory.

        Args:
            dir_path: Path to the directory containing agent configurations.

        Returns:
            List of loaded agent instances.

        Raises:
            AgentLoadError: If the directory cannot be read.
        """
        dir_path = Path(dir_path)

        if not dir_path.exists():
            raise AgentLoadError("Directory not found", str(dir_path))

        if not dir_path.is_dir():
            raise AgentLoadError("Path is not a directory", str(dir_path))

        agents: list[BaseAgent] = []
        errors: list[str] = []

        # Load all YAML files in the directory
        for yaml_file in sorted(dir_path.glob("*.yaml")):
            try:
                agent = self.load_from_yaml(yaml_file)
                agents.append(agent)
                self._loaded_agents[agent.config.agent_id] = agent
            except AgentLoadError as e:
                errors.append(str(e))

        # Also check .yml extension
        for yml_file in sorted(dir_path.glob("*.yml")):
            try:
                agent = self.load_from_yaml(yml_file)
                agents.append(agent)
                self._loaded_agents[agent.config.agent_id] = agent
            except AgentLoadError as e:
                errors.append(str(e))

        if errors and not agents:
            raise AgentLoadError(
                f"Failed to load any agents. Errors: {'; '.join(errors)}",
                str(dir_path),
            )

        return agents

    def create_agent(
        self,
        config_data: dict[str, Any],
        source_path: str | None = None,
    ) -> BaseAgent:
        """Create an agent from configuration dictionary.

        Args:
            config_data: Configuration dictionary.
            source_path: Optional source path for error messages.

        Returns:
            The created agent instance.

        Raises:
            AgentConfigError: If the configuration is invalid.
        """
        try:
            # Parse capabilities
            capabilities = []
            for cap_data in config_data.get("capabilities", []):
                capabilities.append(AgentCapability(**cap_data))

            # Create config
            config = AgentConfig(
                agent_id=config_data.get("agent_id", ""),
                name=config_data.get("name", ""),
                description=config_data.get("description", ""),
                capabilities=capabilities,
                model=config_data.get("model", "claude-haiku-4-5-20251001"),
                max_tokens=config_data.get("max_tokens", 4096),
                temperature=config_data.get("temperature", 0.7),
                system_prompt=config_data.get("system_prompt"),
                metadata=config_data.get("metadata", {}),
            )

        except ValidationError as e:
            raise AgentConfigError(
                f"Invalid agent configuration: {e}", source_path
            ) from e
        except KeyError as e:
            raise AgentConfigError(f"Missing required field: {e}", source_path) from e

        # Determine agent type
        agent_type = config_data.get("agent_type", "simple")
        agent_class = config_data.get("agent_class")

        if agent_class:
            # Load custom agent class
            return self._create_custom_agent(agent_class, config, source_path)

        # Use registered agent type
        if agent_type not in _AGENT_IMPLEMENTATIONS:
            raise AgentConfigError(
                f"Unknown agent type: {agent_type}. "
                f"Available types: {list(_AGENT_IMPLEMENTATIONS.keys())}",
                source_path,
            )

        agent_cls = _AGENT_IMPLEMENTATIONS[agent_type]
        return agent_cls(config)

    def _create_custom_agent(
        self,
        agent_class: str,
        config: AgentConfig,
        source_path: str | None = None,
    ) -> BaseAgent:
        """Create an agent from a custom class.

        Args:
            agent_class: Fully qualified class name (e.g., "src.agents.implementations.researcher.ResearchAgent")
            config: Agent configuration.
            source_path: Optional source path for error messages.

        Returns:
            The created agent instance.

        Raises:
            AgentConfigError: If the class cannot be loaded.
        """
        try:
            # Split module and class name
            module_path, class_name = agent_class.rsplit(".", 1)

            # Import module
            module = importlib.import_module(module_path)

            # Get class
            agent_cls: type[BaseAgent] = getattr(module, class_name)

            # Verify it's a BaseAgent subclass
            if not issubclass(agent_cls, BaseAgent):
                raise AgentConfigError(
                    f"Class {agent_class} is not a BaseAgent subclass",
                    source_path,
                )

            return agent_cls(config)

        except ImportError as e:
            raise AgentConfigError(
                f"Cannot import agent class '{agent_class}': {e}",
                source_path,
            ) from e
        except AttributeError as e:
            raise AgentConfigError(
                f"Class not found in module: {e}",
                source_path,
            ) from e

    def get_loaded_agent(self, agent_id: str) -> BaseAgent | None:
        """Get a previously loaded agent by ID.

        Args:
            agent_id: The agent ID.

        Returns:
            The agent if found, None otherwise.
        """
        return self._loaded_agents.get(agent_id)

    def get_all_loaded_agents(self) -> dict[str, BaseAgent]:
        """Get all loaded agents.

        Returns:
            Dictionary mapping agent IDs to agent instances.
        """
        return dict(self._loaded_agents)

    def clear(self) -> None:
        """Clear all loaded agents."""
        self._loaded_agents.clear()


def validate_yaml_schema(config_data: dict[str, Any]) -> list[str]:
    """Validate YAML configuration schema.

    Args:
        config_data: Configuration dictionary to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    # Required fields
    required_fields = ["agent_id", "name"]
    for field in required_fields:
        if field not in config_data:
            errors.append(f"Missing required field: {field}")

    # Validate agent_id format
    agent_id = config_data.get("agent_id", "")
    if agent_id and not isinstance(agent_id, str):
        errors.append("agent_id must be a string")

    # Validate capabilities
    capabilities = config_data.get("capabilities", [])
    if not isinstance(capabilities, list):
        errors.append("capabilities must be a list")
    else:
        for i, cap in enumerate(capabilities):
            if not isinstance(cap, dict):
                errors.append(f"capabilities[{i}] must be a dictionary")
            else:
                if "name" not in cap:
                    errors.append(f"capabilities[{i}] missing 'name' field")
                if "description" not in cap:
                    errors.append(f"capabilities[{i}] missing 'description' field")

    # Validate numeric fields
    if "max_tokens" in config_data:
        max_tokens = config_data["max_tokens"]
        if not isinstance(max_tokens, int) or max_tokens < 1:
            errors.append("max_tokens must be a positive integer")

    if "temperature" in config_data:
        temperature = config_data["temperature"]
        if (
            not isinstance(temperature, (int, float))
            or temperature < 0
            or temperature > 2
        ):
            errors.append("temperature must be a number between 0 and 2")

    return errors
