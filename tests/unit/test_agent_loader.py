"""Unit tests for AgentLoader."""

import tempfile
from pathlib import Path

import pytest

from src.agents.base import BaseAgent, SimpleAgent
from src.agents.loader import (
    AgentConfigError,
    AgentLoader,
    AgentLoadError,
    get_agent_types,
    register_agent_type,
    validate_yaml_schema,
)


@pytest.fixture
def loader() -> AgentLoader:
    """Create an AgentLoader instance."""
    return AgentLoader()


@pytest.fixture
def valid_yaml_content() -> str:
    """Valid YAML configuration content."""
    return """
agent_id: "test_agent_001"
name: "Test Agent"
description: "A test agent"
capabilities:
  - name: "test_capability"
    description: "A test capability"
    input_schema:
      type: object
    output_schema:
      type: object
model: "claude-haiku-4-5-20251001"
max_tokens: 4096
temperature: 0.7
system_prompt: "You are a test agent."
"""


@pytest.fixture
def minimal_yaml_content() -> str:
    """Minimal valid YAML configuration."""
    return """
agent_id: "minimal_agent"
name: "Minimal Agent"
"""


class TestAgentLoaderLoadFromYaml:
    """Test AgentLoader.load_from_yaml method."""

    def test_load_valid_yaml(
        self,
        loader: AgentLoader,
        valid_yaml_content: str,
    ) -> None:
        """Test loading a valid YAML configuration."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(valid_yaml_content)
            f.flush()

            agent = loader.load_from_yaml(f.name)

            assert isinstance(agent, BaseAgent)
            assert agent.config.agent_id == "test_agent_001"
            assert agent.config.name == "Test Agent"
            assert len(agent.config.capabilities) == 1
            assert agent.config.capabilities[0].name == "test_capability"

    def test_load_minimal_yaml(
        self,
        loader: AgentLoader,
        minimal_yaml_content: str,
    ) -> None:
        """Test loading a minimal YAML configuration."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(minimal_yaml_content)
            f.flush()

            agent = loader.load_from_yaml(f.name)

            assert agent.config.agent_id == "minimal_agent"
            assert agent.config.name == "Minimal Agent"
            # Should have default values
            assert agent.config.model == "claude-haiku-4-5-20251001"
            assert agent.config.max_tokens == 4096

    def test_load_nonexistent_file(self, loader: AgentLoader) -> None:
        """Test loading a non-existent file raises AgentLoadError."""
        with pytest.raises(AgentLoadError, match="Configuration file not found"):
            loader.load_from_yaml("/nonexistent/path.yaml")

    def test_load_directory_instead_of_file(self, loader: AgentLoader) -> None:
        """Test loading a directory raises AgentLoadError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(AgentLoadError, match="Path is not a file"):
                loader.load_from_yaml(tmpdir)

    def test_load_invalid_yaml(self, loader: AgentLoader) -> None:
        """Test loading invalid YAML raises AgentLoadError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            with pytest.raises(AgentLoadError, match="Invalid YAML"):
                loader.load_from_yaml(f.name)

    def test_load_empty_yaml(self, loader: AgentLoader) -> None:
        """Test loading empty YAML raises AgentConfigError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            f.flush()

            with pytest.raises(AgentConfigError, match="Empty configuration file"):
                loader.load_from_yaml(f.name)


class TestAgentLoaderLoadAllFromDirectory:
    """Test AgentLoader.load_all_from_directory method."""

    def test_load_all_from_directory(self, loader: AgentLoader) -> None:
        """Test loading all agents from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple YAML files
            for i in range(3):
                with open(Path(tmpdir) / f"agent_{i}.yaml", "w", encoding="utf-8") as f:
                    f.write(
                        f"""
agent_id: "agent_{i}"
name: "Agent {i}"
"""
                    )

            agents = loader.load_all_from_directory(tmpdir)

            assert len(agents) == 3
            agent_ids = [a.config.agent_id for a in agents]
            assert "agent_0" in agent_ids
            assert "agent_1" in agent_ids
            assert "agent_2" in agent_ids

    def test_load_all_includes_yml_extension(self, loader: AgentLoader) -> None:
        """Test loading includes .yml extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .yaml file
            with open(Path(tmpdir) / "agent1.yaml", "w", encoding="utf-8") as f:
                f.write('agent_id: "agent1"\nname: "Agent 1"')

            # Create .yml file
            with open(Path(tmpdir) / "agent2.yml", "w", encoding="utf-8") as f:
                f.write('agent_id: "agent2"\nname: "Agent 2"')

            agents = loader.load_all_from_directory(tmpdir)

            assert len(agents) == 2
            agent_ids = [a.config.agent_id for a in agents]
            assert "agent1" in agent_ids
            assert "agent2" in agent_ids

    def test_load_all_nonexistent_directory(self, loader: AgentLoader) -> None:
        """Test loading from non-existent directory raises error."""
        with pytest.raises(AgentLoadError, match="Directory not found"):
            loader.load_all_from_directory("/nonexistent/directory")

    def test_load_all_file_instead_of_directory(self, loader: AgentLoader) -> None:
        """Test loading from a file raises error."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            with pytest.raises(AgentLoadError, match="Path is not a directory"):
                loader.load_all_from_directory(f.name)

    def test_load_all_empty_directory(self, loader: AgentLoader) -> None:
        """Test loading from empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents = loader.load_all_from_directory(tmpdir)
            assert agents == []

    def test_load_all_partial_failure(self, loader: AgentLoader) -> None:
        """Test loading continues even if some files fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid file
            with open(Path(tmpdir) / "valid.yaml", "w", encoding="utf-8") as f:
                f.write('agent_id: "valid"\nname: "Valid Agent"')

            # Create invalid file
            with open(Path(tmpdir) / "invalid.yaml", "w", encoding="utf-8") as f:
                f.write("invalid: [")

            agents = loader.load_all_from_directory(tmpdir)

            # Should load the valid one
            assert len(agents) == 1
            assert agents[0].config.agent_id == "valid"


class TestAgentLoaderCreateAgent:
    """Test AgentLoader.create_agent method."""

    def test_create_agent_simple_type(self, loader: AgentLoader) -> None:
        """Test creating an agent with simple type."""
        config_data = {
            "agent_id": "simple_001",
            "name": "Simple Agent",
            "agent_type": "simple",
        }

        agent = loader.create_agent(config_data)

        assert isinstance(agent, SimpleAgent)
        assert agent.config.agent_id == "simple_001"

    def test_create_agent_default_type(self, loader: AgentLoader) -> None:
        """Test creating an agent defaults to simple type."""
        config_data = {
            "agent_id": "default_001",
            "name": "Default Agent",
        }

        agent = loader.create_agent(config_data)

        assert isinstance(agent, SimpleAgent)

    def test_create_agent_unknown_type(self, loader: AgentLoader) -> None:
        """Test creating agent with unknown type raises error."""
        config_data = {
            "agent_id": "unknown_001",
            "name": "Unknown Agent",
            "agent_type": "unknown_type",
        }

        with pytest.raises(AgentConfigError, match="Unknown agent type"):
            loader.create_agent(config_data)

    def test_create_agent_custom_class(self, loader: AgentLoader) -> None:
        """Test creating agent with custom class."""
        config_data = {
            "agent_id": "custom_001",
            "name": "Custom Agent",
            "agent_class": "src.agents.base.SimpleAgent",
        }

        agent = loader.create_agent(config_data)

        assert isinstance(agent, SimpleAgent)
        assert agent.config.agent_id == "custom_001"

    def test_create_agent_invalid_custom_class(self, loader: AgentLoader) -> None:
        """Test creating agent with invalid custom class raises error."""
        config_data = {
            "agent_id": "invalid_001",
            "name": "Invalid Agent",
            "agent_class": "nonexistent.module.Agent",
        }

        with pytest.raises(AgentConfigError, match="Cannot import agent class"):
            loader.create_agent(config_data)

    def test_create_agent_with_capabilities(self, loader: AgentLoader) -> None:
        """Test creating agent with capabilities."""
        config_data = {
            "agent_id": "capable_001",
            "name": "Capable Agent",
            "capabilities": [
                {
                    "name": "cap1",
                    "description": "Capability 1",
                    "input_schema": {"type": "object"},
                },
                {
                    "name": "cap2",
                    "description": "Capability 2",
                },
            ],
        }

        agent = loader.create_agent(config_data)

        assert len(agent.config.capabilities) == 2
        assert agent.config.capabilities[0].name == "cap1"
        assert agent.config.capabilities[1].name == "cap2"


class TestAgentLoaderManagement:
    """Test AgentLoader agent management methods."""

    def test_get_loaded_agent(self, loader: AgentLoader) -> None:
        """Test getting a loaded agent by ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(Path(tmpdir) / "agent.yaml", "w", encoding="utf-8") as f:
                f.write('agent_id: "loaded_agent"\nname: "Loaded Agent"')

            loader.load_all_from_directory(tmpdir)

            agent = loader.get_loaded_agent("loaded_agent")
            assert agent is not None
            assert agent.config.agent_id == "loaded_agent"

    def test_get_loaded_agent_not_found(self, loader: AgentLoader) -> None:
        """Test getting a non-existent loaded agent returns None."""
        agent = loader.get_loaded_agent("nonexistent")
        assert agent is None

    def test_get_all_loaded_agents(self, loader: AgentLoader) -> None:
        """Test getting all loaded agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                with open(Path(tmpdir) / f"agent_{i}.yaml", "w", encoding="utf-8") as f:
                    f.write(f'agent_id: "agent_{i}"\nname: "Agent {i}"')

            loader.load_all_from_directory(tmpdir)

            all_agents = loader.get_all_loaded_agents()

            assert len(all_agents) == 2
            assert "agent_0" in all_agents
            assert "agent_1" in all_agents

    def test_clear(self, loader: AgentLoader) -> None:
        """Test clearing all loaded agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(Path(tmpdir) / "agent.yaml", "w", encoding="utf-8") as f:
                f.write('agent_id: "to_clear"\nname: "To Clear"')

            loader.load_all_from_directory(tmpdir)
            assert len(loader.get_all_loaded_agents()) == 1

            loader.clear()
            assert len(loader.get_all_loaded_agents()) == 0


class TestAgentTypeRegistry:
    """Test agent type registration functions."""

    def test_get_agent_types(self) -> None:
        """Test getting registered agent types."""
        types = get_agent_types()
        assert "simple" in types

    def test_register_agent_type(self) -> None:
        """Test registering a custom agent type."""

        # Create a mock agent class
        class CustomAgent(BaseAgent):
            async def process(self, message):
                pass

        register_agent_type("custom_test", CustomAgent)

        assert "custom_test" in get_agent_types()


class TestValidateYamlSchema:
    """Test YAML schema validation."""

    def test_valid_minimal_schema(self) -> None:
        """Test validating minimal valid schema."""
        config = {
            "agent_id": "test",
            "name": "Test Agent",
        }

        errors = validate_yaml_schema(config)
        assert errors == []

    def test_missing_agent_id(self) -> None:
        """Test validation catches missing agent_id."""
        config = {"name": "Test Agent"}

        errors = validate_yaml_schema(config)
        assert any("agent_id" in e for e in errors)

    def test_missing_name(self) -> None:
        """Test validation catches missing name."""
        config = {"agent_id": "test"}

        errors = validate_yaml_schema(config)
        assert any("name" in e for e in errors)

    def test_invalid_capabilities_not_list(self) -> None:
        """Test validation catches capabilities not being a list."""
        config = {
            "agent_id": "test",
            "name": "Test",
            "capabilities": "not a list",
        }

        errors = validate_yaml_schema(config)
        assert any("capabilities must be a list" in e for e in errors)

    def test_invalid_capability_missing_name(self) -> None:
        """Test validation catches capability missing name."""
        config = {
            "agent_id": "test",
            "name": "Test",
            "capabilities": [{"description": "No name"}],
        }

        errors = validate_yaml_schema(config)
        assert any("missing 'name' field" in e for e in errors)

    def test_invalid_capability_missing_description(self) -> None:
        """Test validation catches capability missing description."""
        config = {
            "agent_id": "test",
            "name": "Test",
            "capabilities": [{"name": "cap1"}],
        }

        errors = validate_yaml_schema(config)
        assert any("missing 'description' field" in e for e in errors)

    def test_invalid_max_tokens(self) -> None:
        """Test validation catches invalid max_tokens."""
        config = {
            "agent_id": "test",
            "name": "Test",
            "max_tokens": -1,
        }

        errors = validate_yaml_schema(config)
        assert any("max_tokens" in e for e in errors)

    def test_invalid_temperature(self) -> None:
        """Test validation catches invalid temperature."""
        config = {
            "agent_id": "test",
            "name": "Test",
            "temperature": 3.0,
        }

        errors = validate_yaml_schema(config)
        assert any("temperature" in e for e in errors)

    def test_valid_full_schema(self) -> None:
        """Test validating a full valid schema."""
        config = {
            "agent_id": "test",
            "name": "Test Agent",
            "description": "A test agent",
            "capabilities": [
                {"name": "cap1", "description": "Capability 1"},
                {"name": "cap2", "description": "Capability 2"},
            ],
            "max_tokens": 4096,
            "temperature": 0.7,
        }

        errors = validate_yaml_schema(config)
        assert errors == []
