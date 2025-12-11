"""Unit tests for main application module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


class TestCreateApp:
    """Tests for create_app function."""

    def test_create_app_default(self) -> None:
        """Test creating app with default configuration."""
        from src.main import create_app

        app = create_app()

        assert app is not None
        assert app.title == "Agent Orchestrator"
        assert hasattr(app.state, "config")

    def test_create_app_with_custom_config(self, tmp_path: Path) -> None:
        """Test creating app with custom config path."""
        from src.main import create_app

        # Create a minimal config file
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
app:
  name: "Test App"
  version: "0.0.1"
  env: testing
  debug: true
  host: "127.0.0.1"
  port: 9000

logging:
  level: DEBUG
  format: console
"""
        )

        app = create_app(config_path=config_file)

        assert app is not None
        assert app.title == "Test App"
        assert app.version == "0.0.1"

    def test_create_app_has_routers(self) -> None:
        """Test that app includes all expected routers."""
        from src.main import create_app

        app = create_app()

        # Check routes exist
        routes = [route.path for route in app.routes]

        # Root endpoints
        assert "/" in routes
        assert "/ready" in routes
        assert "/live" in routes

        # API v1 endpoints should be included
        assert any("/api/v1" in route for route in routes)


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self) -> None:
        """Test root endpoint returns app info."""
        from src.main import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_liveness_endpoint(self) -> None:
        """Test liveness probe endpoint."""
        from src.main import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_endpoint_before_startup(self) -> None:
        """Test readiness probe returns not ready before startup."""
        import src.main as main_module
        from src.main import create_app

        # Reset globals
        main_module._registry = None
        main_module._orchestrator = None

        app = create_app()

        # Note: TestClient triggers lifespan automatically, so this test
        # verifies the behavior when globals are reset after startup.
        # The /ready endpoint checks these globals at request time.
        with TestClient(app) as client:
            # After lifespan starts, globals are initialized
            # Reset them to simulate uninitialized state
            saved_registry = main_module._registry
            saved_orchestrator = main_module._orchestrator

            main_module._registry = None
            main_module._orchestrator = None

            response = client.get("/ready")
            assert response.status_code == 503

            # Restore for proper cleanup
            main_module._registry = saved_registry
            main_module._orchestrator = saved_orchestrator

    def test_readiness_endpoint_after_startup(self) -> None:
        """Test readiness probe returns ready after startup."""
        from src.main import create_app

        app = create_app()

        # Use context manager to ensure lifespan is properly managed
        with TestClient(app) as client:
            response = client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"


class TestMiddleware:
    """Tests for middleware functionality."""

    def test_request_id_header_added(self) -> None:
        """Test that X-Request-ID header is added to responses."""
        from src.main import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/")

        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0

    def test_request_id_header_preserved(self) -> None:
        """Test that provided X-Request-ID is preserved."""
        from src.main import create_app

        app = create_app()
        client = TestClient(app)

        custom_request_id = "test-request-123"
        response = client.get("/", headers={"X-Request-ID": custom_request_id})

        assert response.headers["X-Request-ID"] == custom_request_id


class TestCORSMiddleware:
    """Tests for CORS middleware."""

    def test_cors_headers_in_development(self) -> None:
        """Test CORS headers are present in development mode."""
        from src.main import create_app

        app = create_app()
        client = TestClient(app)

        # Preflight request
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # In development, CORS should allow all origins
        assert response.status_code == 200


class TestAgentLoading:
    """Tests for agent loading on startup."""

    @pytest.mark.asyncio
    async def test_load_agents_from_config_empty_dir(self, tmp_path: Path) -> None:
        """Test loading agents from empty directory."""
        from src.core.registry import AgentRegistry
        from src.main import load_agents_from_config

        registry = AgentRegistry()
        empty_dir = tmp_path / "empty_agents"
        empty_dir.mkdir()

        with patch("src.main.get_agents_config_path", return_value=empty_dir):
            count = await load_agents_from_config(registry)

        # Empty directory should load 0 agents (no error, just warning)
        assert count == 0

    @pytest.mark.asyncio
    async def test_load_agents_from_config_nonexistent_dir(self) -> None:
        """Test loading agents from nonexistent directory."""
        from src.core.registry import AgentRegistry
        from src.main import load_agents_from_config

        registry = AgentRegistry()
        nonexistent = Path("/nonexistent/path/to/agents")

        with patch("src.main.get_agents_config_path", return_value=nonexistent):
            count = await load_agents_from_config(registry)

        assert count == 0


class TestStartupShutdown:
    """Tests for startup and shutdown events."""

    @pytest.mark.asyncio
    async def test_startup_event(self) -> None:
        """Test startup event initializes components."""
        import src.main as main_module
        from src.main import startup_event
        from src.utils.config import AppConfig

        config = AppConfig()

        # Patch agent loading to avoid file system access
        with patch("src.main.load_agents_from_config", return_value=0):
            await startup_event(config)

        assert main_module._registry is not None
        assert main_module._message_bus is not None
        assert main_module._conversation_manager is not None
        assert main_module._orchestrator is not None

        # Cleanup
        await main_module.shutdown_event()

    @pytest.mark.asyncio
    async def test_shutdown_event(self) -> None:
        """Test shutdown event cleans up components."""
        import src.main as main_module
        from src.main import shutdown_event, startup_event
        from src.utils.config import AppConfig

        config = AppConfig()

        with patch("src.main.load_agents_from_config", return_value=0):
            await startup_event(config)

        # Verify components exist
        assert main_module._registry is not None

        # Shutdown
        await shutdown_event()

        # Verify components are cleaned up
        assert main_module._registry is None
        assert main_module._message_bus is None
        assert main_module._conversation_manager is None
        assert main_module._orchestrator is None


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_project_root(self) -> None:
        """Test project root path detection."""
        from src.main import get_project_root

        root = get_project_root()

        assert root.exists()
        assert (root / "src").exists()

    def test_get_config_path(self) -> None:
        """Test config path detection."""
        from src.main import get_config_path

        config_path = get_config_path()

        assert str(config_path).endswith("configs/app.yaml") or str(
            config_path
        ).endswith("configs\\app.yaml")

    def test_get_agents_config_path(self) -> None:
        """Test agents config path detection."""
        from src.main import get_agents_config_path

        agents_path = get_agents_config_path()

        assert str(agents_path).endswith("configs/agents") or str(agents_path).endswith(
            "configs\\agents"
        )


class TestAppIntegration:
    """Integration tests for the full application."""

    def test_full_app_lifecycle(self) -> None:
        """Test full application lifecycle with TestClient."""
        from src.main import create_app

        app = create_app()

        # TestClient handles lifespan automatically
        with TestClient(app) as client:
            # Root endpoint
            response = client.get("/")
            assert response.status_code == 200

            # Health check
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            # Liveness
            response = client.get("/live")
            assert response.status_code == 200

            # Readiness
            response = client.get("/ready")
            assert response.status_code == 200

    def test_api_endpoints_available(self) -> None:
        """Test that API endpoints are available."""
        from src.main import create_app

        app = create_app()

        with TestClient(app) as client:
            # Agent endpoints
            response = client.get("/api/v1/agents")
            assert response.status_code == 200

            # Task creation (should fail with missing data but endpoint exists)
            response = client.post("/api/v1/tasks", json={})
            assert (
                response.status_code == 422
            )  # Validation error (missing 'input' field)

            # Conversation creation (has all defaults, so empty JSON succeeds)
            response = client.post("/api/v1/conversations", json={})
            assert response.status_code == 201  # Created with defaults

            # Agent registration requires data
            response = client.post("/api/v1/agents", json={})
            assert response.status_code == 422  # Validation error
