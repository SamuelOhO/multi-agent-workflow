"""Agent Orchestrator - Main Application Entry Point.

This module creates and configures the FastAPI application with all necessary
middleware, routers, and startup/shutdown handlers.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.agents.loader import AgentLoader
from src.api.routes import api_router, init_dependencies
from src.core.conversation import ConversationManager
from src.core.message_bus import InMemoryMessageBus
from src.core.orchestrator import Orchestrator
from src.core.registry import AgentRegistry
from src.utils.config import AppConfig, Environment, LogFormat, get_config, init_config
from src.utils.error_handlers import register_error_handlers
from src.utils.logging import (
    clear_correlation_id,
    get_logger,
    set_correlation_id,
    setup_logging,
)

# Global instances
_registry: AgentRegistry | None = None
_message_bus: InMemoryMessageBus | None = None
_conversation_manager: ConversationManager | None = None
_orchestrator: Orchestrator | None = None
_agent_loader: AgentLoader | None = None

logger = get_logger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_config_path() -> Path:
    """Get the path to the app.yaml configuration file."""
    return get_project_root() / "configs" / "app.yaml"


def get_agents_config_path() -> Path:
    """Get the path to the agents configuration directory."""
    return get_project_root() / "configs" / "agents"


async def load_agents_from_config(registry: AgentRegistry) -> int:
    """Load agents from YAML configuration files.

    Args:
        registry: The agent registry to register agents to.

    Returns:
        Number of agents loaded.
    """
    global _agent_loader
    _agent_loader = AgentLoader()

    agents_dir = get_agents_config_path()
    loaded_count = 0

    if agents_dir.exists() and agents_dir.is_dir():
        try:
            agents = _agent_loader.load_all_from_directory(agents_dir)
            for agent in agents:
                await registry.register(agent)
                logger.info(
                    "Agent loaded and registered",
                    agent_id=agent.config.agent_id,
                    agent_name=agent.config.name,
                )
                loaded_count += 1
        except Exception as e:
            logger.warning(
                "Failed to load agents from directory",
                path=str(agents_dir),
                error=str(e),
            )
    else:
        logger.info(
            "Agents configuration directory not found, skipping auto-load",
            path=str(agents_dir),
        )

    return loaded_count


async def startup_event(config: AppConfig) -> None:
    """Initialize application components on startup."""
    global _registry, _message_bus, _conversation_manager, _orchestrator

    logger.info(
        "Starting Agent Orchestrator",
        app_name=config.app.name,
        version=config.app.version,
        environment=config.app.env.value,
    )

    # Initialize core components
    _registry = AgentRegistry()
    _message_bus = InMemoryMessageBus()
    _conversation_manager = ConversationManager()
    _orchestrator = Orchestrator(
        registry=_registry,
        message_bus=_message_bus,
        conversation_manager=_conversation_manager,
    )

    # Initialize API dependencies
    init_dependencies(
        registry=_registry,
        conversation_manager=_conversation_manager,
        orchestrator=_orchestrator,
    )

    # Load agents from configuration
    loaded_count = await load_agents_from_config(_registry)
    logger.info("Agents loaded from configuration", count=loaded_count)

    logger.info(
        "Agent Orchestrator started successfully",
        host=config.app.host,
        port=config.app.port,
    )


async def shutdown_event() -> None:
    """Cleanup on application shutdown."""
    global _registry, _message_bus, _conversation_manager, _orchestrator, _agent_loader

    logger.info("Shutting down Agent Orchestrator")

    # Cleanup orchestrator (cancel running tasks)
    if _orchestrator:
        logger.info("Cleaning up orchestrator")
        # Note: Additional cleanup could be added here if needed

    # Cleanup message bus
    if _message_bus:
        logger.info("Cleaning up message bus")
        await _message_bus.clear_history()

    # Cleanup registry
    if _registry:
        agent_count = len(_registry)
        logger.info("Cleaning up agent registry", agents=agent_count)
        # Unregister all agents
        for agent_id in list(_registry._agents.keys()):
            try:
                await _registry.unregister(agent_id)
            except Exception:
                pass

    # Clear agent loader
    if _agent_loader:
        _agent_loader.clear()

    # Reset globals
    _registry = None
    _message_bus = None
    _conversation_manager = None
    _orchestrator = None
    _agent_loader = None

    logger.info("Agent Orchestrator shutdown complete")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager.

    Handles startup and shutdown events.
    """
    # Get config from app state
    config = app.state.config if hasattr(app.state, "config") else AppConfig()

    # Startup
    await startup_event(config)

    yield

    # Shutdown
    await shutdown_event()


def create_app(
    config_path: str | Path | None = None,
    env_file: str | Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config_path: Optional path to YAML configuration file.
        env_file: Optional path to .env file.

    Returns:
        Configured FastAPI application instance.
    """
    # Load configuration
    if config_path is None:
        default_config_path = get_config_path()
        if default_config_path.exists():
            config_path = default_config_path

    config = init_config(yaml_path=config_path, env_file=env_file)

    # Setup logging
    setup_logging(
        level=config.logging.level,
        json_format=config.logging.format == LogFormat.JSON,
    )

    # Create FastAPI app
    app = FastAPI(
        title=config.app.name,
        description="Multi-Agent Orchestration System - Enables multiple AI agents to collaborate on complex tasks",
        version=config.app.version,
        docs_url="/docs" if config.app.debug else None,
        redoc_url="/redoc" if config.app.debug else None,
        openapi_url="/openapi.json" if config.app.debug else None,
        lifespan=lifespan,
    )

    # Store config in app state
    app.state.config = config

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if config.app.env == Environment.DEVELOPMENT else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next: Any) -> Any:
        """Add request ID to each request for tracing."""
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id
        set_correlation_id(request_id)

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        clear_correlation_id()

        return response

    # Add logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Any) -> Any:
        """Log incoming requests and responses."""
        logger.info(
            "Request received",
            method=request.method,
            path=str(request.url.path),
            client=request.client.host if request.client else "unknown",
        )

        response = await call_next(request)

        logger.info(
            "Response sent",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
        )

        return response

    # Register error handlers
    register_error_handlers(app)

    # Include API routers
    app.include_router(api_router)

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> dict[str, Any]:
        """Root endpoint with basic information."""
        return {
            "name": config.app.name,
            "version": config.app.version,
            "status": "running",
            "docs": "/docs" if config.app.debug else "disabled",
        }

    # Readiness probe
    @app.get("/ready", tags=["Health"])
    async def readiness() -> JSONResponse:
        """Kubernetes readiness probe."""
        global _registry, _orchestrator

        if _registry is None or _orchestrator is None:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "message": "Service not initialized"},
            )

        return JSONResponse(
            status_code=200,
            content={"status": "ready"},
        )

    # Liveness probe
    @app.get("/live", tags=["Health"])
    async def liveness() -> JSONResponse:
        """Kubernetes liveness probe."""
        return JSONResponse(
            status_code=200,
            content={"status": "alive"},
        )

    return app


# Create the application instance
app = create_app()


def run_dev_server() -> None:
    """Run the development server with hot-reload."""
    import uvicorn

    config = get_config()

    uvicorn.run(
        "src.main:app",
        host=config.app.host,
        port=config.app.port,
        reload=True,
        reload_dirs=["src"],
        log_level="info",
    )


def run_prod_server() -> None:
    """Run the production server."""
    import uvicorn

    config = get_config()

    uvicorn.run(
        "src.main:app",
        host=config.app.host,
        port=config.app.port,
        reload=False,
        workers=4,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    run_dev_server()
