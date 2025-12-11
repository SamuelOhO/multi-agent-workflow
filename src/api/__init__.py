"""API module.

Provides FastAPI routers, schemas, and dependencies.
"""

from .routes import (
    agent_router,
    api_router,
    conversation_router,
    init_dependencies,
    task_router,
)
from .schemas import (
    AddMessageRequest,
    AgentCapabilitySchema,
    AgentHealthResponse,
    AgentResponse,
    APIResponse,
    ConversationResponse,
    CreateConversationRequest,
    CreateTaskRequest,
    ErrorResponse,
    MessageResponse,
    PaginatedResponse,
    RegisterAgentRequest,
    TaskResultResponse,
    TaskStatusResponse,
)

__all__ = [
    # Routers
    "api_router",
    "agent_router",
    "task_router",
    "conversation_router",
    # Functions
    "init_dependencies",
    # Schemas - Common
    "APIResponse",
    "ErrorResponse",
    "PaginatedResponse",
    # Schemas - Agent
    "AgentCapabilitySchema",
    "RegisterAgentRequest",
    "AgentResponse",
    "AgentHealthResponse",
    # Schemas - Task
    "CreateTaskRequest",
    "TaskStatusResponse",
    "TaskResultResponse",
    # Schemas - Conversation
    "CreateConversationRequest",
    "ConversationResponse",
    "MessageResponse",
    "AddMessageRequest",
]
