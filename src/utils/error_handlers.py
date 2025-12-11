"""Global error handlers for FastAPI application.

This module provides exception handlers that convert exceptions to standardized
API responses with appropriate HTTP status codes.
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from .exceptions import (
    AgentOrchestratorError,
    APIError,
    ExternalServiceError,
)
from .logging import get_logger

logger = get_logger(__name__)


def create_error_response(
    status_code: int,
    error: str,
    message: str,
    details: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    """Create a standardized error response.

    Args:
        status_code: HTTP status code
        error: Error type/code
        message: Human-readable error message
        details: Optional additional error details
        request_id: Optional request ID for tracing

    Returns:
        JSONResponse with error information
    """
    content: dict[str, Any] = {
        "success": False,
        "error": {
            "code": error,
            "message": message,
        },
    }

    if details:
        content["error"]["details"] = details

    if request_id:
        content["metadata"] = {"request_id": request_id}

    return JSONResponse(status_code=status_code, content=content)


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle APIError exceptions.

    Args:
        request: FastAPI request
        exc: APIError exception

    Returns:
        Standardized JSON error response
    """
    request_id = getattr(request.state, "request_id", None)

    logger.warning(
        "API error occurred",
        error=exc.__class__.__name__,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
        request_id=request_id,
        path=str(request.url.path),
    )

    return create_error_response(
        status_code=exc.status_code,
        error=exc.__class__.__name__,
        message=exc.message,
        details=exc.details if exc.details else None,
        request_id=request_id,
    )


async def orchestrator_error_handler(
    request: Request, exc: AgentOrchestratorError
) -> JSONResponse:
    """Handle general AgentOrchestratorError exceptions.

    Args:
        request: FastAPI request
        exc: AgentOrchestratorError exception

    Returns:
        Standardized JSON error response
    """
    request_id = getattr(request.state, "request_id", None)

    logger.error(
        "Orchestrator error occurred",
        error=exc.__class__.__name__,
        message=exc.message,
        details=exc.details,
        request_id=request_id,
        path=str(request.url.path),
    )

    return create_error_response(
        status_code=500,
        error=exc.__class__.__name__,
        message=exc.message,
        details=exc.details if exc.details else None,
        request_id=request_id,
    )


async def external_service_error_handler(
    request: Request, exc: ExternalServiceError
) -> JSONResponse:
    """Handle ExternalServiceError exceptions.

    Args:
        request: FastAPI request
        exc: ExternalServiceError exception

    Returns:
        Standardized JSON error response
    """
    request_id = getattr(request.state, "request_id", None)

    logger.error(
        "External service error",
        error=exc.__class__.__name__,
        message=exc.message,
        details=exc.details,
        request_id=request_id,
        path=str(request.url.path),
    )

    return create_error_response(
        status_code=502,
        error=exc.__class__.__name__,
        message=exc.message,
        details=exc.details if exc.details else None,
        request_id=request_id,
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle FastAPI RequestValidationError exceptions.

    Args:
        request: FastAPI request
        exc: RequestValidationError exception

    Returns:
        Standardized JSON error response with validation details
    """
    request_id = getattr(request.state, "request_id", None)

    errors = []
    for error in exc.errors():
        loc = ".".join(str(x) for x in error["loc"])
        errors.append(
            {
                "field": loc,
                "message": error["msg"],
                "type": error["type"],
            }
        )

    logger.warning(
        "Request validation error",
        errors=errors,
        request_id=request_id,
        path=str(request.url.path),
    )

    return create_error_response(
        status_code=422,
        error="ValidationError",
        message="Request validation failed",
        details={"validation_errors": errors},
        request_id=request_id,
    )


async def pydantic_validation_error_handler(
    request: Request, exc: PydanticValidationError
) -> JSONResponse:
    """Handle Pydantic ValidationError exceptions.

    Args:
        request: FastAPI request
        exc: PydanticValidationError exception

    Returns:
        Standardized JSON error response with validation details
    """
    request_id = getattr(request.state, "request_id", None)

    errors = []
    for error in exc.errors():
        loc = ".".join(str(x) for x in error["loc"])
        errors.append(
            {
                "field": loc,
                "message": error["msg"],
                "type": error["type"],
            }
        )

    logger.warning(
        "Pydantic validation error",
        errors=errors,
        request_id=request_id,
        path=str(request.url.path),
    )

    return create_error_response(
        status_code=422,
        error="ValidationError",
        message="Data validation failed",
        details={"validation_errors": errors},
        request_id=request_id,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions.

    Args:
        request: FastAPI request
        exc: Any exception

    Returns:
        Generic 500 error response
    """
    request_id = getattr(request.state, "request_id", None)

    logger.exception(
        "Unexpected error occurred",
        error=exc.__class__.__name__,
        message=str(exc),
        request_id=request_id,
        path=str(request.url.path),
    )

    return create_error_response(
        status_code=500,
        error="InternalServerError",
        message="An unexpected error occurred",
        request_id=request_id,
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all error handlers with the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    # Register handlers from most specific to most general
    # Note: ignore comments needed because FastAPI's type hints are overly restrictive
    app.add_exception_handler(APIError, api_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ExternalServiceError, external_service_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(AgentOrchestratorError, orchestrator_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(PydanticValidationError, pydantic_validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Error handlers registered")
