"""
Request/response logging middleware with correlation ID tracking.

This middleware:
- Generates or extracts request IDs for correlation
- Logs all incoming requests and outgoing responses
- Measures request duration
- Sets correlation context for downstream logs
- Adds request ID to response headers for tracing
"""
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import (
    get_logger,
    set_correlation_context,
    clear_correlation_context,
)

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive API request/response logging.

    Features:
    - Automatic request ID generation (or extraction from X-Request-ID header)
    - Correlation ID extraction from headers (X-Job-ID, X-User-ID)
    - Request entry/exit logging with timing
    - Error logging with duration
    - Response header injection for tracing
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process each request with logging and correlation context."""
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Extract correlation IDs from headers (may be set by upstream services)
        job_id = request.headers.get("X-Job-ID")
        user_id = request.headers.get("X-User-ID")

        # Set correlation context for all downstream logs
        set_correlation_context(
            request_id=request_id,
            job_id=job_id,
            user_id=user_id,
        )

        start_time = time.time()

        # Log request entry
        logger.info(
            "api.request.start",
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params) if request.query_params else None,
            client_ip=request.client.host if request.client else None,
        )

        try:
            response = await call_next(request)
            duration_ms = int((time.time() - start_time) * 1000)

            # Log successful response
            logger.info(
                "api.request.complete",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

            # Add request ID to response headers for tracing
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log error with details
            logger.error(
                "api.request.error",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
                error_detail=str(e),
                error_type=type(e).__name__,
            )
            raise

        finally:
            # Clear correlation context to prevent leakage between requests
            clear_correlation_context()
