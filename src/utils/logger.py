"""
Structured logging configuration using structlog.

Provides:
- Structured logging with JSON output (production) or console (development)
- Correlation ID context variables for automatic propagation across logs
- Utilities for setting/clearing correlation context
"""
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Optional

import structlog
from config import settings


# =============================================================================
# CORRELATION ID CONTEXT VARIABLES
# =============================================================================
# These context variables automatically propagate correlation IDs across
# async operations and are added to all log entries via the add_correlation_ids
# processor.

job_id_var: ContextVar[Optional[str]] = ContextVar("job_id", default=None)
task_id_var: ContextVar[Optional[str]] = ContextVar("task_id", default=None)
conversation_id_var: ContextVar[Optional[str]] = ContextVar("conversation_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def set_correlation_context(
    job_id: Optional[str] = None,
    task_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    """
    Set correlation IDs in context for automatic log propagation.

    Call this at the start of a request/job to ensure all downstream
    logs include these correlation IDs.

    Args:
        job_id: Unique job identifier
        task_id: Task identifier within a job
        conversation_id: Chat conversation identifier
        user_id: User identifier
        request_id: HTTP request identifier (auto-generated if not provided)
    """
    if job_id is not None:
        job_id_var.set(job_id)
    if task_id is not None:
        task_id_var.set(task_id)
    if conversation_id is not None:
        conversation_id_var.set(conversation_id)
    if user_id is not None:
        user_id_var.set(user_id)
    if request_id is not None:
        request_id_var.set(request_id)


def clear_correlation_context() -> None:
    """
    Clear all correlation context variables.

    Call this at the end of a request/job to prevent context leakage
    between requests.
    """
    job_id_var.set(None)
    task_id_var.set(None)
    conversation_id_var.set(None)
    user_id_var.set(None)
    request_id_var.set(None)


def add_correlation_ids(
    logger: Any,
    method_name: str,
    event_dict: dict,
) -> dict:
    """
    Structlog processor that adds correlation IDs to all log entries.

    This processor is added to the structlog pipeline and automatically
    injects any set correlation IDs into every log message.
    """
    if job_id_var.get():
        event_dict["job_id"] = job_id_var.get()
    if task_id_var.get():
        event_dict["task_id"] = task_id_var.get()
    if conversation_id_var.get():
        event_dict["conversation_id"] = conversation_id_var.get()
    if user_id_var.get():
        event_dict["user_id"] = user_id_var.get()
    if request_id_var.get():
        event_dict["request_id"] = request_id_var.get()
    return event_dict


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================


def configure_logging() -> None:
    """Configure structured logging."""

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        add_correlation_ids,  # Add correlation IDs to all logs
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.enable_structured_logging:
        # JSON output for production
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Human-readable output for development
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


# Initialize logging on import
configure_logging()
