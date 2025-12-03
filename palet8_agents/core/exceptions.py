"""
Structured exceptions for the agent framework.

This module defines a hierarchy of exceptions with error codes
for consistent error handling across the agent system.
"""

from typing import Any, Dict, Optional


class AgentError(Exception):
    """
    Base exception for all agent-related errors.

    All custom exceptions in the agent framework inherit from this class,
    providing consistent error code and detail handling.
    """

    error_code: str = "AGENT_ERROR"
    http_status: int = 500

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the exception.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        if error_code:
            self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class AgentConfigError(AgentError):
    """
    Configuration-related errors.

    Raised when there's an issue with agent configuration,
    such as missing required settings or invalid values.
    """

    error_code = "CONFIG_ERROR"
    http_status = 500


class AgentExecutionError(AgentError):
    """
    Agent execution errors.

    Raised when an agent fails during execution,
    such as unexpected state or logic errors.
    """

    error_code = "EXECUTION_ERROR"
    http_status = 500


class LLMClientError(AgentError):
    """
    LLM API client errors.

    Base class for errors related to LLM API interactions.
    """

    error_code = "LLM_ERROR"
    http_status = 502


class LLMTimeoutError(LLMClientError):
    """Raised when LLM API call times out."""

    error_code = "LLM_TIMEOUT"
    http_status = 504


class LLMResponseError(LLMClientError):
    """Raised when LLM returns an invalid or unexpected response."""

    error_code = "LLM_RESPONSE_ERROR"
    http_status = 502


class RateLimitError(LLMClientError):
    """
    Rate limit exceeded error.

    Raised when the LLM provider's rate limit is exceeded.
    Includes retry-after information when available.
    """

    error_code = "RATE_LIMIT_ERROR"
    http_status = 429

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize rate limit error.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            **kwargs: Additional arguments for parent class
        """
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        if retry_after:
            self.details["retry_after"] = retry_after


class ToolError(AgentError):
    """
    Tool execution errors.

    Raised when a tool fails during execution.
    """

    error_code = "TOOL_ERROR"
    http_status = 500


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found in the registry."""

    error_code = "TOOL_NOT_FOUND"
    http_status = 404


class ToolValidationError(ToolError):
    """Raised when tool input validation fails."""

    error_code = "TOOL_VALIDATION_ERROR"
    http_status = 400


class SafetyViolationError(AgentError):
    """
    Safety policy violation error.

    Raised when content violates safety policies.
    Contains information about which policies were violated.
    """

    error_code = "SAFETY_VIOLATION"
    http_status = 400

    def __init__(
        self,
        message: str,
        violations: Optional[list] = None,
        severity: str = "blocked",
        **kwargs,
    ):
        """
        Initialize safety violation error.

        Args:
            message: Error message
            violations: List of violated policies
            severity: Severity level (warning, blocked)
            **kwargs: Additional arguments for parent class
        """
        super().__init__(message, **kwargs)
        self.violations = violations or []
        self.severity = severity
        self.details["violations"] = self.violations
        self.details["severity"] = self.severity


class InsufficientCreditsError(AgentError):
    """
    Insufficient credits error.

    Raised when user doesn't have enough credits for an operation.
    """

    error_code = "INSUFFICIENT_CREDITS"
    http_status = 402

    def __init__(
        self,
        message: str,
        required: float = 0.0,
        available: float = 0.0,
        **kwargs,
    ):
        """
        Initialize insufficient credits error.

        Args:
            message: Error message
            required: Credits required for operation
            available: Credits currently available
            **kwargs: Additional arguments for parent class
        """
        super().__init__(message, **kwargs)
        self.required = required
        self.available = available
        self.details["required"] = required
        self.details["available"] = available


class ContextNotReadyError(AgentError):
    """
    Context not ready error.

    Raised when trying to proceed without sufficient context,
    typically requiring more user input.
    """

    error_code = "CONTEXT_NOT_READY"
    http_status = 400

    def __init__(
        self,
        message: str,
        missing_fields: Optional[list] = None,
        **kwargs,
    ):
        """
        Initialize context not ready error.

        Args:
            message: Error message
            missing_fields: List of missing context fields
            **kwargs: Additional arguments for parent class
        """
        super().__init__(message, **kwargs)
        self.missing_fields = missing_fields or []
        self.details["missing_fields"] = self.missing_fields


class EvaluationFailedError(AgentError):
    """
    Evaluation failed error.

    Raised when generated content fails quality evaluation.
    """

    error_code = "EVALUATION_FAILED"
    http_status = 400

    def __init__(
        self,
        message: str,
        score: float = 0.0,
        threshold: float = 0.0,
        feedback: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize evaluation failed error.

        Args:
            message: Error message
            score: Achieved quality score
            threshold: Required threshold
            feedback: Feedback for improvement
            **kwargs: Additional arguments for parent class
        """
        super().__init__(message, **kwargs)
        self.score = score
        self.threshold = threshold
        self.feedback = feedback
        self.details["score"] = score
        self.details["threshold"] = threshold
        if feedback:
            self.details["feedback"] = feedback


class MaxRetriesExceededError(AgentError):
    """
    Max retries exceeded error.

    Raised when an operation exceeds the maximum retry count.
    """

    error_code = "MAX_RETRIES_EXCEEDED"
    http_status = 500

    def __init__(
        self,
        message: str,
        retries: int = 0,
        **kwargs,
    ):
        """
        Initialize max retries error.

        Args:
            message: Error message
            retries: Number of retries attempted
            **kwargs: Additional arguments for parent class
        """
        super().__init__(message, **kwargs)
        self.retries = retries
        self.details["retries"] = retries


class DatabaseError(AgentError):
    """
    Database operation error.

    Raised when a database operation fails.
    """

    error_code = "DATABASE_ERROR"
    http_status = 500


class VectorDBError(DatabaseError):
    """
    Vector database operation error.

    Raised when a vector database operation fails.
    """

    error_code = "VECTOR_DB_ERROR"
    http_status = 500
