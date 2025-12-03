"""
Unit tests for palet8_agents.core.exceptions module.
"""

import pytest

from palet8_agents.core.exceptions import (
    AgentError,
    AgentConfigError,
    AgentExecutionError,
    LLMClientError,
    LLMTimeoutError,
    LLMResponseError,
    RateLimitError,
    ToolError,
    ToolNotFoundError,
    ToolValidationError,
    SafetyViolationError,
    InsufficientCreditsError,
    ContextNotReadyError,
    EvaluationFailedError,
    MaxRetriesExceededError,
    DatabaseError,
    VectorDBError,
)


class TestAgentError:
    """Tests for base AgentError."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = AgentError("Something went wrong")
        assert str(error) == "[AGENT_ERROR] Something went wrong"
        assert error.message == "Something went wrong"
        assert error.error_code == "AGENT_ERROR"

    def test_custom_error_code(self):
        """Test error with custom code."""
        error = AgentError("Test error", error_code="CUSTOM_ERROR")
        assert error.error_code == "CUSTOM_ERROR"

    def test_error_details(self):
        """Test error with details."""
        error = AgentError(
            "Test error",
            details={"key": "value"},
        )
        assert error.details == {"key": "value"}

    def test_to_dict(self):
        """Test error serialization."""
        error = AgentError(
            "Test error",
            error_code="TEST",
            details={"extra": "info"},
        )
        data = error.to_dict()
        assert data["error"] is True
        assert data["error_code"] == "TEST"
        assert data["message"] == "Test error"
        assert data["details"]["extra"] == "info"


class TestSpecificErrors:
    """Tests for specific error types."""

    def test_agent_config_error(self):
        """Test AgentConfigError."""
        error = AgentConfigError("Missing config key")
        assert error.error_code == "CONFIG_ERROR"
        assert error.http_status == 500

    def test_agent_execution_error(self):
        """Test AgentExecutionError."""
        error = AgentExecutionError("Execution failed")
        assert error.error_code == "EXECUTION_ERROR"

    def test_llm_client_error(self):
        """Test LLMClientError."""
        error = LLMClientError("API error")
        assert error.error_code == "LLM_ERROR"
        assert error.http_status == 502

    def test_llm_timeout_error(self):
        """Test LLMTimeoutError."""
        error = LLMTimeoutError("Request timed out")
        assert error.error_code == "LLM_TIMEOUT"
        assert error.http_status == 504

    def test_llm_response_error(self):
        """Test LLMResponseError."""
        error = LLMResponseError("Invalid response")
        assert error.error_code == "LLM_RESPONSE_ERROR"

    def test_rate_limit_error(self):
        """Test RateLimitError with retry_after."""
        error = RateLimitError("Rate limited", retry_after=60)
        assert error.error_code == "RATE_LIMIT_ERROR"
        assert error.http_status == 429
        assert error.retry_after == 60
        assert error.details["retry_after"] == 60

    def test_tool_error(self):
        """Test ToolError."""
        error = ToolError("Tool failed")
        assert error.error_code == "TOOL_ERROR"

    def test_tool_not_found_error(self):
        """Test ToolNotFoundError."""
        error = ToolNotFoundError("Tool not found")
        assert error.error_code == "TOOL_NOT_FOUND"
        assert error.http_status == 404

    def test_tool_validation_error(self):
        """Test ToolValidationError."""
        error = ToolValidationError("Invalid input")
        assert error.error_code == "TOOL_VALIDATION_ERROR"
        assert error.http_status == 400


class TestSafetyViolationError:
    """Tests for SafetyViolationError."""

    def test_basic_violation(self):
        """Test basic safety violation."""
        error = SafetyViolationError("Content blocked")
        assert error.error_code == "SAFETY_VIOLATION"
        assert error.http_status == 400

    def test_violation_with_details(self):
        """Test safety violation with violation details."""
        error = SafetyViolationError(
            "Content blocked",
            violations=["nsfw", "violence"],
            severity="blocked",
        )
        assert error.violations == ["nsfw", "violence"]
        assert error.severity == "blocked"
        assert "nsfw" in error.details["violations"]


class TestInsufficientCreditsError:
    """Tests for InsufficientCreditsError."""

    def test_credit_error(self):
        """Test insufficient credits error."""
        error = InsufficientCreditsError(
            "Not enough credits",
            required=100.0,
            available=50.0,
        )
        assert error.error_code == "INSUFFICIENT_CREDITS"
        assert error.http_status == 402
        assert error.required == 100.0
        assert error.available == 50.0
        assert error.details["required"] == 100.0


class TestContextNotReadyError:
    """Tests for ContextNotReadyError."""

    def test_context_error(self):
        """Test context not ready error."""
        error = ContextNotReadyError(
            "Missing information",
            missing_fields=["product_type", "style"],
        )
        assert error.error_code == "CONTEXT_NOT_READY"
        assert error.missing_fields == ["product_type", "style"]


class TestEvaluationFailedError:
    """Tests for EvaluationFailedError."""

    def test_evaluation_error(self):
        """Test evaluation failed error."""
        error = EvaluationFailedError(
            "Quality too low",
            score=0.3,
            threshold=0.5,
            feedback="Improve composition",
        )
        assert error.error_code == "EVALUATION_FAILED"
        assert error.score == 0.3
        assert error.threshold == 0.5
        assert error.details["feedback"] == "Improve composition"


class TestMaxRetriesExceededError:
    """Tests for MaxRetriesExceededError."""

    def test_retry_error(self):
        """Test max retries exceeded error."""
        error = MaxRetriesExceededError(
            "Too many retries",
            retries=3,
        )
        assert error.error_code == "MAX_RETRIES_EXCEEDED"
        assert error.retries == 3


class TestDatabaseErrors:
    """Tests for database-related errors."""

    def test_database_error(self):
        """Test DatabaseError."""
        error = DatabaseError("Connection failed")
        assert error.error_code == "DATABASE_ERROR"

    def test_vector_db_error(self):
        """Test VectorDBError."""
        error = VectorDBError("Embedding failed")
        assert error.error_code == "VECTOR_DB_ERROR"
        assert isinstance(error, DatabaseError)


class TestErrorInheritance:
    """Tests for error inheritance."""

    def test_all_inherit_from_agent_error(self):
        """Test that all errors inherit from AgentError."""
        errors = [
            AgentConfigError("test"),
            AgentExecutionError("test"),
            LLMClientError("test"),
            LLMTimeoutError("test"),
            RateLimitError("test"),
            ToolError("test"),
            SafetyViolationError("test"),
            InsufficientCreditsError("test"),
            DatabaseError("test"),
        ]
        for error in errors:
            assert isinstance(error, AgentError)
            assert isinstance(error, Exception)

    def test_llm_errors_inherit_from_llm_client_error(self):
        """Test LLM error hierarchy."""
        assert issubclass(LLMTimeoutError, LLMClientError)
        assert issubclass(LLMResponseError, LLMClientError)
        assert issubclass(RateLimitError, LLMClientError)

    def test_tool_errors_inherit_from_tool_error(self):
        """Test tool error hierarchy."""
        assert issubclass(ToolNotFoundError, ToolError)
        assert issubclass(ToolValidationError, ToolError)

    def test_vector_db_inherits_from_database_error(self):
        """Test database error hierarchy."""
        assert issubclass(VectorDBError, DatabaseError)
