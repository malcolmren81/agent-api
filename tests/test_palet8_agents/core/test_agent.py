"""
Unit tests for palet8_agents.core.agent module.
"""

import pytest
from datetime import datetime

from palet8_agents.core.agent import (
    AgentState,
    AgentContext,
    AgentResult,
    BaseAgent,
)


class TestAgentState:
    """Tests for AgentState enum."""

    def test_state_values(self):
        """Test all expected state values exist."""
        assert AgentState.IDLE.value == "idle"
        assert AgentState.RUNNING.value == "running"
        assert AgentState.COMPLETED.value == "completed"
        assert AgentState.FAILED.value == "failed"
        assert AgentState.PAUSED.value == "paused"
        assert AgentState.WAITING_FOR_INPUT.value == "waiting_for_input"


class TestAgentContext:
    """Tests for AgentContext dataclass."""

    def test_context_creation(self):
        """Test basic context creation."""
        context = AgentContext(
            user_id="user123",
            job_id="job456",
        )
        assert context.user_id == "user123"
        assert context.job_id == "job456"
        assert context.credit_balance == 0.0
        assert context.credits_used == 0.0
        assert context.safety_score == 1.0

    def test_context_with_optional_fields(self):
        """Test context with all optional fields."""
        context = AgentContext(
            user_id="user123",
            job_id="job456",
            conversation_id="conv789",
            credit_balance=100.0,
            reasoning_model="gpt-4",
            image_model="dall-e-3",
        )
        assert context.conversation_id == "conv789"
        assert context.credit_balance == 100.0

    def test_deduct_credits_success(self):
        """Test successful credit deduction."""
        context = AgentContext(
            user_id="user123",
            job_id="job456",
            credit_balance=100.0,
        )
        result = context.deduct_credits(30.0)
        assert result is True
        assert context.credit_balance == 70.0
        assert context.credits_used == 30.0

    def test_deduct_credits_insufficient(self):
        """Test credit deduction with insufficient balance."""
        context = AgentContext(
            user_id="user123",
            job_id="job456",
            credit_balance=10.0,
        )
        result = context.deduct_credits(50.0)
        assert result is False
        assert context.credit_balance == 10.0
        assert context.credits_used == 0.0

    def test_add_safety_flag(self):
        """Test adding safety flags."""
        context = AgentContext(
            user_id="user123",
            job_id="job456",
        )
        context.add_safety_flag("nsfw", penalty=0.2)
        assert "nsfw" in context.safety_flags
        assert context.safety_score == 0.8

    def test_add_duplicate_safety_flag(self):
        """Test adding duplicate safety flag doesn't double penalty."""
        context = AgentContext(
            user_id="user123",
            job_id="job456",
        )
        context.add_safety_flag("nsfw", penalty=0.2)
        context.add_safety_flag("nsfw", penalty=0.2)
        assert len(context.safety_flags) == 1
        assert context.safety_score == 0.8

    def test_to_dict(self):
        """Test context serialization."""
        context = AgentContext(
            user_id="user123",
            job_id="job456",
        )
        data = context.to_dict()
        assert data["user_id"] == "user123"
        assert data["job_id"] == "job456"
        assert "created_at" in data


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_result_success(self):
        """Test successful result creation."""
        result = AgentResult(
            success=True,
            data={"output": "test"},
        )
        assert result.success is True
        assert result.data == {"output": "test"}
        assert result.error is None

    def test_result_failure(self):
        """Test failure result creation."""
        result = AgentResult(
            success=False,
            data=None,
            error="Something went wrong",
            error_code="TEST_ERROR",
        )
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.error_code == "TEST_ERROR"

    def test_result_with_metrics(self):
        """Test result with performance metrics."""
        result = AgentResult(
            success=True,
            data={"output": "test"},
            duration_ms=150,
            tokens_used=500,
            cost_usd=0.002,
        )
        assert result.duration_ms == 150
        assert result.tokens_used == 500
        assert result.cost_usd == 0.002

    def test_to_dict(self):
        """Test result serialization."""
        result = AgentResult(
            success=True,
            data={"test": "value"},
            agent_name="test_agent",
        )
        data = result.to_dict()
        assert data["success"] is True
        assert data["agent_name"] == "test_agent"


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""

    def test_cannot_instantiate_base_agent(self):
        """Test that BaseAgent cannot be directly instantiated."""
        with pytest.raises(TypeError):
            BaseAgent(name="test", description="test agent")

    def test_concrete_agent_implementation(self):
        """Test that a concrete implementation can be created."""

        class TestAgent(BaseAgent):
            async def run(self, context, user_input=None):
                return AgentResult(success=True, data="test")

        agent = TestAgent(name="test", description="A test agent")
        assert agent.name == "test"
        assert agent.description == "A test agent"
        assert agent.state == AgentState.IDLE

    @pytest.mark.asyncio
    async def test_agent_execution_tracking(self):
        """Test that execution timing is tracked."""

        class TestAgent(BaseAgent):
            async def run(self, context, user_input=None):
                self._start_execution()
                result = self._create_result(
                    success=True,
                    data="test",
                )
                return result

        agent = TestAgent(name="test", description="A test agent")
        context = AgentContext(user_id="user", job_id="job")
        result = await agent.run(context)

        assert result.success is True
        assert result.duration_ms >= 0
        assert agent.state == AgentState.COMPLETED

    def test_agent_repr(self):
        """Test agent string representation."""

        class TestAgent(BaseAgent):
            async def run(self, context, user_input=None):
                return AgentResult(success=True, data="test")

        agent = TestAgent(name="test", description="A test agent")
        assert "TestAgent" in repr(agent)
        assert "test" in repr(agent)
