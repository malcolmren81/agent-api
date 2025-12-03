"""Tests for palet8_agents.agents.evaluator_agent_v2 module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import directly to avoid Prisma dependency issues
import sys
import importlib.util


# Load modules directly to avoid import chain issues
def load_module(name, path):
    """Load a Python module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load required modules
models_module = load_module('palet8_agents.models', 'palet8_agents/models/__init__.py')
PromptQualityResult = models_module.PromptQualityResult
ResultQualityResult = models_module.ResultQualityResult
EvaluationPlan = models_module.EvaluationPlan
EvaluationFeedback = models_module.EvaluationFeedback
RetrySuggestion = models_module.RetrySuggestion

tools_base_module = load_module('palet8_agents.tools.base', 'palet8_agents/tools/base.py')
ToolResult = tools_base_module.ToolResult


# Mock AgentContext for testing
@dataclass
class AgentContext:
    """Mock AgentContext for testing."""
    job_id: str = ""
    user_id: str = ""
    plan: Optional[Dict[str, Any]] = None
    requirements: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.plan is None:
            self.plan = {}
        if self.requirements is None:
            self.requirements = {}


# Load evaluator agent module
evaluator_module = load_module(
    'palet8_agents.agents.evaluator_agent_v2',
    'palet8_agents/agents/evaluator_agent_v2.py'
)
EvaluatorAgentV2 = evaluator_module.EvaluatorAgentV2


class TestEvaluatorAgentV2Init:
    """Tests for EvaluatorAgentV2 initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        agent = EvaluatorAgentV2()

        assert agent.name == "evaluator"
        assert agent.model_profile == "evaluator"
        assert "Quality gate" in agent.description

    def test_init_with_tools(self):
        """Test initialization with tools."""
        mock_tool = MagicMock()
        mock_tool.name = "prompt_quality"

        agent = EvaluatorAgentV2(tools=[mock_tool])

        assert len(agent.tools) == 1


class TestEvaluatorAgentV2CreatePlan:
    """Tests for Phase 1: create_plan (prompt quality)."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return EvaluatorAgentV2()

    @pytest.fixture
    def context(self):
        """Create test context."""
        ctx = AgentContext(job_id="job-123", user_id="user-456")
        ctx.plan = {
            "prompt": "A beautiful sunset over the ocean",
            "negative_prompt": "blurry, low quality",
            "mode": "STANDARD",
            "dimensions": {"subject": "sunset", "background": "ocean"},
        }
        ctx.requirements = {"product_type": "poster"}
        return ctx

    @pytest.mark.asyncio
    async def test_create_plan_pass(self, agent, context):
        """Test create_plan when prompt quality passes."""
        # Mock call_tool
        async def mock_call_tool(tool_name, action, **kwargs):
            if tool_name == "prompt_quality" and action == "assess_quality":
                return ToolResult(
                    success=True,
                    data={
                        "overall": 0.85,
                        "dimensions": {"coverage": 0.9, "clarity": 0.8},
                        "mode": "STANDARD",
                        "threshold": 0.7,
                        "decision": "PASS",
                        "feedback": [],
                        "failed_dimensions": [],
                    },
                )
            elif tool_name == "image_evaluation" and action == "get_thresholds":
                return ToolResult(success=True, data={"thresholds": {"overall": 0.8}})
            elif tool_name == "image_evaluation" and action == "get_weights":
                return ToolResult(success=True, data={"weights": {"prompt_fidelity": 0.25}})
            return ToolResult(success=False, data=None)

        agent.call_tool = mock_call_tool

        result = await agent.run(context, phase="create_plan")

        assert result.success is True
        assert result.data["action"] == "proceed_to_generation"
        assert result.next_agent is None

    @pytest.mark.asyncio
    async def test_create_plan_fix_required(self, agent, context):
        """Test create_plan when prompt needs fixing."""
        async def mock_call_tool(tool_name, action, **kwargs):
            if tool_name == "prompt_quality" and action == "assess_quality":
                return ToolResult(
                    success=True,
                    data={
                        "overall": 0.55,
                        "dimensions": {"coverage": 0.5, "clarity": 0.6},
                        "mode": "STANDARD",
                        "threshold": 0.7,
                        "decision": "FIX_REQUIRED",
                        "feedback": ["Missing style details"],
                        "failed_dimensions": ["coverage"],
                    },
                )
            elif tool_name == "image_evaluation":
                return ToolResult(success=True, data={"thresholds": {}, "weights": {}})
            return ToolResult(success=False, data=None)

        agent.call_tool = mock_call_tool

        result = await agent.run(context, phase="create_plan")

        assert result.success is True
        assert result.data["action"] == "fix_required"
        assert result.next_agent == "planner"
        assert "feedback" in result.data

    @pytest.mark.asyncio
    async def test_create_plan_policy_fail(self, agent, context):
        """Test create_plan when policy violation detected."""
        async def mock_call_tool(tool_name, action, **kwargs):
            if tool_name == "prompt_quality" and action == "assess_quality":
                return ToolResult(
                    success=True,
                    data={
                        "overall": 0.0,
                        "dimensions": {},
                        "mode": "STANDARD",
                        "threshold": 0.7,
                        "decision": "POLICY_FAIL",
                        "feedback": ["Policy violation"],
                        "failed_dimensions": [],
                    },
                )
            elif tool_name == "image_evaluation":
                return ToolResult(success=True, data={"thresholds": {}, "weights": {}})
            return ToolResult(success=False, data=None)

        agent.call_tool = mock_call_tool

        result = await agent.run(context, phase="create_plan")

        assert result.success is False
        assert result.error_code == "POLICY_VIOLATION"
        assert result.next_agent is None

    @pytest.mark.asyncio
    async def test_create_plan_tool_failure_fallback(self, agent, context):
        """Test create_plan falls back gracefully on tool failure."""
        async def mock_call_tool(tool_name, action, **kwargs):
            return ToolResult(success=False, data=None, error="Tool unavailable")

        agent.call_tool = mock_call_tool

        result = await agent.run(context, phase="create_plan")

        # Should fallback to pass with warning
        assert result.success is True
        assert result.data["action"] == "proceed_to_generation"


class TestEvaluatorAgentV2Execute:
    """Tests for Phase 2: execute (result quality)."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return EvaluatorAgentV2()

    @pytest.fixture
    def context(self):
        """Create test context."""
        ctx = AgentContext(job_id="job-123", user_id="user-456")
        ctx.metadata["evaluation_plan"] = {
            "job_id": "job-123",
            "prompt": "A sunset",
            "negative_prompt": "blurry",
            "mode": "STANDARD",
        }
        ctx.metadata["assembly_request"] = {
            "prompt": "A sunset",
            "mode": "STANDARD",
        }
        return ctx

    @pytest.fixture
    def image_data(self):
        """Create test image data."""
        return {
            "width": 1024,
            "height": 1024,
            "url": "http://example.com/image.png",
            "description": "A beautiful sunset over the ocean",
        }

    @pytest.mark.asyncio
    async def test_execute_approve(self, agent, context, image_data):
        """Test execute when result quality is approved."""
        async def mock_call_tool(tool_name, action, **kwargs):
            if tool_name == "image_evaluation" and action == "evaluate_image":
                return ToolResult(
                    success=True,
                    data={
                        "overall": 0.9,
                        "dimensions": {"prompt_fidelity": 0.9, "technical_quality": 0.85},
                        "mode": "STANDARD",
                        "threshold": 0.8,
                        "decision": "APPROVE",
                        "feedback": [],
                        "failed_dimensions": [],
                        "retry_suggestions": [],
                        "is_acceptable": True,
                    },
                )
            elif tool_name == "image_evaluation" and action == "should_retry":
                return ToolResult(success=True, data={"should_retry": False})
            return ToolResult(success=False, data=None)

        agent.call_tool = mock_call_tool

        result = await agent.run(context, phase="execute", image_data=image_data)

        assert result.success is True
        assert result.data["action"] == "approved"
        assert result.next_agent == "pali"

    @pytest.mark.asyncio
    async def test_execute_reject(self, agent, context, image_data):
        """Test execute when result is rejected."""
        async def mock_call_tool(tool_name, action, **kwargs):
            if tool_name == "image_evaluation" and action == "evaluate_image":
                return ToolResult(
                    success=True,
                    data={
                        "overall": 0.5,
                        "dimensions": {"prompt_fidelity": 0.4, "technical_quality": 0.6},
                        "mode": "STANDARD",
                        "threshold": 0.8,
                        "decision": "REJECT",
                        "feedback": ["Subject not matching prompt"],
                        "failed_dimensions": ["prompt_fidelity"],
                        "retry_suggestions": [
                            {"dimension": "prompt_fidelity", "suggested_changes": ["Be more specific"]}
                        ],
                        "is_acceptable": False,
                    },
                )
            elif tool_name == "image_evaluation" and action == "should_retry":
                return ToolResult(success=True, data={"should_retry": True})
            return ToolResult(success=False, data=None)

        agent.call_tool = mock_call_tool

        result = await agent.run(context, phase="execute", image_data=image_data)

        assert result.success is True
        assert result.data["action"] == "rejected"
        assert result.data["should_retry"] is True
        assert result.next_agent == "planner"

    @pytest.mark.asyncio
    async def test_execute_reject_no_retry(self, agent, context, image_data):
        """Test execute when rejected but should not retry."""
        async def mock_call_tool(tool_name, action, **kwargs):
            if tool_name == "image_evaluation" and action == "evaluate_image":
                return ToolResult(
                    success=True,
                    data={
                        "overall": 0.5,
                        "dimensions": {},
                        "mode": "STANDARD",
                        "threshold": 0.8,
                        "decision": "REJECT",
                        "feedback": ["Quality too low"],
                        "failed_dimensions": ["technical_quality"],
                        "retry_suggestions": [],
                        "is_acceptable": False,
                    },
                )
            elif tool_name == "image_evaluation" and action == "should_retry":
                return ToolResult(success=True, data={"should_retry": False})
            return ToolResult(success=False, data=None)

        agent.call_tool = mock_call_tool

        result = await agent.run(context, phase="execute", image_data=image_data)

        assert result.success is True
        assert result.data["action"] == "rejected"
        assert result.data["should_retry"] is False
        assert result.next_agent is None  # No retry

    @pytest.mark.asyncio
    async def test_execute_missing_image_data(self, agent, context):
        """Test execute fails without image data."""
        result = await agent.run(context, phase="execute", image_data=None)

        assert result.success is False
        assert result.error_code == "MISSING_IMAGE_DATA"

    @pytest.mark.asyncio
    async def test_execute_policy_fail(self, agent, context, image_data):
        """Test execute when policy violation in result."""
        async def mock_call_tool(tool_name, action, **kwargs):
            if tool_name == "image_evaluation" and action == "evaluate_image":
                return ToolResult(
                    success=True,
                    data={
                        "overall": 0.0,
                        "dimensions": {},
                        "mode": "STANDARD",
                        "threshold": 0.8,
                        "decision": "POLICY_FAIL",
                        "feedback": ["Unsafe content detected"],
                        "failed_dimensions": [],
                        "retry_suggestions": [],
                        "is_acceptable": False,
                    },
                )
            elif tool_name == "image_evaluation" and action == "should_retry":
                return ToolResult(success=True, data={"should_retry": False})
            return ToolResult(success=False, data=None)

        agent.call_tool = mock_call_tool

        result = await agent.run(context, phase="execute", image_data=image_data)

        assert result.success is False
        assert result.error_code == "POLICY_VIOLATION"


class TestEvaluatorAgentV2EdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return EvaluatorAgentV2()

    @pytest.mark.asyncio
    async def test_unknown_phase(self, agent):
        """Test handling of unknown phase."""
        context = AgentContext(job_id="job-123", user_id="user-456")

        result = await agent.run(context, phase="unknown_phase")

        assert result.success is False
        assert result.error_code == "INVALID_PHASE"

    @pytest.mark.asyncio
    async def test_create_plan_from_assembly_request(self, agent):
        """Test create_plan extracts data from assembly_request in metadata."""
        context = AgentContext(job_id="job-123", user_id="user-456")
        context.metadata["assembly_request"] = {
            "prompt": "A cat sitting",
            "negative_prompt": "blurry",
            "mode": "RELAX",
            "dimensions": {"subject": "cat"},
        }
        context.requirements = {"product_type": "mug"}

        async def mock_call_tool(tool_name, action, **kwargs):
            if tool_name == "prompt_quality" and action == "assess_quality":
                # Verify it got the right params
                assert kwargs.get("prompt") == "A cat sitting"
                assert kwargs.get("mode") == "RELAX"
                return ToolResult(
                    success=True,
                    data={
                        "overall": 0.8,
                        "dimensions": {},
                        "mode": "RELAX",
                        "threshold": 0.5,
                        "decision": "PASS",
                        "feedback": [],
                        "failed_dimensions": [],
                    },
                )
            elif tool_name == "image_evaluation":
                return ToolResult(success=True, data={"thresholds": {}, "weights": {}})
            return ToolResult(success=False, data=None)

        agent.call_tool = mock_call_tool

        result = await agent.run(context, phase="create_plan")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_without_stored_plan(self, agent):
        """Test execute creates minimal plan when none stored."""
        context = AgentContext(job_id="job-123", user_id="user-456")
        context.metadata["assembly_request"] = {
            "prompt": "A dog",
            "mode": "STANDARD",
        }
        image_data = {"width": 1024, "height": 1024}

        async def mock_call_tool(tool_name, action, **kwargs):
            if tool_name == "image_evaluation" and action == "evaluate_image":
                return ToolResult(
                    success=True,
                    data={
                        "overall": 0.85,
                        "dimensions": {},
                        "mode": "STANDARD",
                        "threshold": 0.8,
                        "decision": "APPROVE",
                        "feedback": [],
                        "failed_dimensions": [],
                        "retry_suggestions": [],
                        "is_acceptable": True,
                    },
                )
            elif tool_name == "image_evaluation" and action == "should_retry":
                return ToolResult(success=True, data={"should_retry": False})
            return ToolResult(success=False, data=None)

        agent.call_tool = mock_call_tool

        result = await agent.run(context, phase="execute", image_data=image_data)

        assert result.success is True
        assert result.data["action"] == "approved"


class TestDecisionMethods:
    """Tests for decision helper methods."""

    def test_decide_prompt_result_pass(self):
        """Test _decide_prompt_result for PASS."""
        agent = EvaluatorAgentV2()
        quality = PromptQualityResult(
            overall=0.85,
            dimensions={},
            mode="STANDARD",
            threshold=0.7,
            decision="PASS",
        )
        plan = EvaluationPlan(job_id="job-123", prompt="Test")

        result = agent._decide_prompt_result(quality, plan)

        assert result.success is True
        assert result.data["action"] == "proceed_to_generation"

    def test_decide_prompt_result_fix_required(self):
        """Test _decide_prompt_result for FIX_REQUIRED."""
        agent = EvaluatorAgentV2()
        quality = PromptQualityResult(
            overall=0.5,
            dimensions={"coverage": 0.4},
            mode="STANDARD",
            threshold=0.7,
            decision="FIX_REQUIRED",
            feedback=["Missing details"],
            failed_dimensions=["coverage"],
        )
        plan = EvaluationPlan(job_id="job-123", prompt="Test")

        result = agent._decide_prompt_result(quality, plan)

        assert result.success is True
        assert result.data["action"] == "fix_required"
        assert result.next_agent == "planner"

    def test_decide_result_approve(self):
        """Test _decide_result for APPROVE."""
        agent = EvaluatorAgentV2()
        quality = ResultQualityResult(
            overall=0.9,
            dimensions={},
            mode="STANDARD",
            threshold=0.8,
            decision="APPROVE",
        )
        plan = EvaluationPlan(job_id="job-123", prompt="Test")

        result = agent._decide_result(quality, plan, should_retry=False)

        assert result.success is True
        assert result.data["action"] == "approved"
        assert result.next_agent == "pali"

    def test_decide_result_reject_with_retry(self):
        """Test _decide_result for REJECT with retry."""
        agent = EvaluatorAgentV2()
        quality = ResultQualityResult(
            overall=0.5,
            dimensions={},
            mode="STANDARD",
            threshold=0.8,
            decision="REJECT",
            feedback=["Quality too low"],
            failed_dimensions=["technical_quality"],
        )
        plan = EvaluationPlan(job_id="job-123", prompt="Test")

        result = agent._decide_result(quality, plan, should_retry=True)

        assert result.success is True
        assert result.data["action"] == "rejected"
        assert result.next_agent == "planner"

    def test_decide_result_reject_no_retry(self):
        """Test _decide_result for REJECT without retry."""
        agent = EvaluatorAgentV2()
        quality = ResultQualityResult(
            overall=0.5,
            dimensions={},
            mode="STANDARD",
            threshold=0.8,
            decision="REJECT",
        )
        plan = EvaluationPlan(job_id="job-123", prompt="Test")

        result = agent._decide_result(quality, plan, should_retry=False)

        assert result.success is True
        assert result.data["action"] == "rejected"
        assert result.next_agent is None
