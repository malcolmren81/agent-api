"""Tests for palet8_agents.agents.planner_agent_v2 module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

# Import directly to avoid Prisma dependency issues
import sys
import importlib.util


# Load the module directly
def load_planner_agent_v2():
    """Load the planner_agent_v2 module directly."""
    spec = importlib.util.spec_from_file_location(
        'planner_agent_v2',
        'palet8_agents/agents/planner_agent_v2.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['palet8_agents.agents.planner_agent_v2'] = module
    spec.loader.exec_module(module)
    return module


# Load module
planner_module = load_planner_agent_v2()
PlannerAgentV2 = planner_module.PlannerAgentV2

from palet8_agents.core.agent import AgentContext, AgentResult
from palet8_agents.models import (
    ContextCompleteness,
    SafetyClassification,
    PipelineConfig,
)
from palet8_agents.models.planning import PlanningTask, PromptPlan, ContextSummary


class TestPlannerAgentV2Init:
    """Tests for PlannerAgentV2 initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        agent = PlannerAgentV2()
        assert agent.name == "planner"
        assert agent.description == "Thin coordinator for planning, routing, and model selection"
        assert agent.model_profile == "planner"
        assert agent.min_context_completeness == 0.5

    def test_init_with_services(self):
        """Test initialization with services."""
        mock_text = MagicMock()
        mock_context = MagicMock()
        mock_model = MagicMock()
        mock_safety = MagicMock()

        agent = PlannerAgentV2(
            text_service=mock_text,
            context_analysis_service=mock_context,
            model_selection_service=mock_model,
            safety_classification_service=mock_safety,
        )

        assert agent._text_service == mock_text
        assert agent._context_analysis_service == mock_context
        assert agent._model_selection_service == mock_model
        assert agent._safety_classification_service == mock_safety
        assert agent._owns_services is False


class TestPlannerAgentV2ComplexityClassification:
    """Tests for complexity classification."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return PlannerAgentV2()

    def test_classify_simple(self, agent):
        """Test simple complexity classification."""
        requirements = {
            "subject": "a cat",
        }
        complexity = agent._classify_complexity(requirements)
        assert complexity == "simple"

    def test_classify_standard(self, agent):
        """Test standard complexity classification."""
        requirements = {
            "subject": "a cat",
            "composition": "centered",
        }
        complexity = agent._classify_complexity(requirements)
        assert complexity == "standard"

    def test_classify_standard_with_style(self, agent):
        """Test standard with style."""
        requirements = {
            "subject": "a cat",
            "style": "cartoon",
        }
        complexity = agent._classify_complexity(requirements)
        assert complexity == "standard"

    def test_classify_standard_with_elements(self, agent):
        """Test standard with multiple include_elements."""
        requirements = {
            "subject": "a cat",
            "include_elements": ["flowers", "grass", "butterfly"],
        }
        complexity = agent._classify_complexity(requirements)
        assert complexity == "standard"

    def test_classify_advanced_with_print_method(self, agent):
        """Test advanced when print_method is present (technical constraint)."""
        requirements = {
            "subject": "a cat",
            "print_method": "screen_print",
        }
        complexity = agent._classify_complexity(requirements)
        assert complexity == "advanced"

    def test_classify_advanced_many_factors(self, agent):
        """Test advanced with many stylistic factors."""
        requirements = {
            "subject": "a cat",
            "composition": "rule of thirds",
            "style": "watercolor",
            "reference_image": "http://example.com/ref.png",
            "colors": ["blue", "green"],
        }
        complexity = agent._classify_complexity(requirements)
        assert complexity == "advanced"

    def test_classify_advanced_all_factors(self, agent):
        """Test advanced with all factors present."""
        requirements = {
            "subject": "a cat",
            "composition": "dynamic",
            "print_method": "dtg",
            "reference_image": "http://example.com/ref.png",
            "include_elements": ["sun", "moon", "stars"],
            "style": "illustration",
            "colors": ["warm tones"],
        }
        complexity = agent._classify_complexity(requirements)
        assert complexity == "advanced"


class TestPlannerAgentV2ContextSufficiency:
    """Tests for context sufficiency checks."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return PlannerAgentV2()

    def test_is_context_sufficient_true(self, agent):
        """Test context is sufficient."""
        completeness = ContextCompleteness(
            score=0.7,
            is_sufficient=True,
            missing_fields=[],
            clarifying_questions=[],
        )
        assert agent._is_context_sufficient(completeness) is True

    def test_is_context_sufficient_false_low_score(self, agent):
        """Test context insufficient due to low score."""
        completeness = ContextCompleteness(
            score=0.3,
            is_sufficient=False,
            missing_fields=["subject"],
            clarifying_questions=["What subject?"],
        )
        assert agent._is_context_sufficient(completeness) is False

    def test_is_context_sufficient_false_not_sufficient_flag(self, agent):
        """Test context insufficient due to is_sufficient=False."""
        completeness = ContextCompleteness(
            score=0.6,
            is_sufficient=False,
            missing_fields=["style"],
            clarifying_questions=["What style?"],
        )
        assert agent._is_context_sufficient(completeness) is False

    def test_is_context_sufficient_edge_case_threshold(self, agent):
        """Test context at exactly threshold."""
        completeness = ContextCompleteness(
            score=0.5,
            is_sufficient=True,
            missing_fields=[],
            clarifying_questions=[],
        )
        assert agent._is_context_sufficient(completeness) is True


class TestPlannerAgentV2SafetyDecision:
    """Tests for safety decision-making."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return PlannerAgentV2()

    def test_is_safe_to_proceed_true(self, agent):
        """Test safe to proceed when is_safe=True."""
        safety = SafetyClassification(
            is_safe=True,
            requires_review=False,
            risk_level="low",
            categories=[],
        )
        assert agent._is_safe_to_proceed(safety) is True

    def test_is_safe_to_proceed_false(self, agent):
        """Test not safe to proceed when is_safe=False."""
        safety = SafetyClassification(
            is_safe=False,
            requires_review=True,
            risk_level="high",
            categories=["nsfw"],
            reason="Detected inappropriate content",
        )
        assert agent._is_safe_to_proceed(safety) is False


class TestPlannerAgentV2InitialPhase:
    """Tests for initial phase handling."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return PlannerAgentV2()

    @pytest.fixture
    def mock_context(self):
        """Create mock AgentContext."""
        return AgentContext(
            user_id="user-123",
            job_id="job-456",
            requirements={
                "subject": "a cat",
                "style": "cartoon",
            },
        )

    @pytest.mark.asyncio
    async def test_handle_initial_insufficient_context(self, agent, mock_context):
        """Test initial phase with insufficient context."""
        # Mock context analysis service to return insufficient
        mock_service = MagicMock()
        mock_service.evaluate_completeness.return_value = ContextCompleteness(
            score=0.3,
            is_sufficient=False,
            missing_fields=["mood"],
            clarifying_questions=["What mood should the design have?"],
        )
        agent._context_analysis_service = mock_service

        result = await agent.run(mock_context, phase="initial")

        assert result.success is True
        assert result.data["action"] == "needs_clarification"
        assert result.next_agent == "pali"
        assert "questions" in result.data
        assert "missing_fields" in result.data

    @pytest.mark.asyncio
    async def test_handle_initial_safety_blocked(self, agent, mock_context):
        """Test initial phase blocked by safety."""
        # Mock services
        mock_context_service = MagicMock()
        mock_context_service.evaluate_completeness.return_value = ContextCompleteness(
            score=0.8,
            is_sufficient=True,
            missing_fields=[],
            clarifying_questions=[],
        )
        agent._context_analysis_service = mock_context_service

        # Mock safety service to return unsafe
        from palet8_agents.models.safety import SafetyFlag, SafetyCategory, SafetySeverity
        mock_safety_service = MagicMock()
        mock_safety_service.classify_content = AsyncMock(return_value=SafetyFlag(
            category=SafetyCategory.NSFW,
            severity=SafetySeverity.CRITICAL,
            score=1.0,
            description="Blocked content",
            source="requirements",
        ))
        agent._safety_classification_service = mock_safety_service

        result = await agent.run(mock_context, phase="initial")

        assert result.success is False
        assert result.error_code == "SAFETY_BLOCKED"
        assert "blocked" in result.data["action"]

    @pytest.mark.asyncio
    async def test_handle_initial_success(self, agent, mock_context):
        """Test successful initial phase."""
        # Mock context analysis service
        mock_context_service = MagicMock()
        mock_context_service.evaluate_completeness.return_value = ContextCompleteness(
            score=0.8,
            is_sufficient=True,
            missing_fields=[],
            clarifying_questions=[],
        )
        agent._context_analysis_service = mock_context_service

        # Mock safety service to return safe
        mock_safety_service = MagicMock()
        mock_safety_service.classify_content = AsyncMock(return_value=None)
        agent._safety_classification_service = mock_safety_service

        result = await agent.run(mock_context, phase="initial")

        assert result.success is True
        assert result.data["action"] == "build_prompt"
        assert result.next_agent == "react_prompt"
        assert "planning_task" in result.data
        assert mock_context.metadata.get("planning_task") is not None


class TestPlannerAgentV2PostPromptPhase:
    """Tests for post_prompt phase handling."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return PlannerAgentV2()

    @pytest.fixture
    def mock_context_with_prompt_plan(self):
        """Create mock context with prompt_plan."""
        prompt_plan = PromptPlan(
            prompt="A cute cartoon cat",
            negative_prompt="blurry, low quality",
            dimensions={"subject": "cat", "aesthetic": "cartoon"},
            quality_score=0.85,
            quality_acceptable=True,
            mode="STANDARD",
        )
        context = AgentContext(
            user_id="user-123",
            job_id="job-456",
            requirements={"subject": "a cat", "style": "cartoon"},
            metadata={
                "prompt_plan": prompt_plan.to_dict(),
                "safety": SafetyClassification(
                    is_safe=True,
                    requires_review=False,
                    risk_level="low",
                    categories=[],
                ).to_dict(),
            },
        )
        return context

    @pytest.mark.asyncio
    async def test_handle_post_prompt_missing_plan(self, agent):
        """Test post_prompt with missing prompt_plan."""
        context = AgentContext(
            user_id="user-123",
            job_id="job-456",
        )

        result = await agent.run(context, phase="post_prompt")

        assert result.success is False
        assert result.error_code == "MISSING_PROMPT_PLAN"

    @pytest.mark.asyncio
    async def test_handle_post_prompt_success(self, agent, mock_context_with_prompt_plan):
        """Test successful post_prompt phase."""
        # Mock model selection service
        mock_model_service = MagicMock()
        mock_model_service.select_model = AsyncMock(return_value=(
            "midjourney-v7",
            "Best for cartoon style",
            ["flux-2", "sdxl"],
        ))
        mock_model_service.select_pipeline = AsyncMock(return_value=PipelineConfig(
            pipeline_type="single",
        ))
        agent._model_selection_service = mock_model_service

        result = await agent.run(mock_context_with_prompt_plan, phase="post_prompt")

        assert result.success is True
        assert result.data["action"] == "evaluate_plan"
        assert result.next_agent == "evaluator"
        assert "assembly_request" in result.data
        assert mock_context_with_prompt_plan.metadata.get("assembly_request") is not None


class TestPlannerAgentV2FixPlanPhase:
    """Tests for fix_plan phase handling."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return PlannerAgentV2()

    @pytest.fixture
    def mock_context_with_previous_plan(self):
        """Create context with previous prompt plan."""
        prompt_plan = PromptPlan(
            prompt="A cat",
            negative_prompt="",
            quality_score=0.4,
            quality_acceptable=False,
        )
        context = AgentContext(
            user_id="user-123",
            job_id="job-456",
            requirements={"subject": "a cat"},
            metadata={
                "prompt_plan": prompt_plan.to_dict(),
                "complexity": "standard",
            },
        )
        return context

    @pytest.mark.asyncio
    async def test_handle_fix_plan(self, agent, mock_context_with_previous_plan):
        """Test fix_plan phase delegation."""
        evaluation_feedback = {
            "passed": False,
            "overall_score": 0.4,
            "issues": ["Prompt too short", "Missing style details"],
            "retry_suggestions": ["Add more descriptive details"],
        }

        result = await agent.run(
            mock_context_with_previous_plan,
            phase="fix_plan",
            evaluation_feedback=evaluation_feedback,
        )

        assert result.success is True
        assert result.data["action"] == "fix_prompt"
        assert result.next_agent == "react_prompt"

        planning_task = result.data["planning_task"]
        assert planning_task["phase"] == "fix_plan"
        assert planning_task["previous_plan"] is not None
        assert planning_task["evaluation_feedback"] == evaluation_feedback


class TestPlannerAgentV2EditPhase:
    """Tests for edit phase handling."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return PlannerAgentV2()

    @pytest.fixture
    def mock_context_with_existing_plan(self):
        """Create context with existing prompt plan."""
        prompt_plan = PromptPlan(
            prompt="A happy cat in a garden",
            negative_prompt="blurry",
            quality_score=0.85,
            quality_acceptable=True,
        )
        context = AgentContext(
            user_id="user-123",
            job_id="job-456",
            requirements={"subject": "a cat", "mood": "happy"},
            metadata={
                "prompt_plan": prompt_plan.to_dict(),
                "complexity": "standard",
            },
        )
        return context

    @pytest.mark.asyncio
    async def test_handle_edit(self, agent, mock_context_with_existing_plan):
        """Test edit phase delegation."""
        edit_instructions = "Make the background darker"

        result = await agent.run(
            mock_context_with_existing_plan,
            phase="edit",
            user_input=edit_instructions,
        )

        assert result.success is True
        assert result.data["action"] == "edit_prompt"
        assert result.next_agent == "react_prompt"

        planning_task = result.data["planning_task"]
        assert planning_task["phase"] == "edit"
        assert planning_task["previous_plan"] is not None
        assert planning_task["edit_instructions"] == edit_instructions


class TestPlannerAgentV2UnknownPhase:
    """Tests for unknown phase handling."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return PlannerAgentV2()

    @pytest.mark.asyncio
    async def test_unknown_phase(self, agent):
        """Test handling of unknown phase."""
        context = AgentContext(user_id="user", job_id="job")

        result = await agent.run(context, phase="unknown_phase")

        assert result.success is False
        assert result.error_code == "UNKNOWN_PHASE"


class TestPlannerAgentV2ModelSelection:
    """Tests for model selection."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return PlannerAgentV2()

    @pytest.mark.asyncio
    async def test_select_model_success(self, agent):
        """Test successful model selection."""
        mock_service = MagicMock()
        mock_service.select_model = AsyncMock(return_value=(
            "imagen-4-ultra",
            "Best for realistic images",
            ["midjourney-v7", "flux-2"],
        ))
        agent._model_selection_service = mock_service

        model_id, rationale, alternatives = await agent._select_model(
            mode="STANDARD",
            requirements={"subject": "landscape"},
        )

        assert model_id == "imagen-4-ultra"
        assert "realistic" in rationale.lower()
        assert len(alternatives) == 2

    @pytest.mark.asyncio
    async def test_select_model_fallback(self, agent):
        """Test model selection fallback on error."""
        mock_service = MagicMock()
        mock_service.select_model = AsyncMock(side_effect=Exception("Service error"))
        agent._model_selection_service = mock_service

        model_id, rationale, alternatives = await agent._select_model(
            mode="STANDARD",
            requirements={},
        )

        assert model_id == "default-model"
        assert "fallback" in rationale.lower()
        assert alternatives == []


class TestPlannerAgentV2PipelineSelection:
    """Tests for pipeline selection."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return PlannerAgentV2()

    @pytest.mark.asyncio
    async def test_select_pipeline_single(self, agent):
        """Test single pipeline selection."""
        mock_service = MagicMock()
        mock_service.select_pipeline = AsyncMock(return_value=PipelineConfig(
            pipeline_type="single",
        ))
        agent._model_selection_service = mock_service

        pipeline = await agent._select_pipeline(
            requirements={"subject": "cat"},
            prompt="A cute cat",
        )

        assert pipeline.pipeline_type == "single"

    @pytest.mark.asyncio
    async def test_select_pipeline_dual(self, agent):
        """Test dual pipeline selection."""
        mock_service = MagicMock()
        mock_service.select_pipeline = AsyncMock(return_value=PipelineConfig(
            pipeline_type="dual",
            pipeline_name="text_in_image",
            stage_1_model="flux-2-flex",
            stage_1_purpose="Layout and composition",
            stage_2_model="qwen-image-edit",
            stage_2_purpose="Text placement",
        ))
        agent._model_selection_service = mock_service

        pipeline = await agent._select_pipeline(
            requirements={"subject": "poster with text"},
            prompt="A poster with typography",
        )

        assert pipeline.pipeline_type == "dual"
        assert pipeline.stage_1_model == "flux-2-flex"
        assert pipeline.stage_2_model == "qwen-image-edit"

    @pytest.mark.asyncio
    async def test_select_pipeline_fallback(self, agent):
        """Test pipeline selection fallback on error."""
        mock_service = MagicMock()
        mock_service.select_pipeline = AsyncMock(side_effect=Exception("Service error"))
        agent._model_selection_service = mock_service

        pipeline = await agent._select_pipeline(
            requirements={},
            prompt="A cat",
        )

        assert pipeline.pipeline_type == "single"


class TestPlannerAgentV2AssemblyRequest:
    """Tests for assembly request building."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return PlannerAgentV2()

    def test_build_assembly_request(self, agent):
        """Test building assembly request."""
        context = AgentContext(
            user_id="user-123",
            job_id="job-456",
            requirements={
                "subject": "a cat",
                "product_type": "poster",
                "print_method": "screen_print",
            },
        )

        prompt_plan = PromptPlan(
            prompt="A beautiful cartoon cat",
            negative_prompt="blurry, low quality",
            dimensions={"subject": "cat", "aesthetic": "cartoon"},
            quality_score=0.85,
            quality_acceptable=True,
            mode="STANDARD",
            revision_count=1,
            context_summary=ContextSummary(
                user_history_count=5,
                art_references_count=2,
            ),
        )

        pipeline = PipelineConfig(pipeline_type="single")

        safety = SafetyClassification(
            is_safe=True,
            requires_review=False,
            risk_level="low",
            categories=[],
        )

        assembly = agent._build_assembly_request(
            context=context,
            prompt_plan=prompt_plan,
            model_id="midjourney-v7",
            model_rationale="Best for cartoons",
            model_alternatives=["flux-2"],
            pipeline=pipeline,
            safety=safety,
        )

        assert assembly.prompt == "A beautiful cartoon cat"
        assert assembly.negative_prompt == "blurry, low quality"
        assert assembly.mode == "STANDARD"
        assert assembly.model_id == "midjourney-v7"
        assert assembly.model_rationale == "Best for cartoons"
        assert assembly.pipeline.pipeline_type == "single"
        assert assembly.job_id == "job-456"
        assert assembly.user_id == "user-123"
        assert assembly.product_type == "poster"
        assert assembly.print_method == "screen_print"
        assert assembly.revision_count == 1
        assert assembly.safety.is_safe is True

    def test_build_assembly_request_default_safety(self, agent):
        """Test building assembly request with default safety."""
        context = AgentContext(
            user_id="user-123",
            job_id="job-456",
        )

        prompt_plan = PromptPlan(
            prompt="Test prompt",
        )

        pipeline = PipelineConfig(pipeline_type="single")

        assembly = agent._build_assembly_request(
            context=context,
            prompt_plan=prompt_plan,
            model_id="test-model",
            model_rationale="Test",
            model_alternatives=[],
            pipeline=pipeline,
            safety=None,
        )

        assert assembly.safety.is_safe is True
        assert assembly.safety.risk_level == "low"


class TestPlannerAgentV2ContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with PlannerAgentV2() as agent:
            assert agent.name == "planner"

    @pytest.mark.asyncio
    async def test_close_with_owned_services(self):
        """Test close cleans up owned services."""
        agent = PlannerAgentV2()

        # Create services (agent owns them)
        mock_text_service = MagicMock()
        mock_text_service.close = AsyncMock()
        agent._text_service = mock_text_service
        agent._owns_services = True

        await agent.close()

        # Service was closed and then set to None
        mock_text_service.close.assert_called_once()
        assert agent._text_service is None

    @pytest.mark.asyncio
    async def test_close_with_external_services(self):
        """Test close doesn't touch external services."""
        mock_text = MagicMock()
        mock_text.close = AsyncMock()

        agent = PlannerAgentV2(text_service=mock_text)

        await agent.close()

        mock_text.close.assert_not_called()
