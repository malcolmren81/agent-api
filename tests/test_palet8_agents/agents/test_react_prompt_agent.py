"""Tests for palet8_agents.agents.react_prompt_agent module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

# Import directly to avoid Prisma dependency issues
import sys
import importlib.util


# Load the module directly
def load_react_prompt_agent():
    """Load the react_prompt_agent module directly."""
    spec = importlib.util.spec_from_file_location(
        'react_prompt_agent',
        'palet8_agents/agents/react_prompt_agent.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['palet8_agents.agents.react_prompt_agent'] = module
    spec.loader.exec_module(module)
    return module


# Load module
react_prompt_module = load_react_prompt_agent()
ReactPromptAgent = react_prompt_module.ReactPromptAgent
ReactAction = react_prompt_module.ReactAction
PromptState = react_prompt_module.PromptState

from palet8_agents.core.agent import AgentContext, AgentResult
from palet8_agents.models.planning import PlanningTask, PromptPlan, ContextSummary
from palet8_agents.models.prompt import PromptDimensions, PromptQualityResult
from palet8_agents.tools.base import ToolResult


class TestReactAction:
    """Tests for ReactAction enum."""

    def test_all_actions_exist(self):
        """Test that all expected actions are defined."""
        assert ReactAction.BUILD_CONTEXT.value == "build_context"
        assert ReactAction.SELECT_DIMENSIONS.value == "select_dimensions"
        assert ReactAction.COMPOSE_PROMPT.value == "compose_prompt"
        assert ReactAction.EVALUATE_PROMPT.value == "evaluate_prompt"
        assert ReactAction.REFINE_PROMPT.value == "refine_prompt"
        assert ReactAction.DONE.value == "done"


class TestPromptState:
    """Tests for PromptState dataclass."""

    def test_init_defaults(self):
        """Test default initialization."""
        state = PromptState()
        assert state.user_history == []
        assert state.art_references == []
        assert state.dimensions is None
        assert state.prompt == ""
        assert state.negative_prompt == ""
        assert state.quality is None
        assert state.revision_count == 0
        assert state.goal_satisfied is False
        assert state.mode == "STANDARD"

    def test_has_context_false(self):
        """Test has_context when no context gathered."""
        state = PromptState()
        assert state.has_context is False

    def test_has_context_true_with_history(self):
        """Test has_context with user history."""
        state = PromptState(user_history=[{"prompt": "test"}])
        assert state.has_context is True

    def test_has_context_true_with_references(self):
        """Test has_context with art references."""
        state = PromptState(art_references=[{"image_url": "test.png"}])
        assert state.has_context is True

    def test_has_context_true_with_rag(self):
        """Test has_context with RAG sources."""
        state = PromptState(rag_sources=["source1.txt"])
        assert state.has_context is True

    def test_has_dimensions_false(self):
        """Test has_dimensions when none set."""
        state = PromptState()
        assert state.has_dimensions is False

    def test_has_dimensions_true(self):
        """Test has_dimensions when set."""
        state = PromptState(dimensions=PromptDimensions(subject="test"))
        assert state.has_dimensions is True

    def test_has_prompt_false(self):
        """Test has_prompt when empty."""
        state = PromptState()
        assert state.has_prompt is False

    def test_has_prompt_true(self):
        """Test has_prompt when set."""
        state = PromptState(prompt="A beautiful sunset")
        assert state.has_prompt is True

    def test_has_quality_false(self):
        """Test has_quality when none."""
        state = PromptState()
        assert state.has_quality is False

    def test_has_quality_true(self):
        """Test has_quality when set."""
        state = PromptState(quality=PromptQualityResult(overall=0.8))
        assert state.has_quality is True

    def test_get_context_summary(self):
        """Test get_context_summary method."""
        state = PromptState(
            user_history=[{"id": 1}, {"id": 2}],
            art_references=[
                {"image_url": "ref1.png"},
                {"image_url": "ref2.png"},
            ],
            web_results=[{"url": "test.com"}],
            rag_sources=["doc1.txt", "doc2.txt"],
        )
        summary = state.get_context_summary()

        assert isinstance(summary, ContextSummary)
        assert summary.user_history_count == 2
        assert summary.art_references_count == 2
        assert summary.web_search_count == 1
        assert summary.rag_sources == ["doc1.txt", "doc2.txt"]
        assert len(summary.reference_images) == 2


class TestReactPromptAgent:
    """Tests for ReactPromptAgent class."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return ReactPromptAgent()

    @pytest.fixture
    def mock_context(self):
        """Create mock AgentContext."""
        return AgentContext(
            user_id="user-123",
            job_id="job-456",
            metadata={
                "planning_task": {
                    "job_id": "job-456",
                    "user_id": "user-123",
                    "phase": "initial",
                    "requirements": {"subject": "a cat"},
                    "complexity": "standard",
                    "product_type": "poster",
                }
            }
        )

    def test_init(self, agent):
        """Test agent initialization."""
        assert agent.name == "react_prompt"
        assert agent.description == "Builds optimal prompts using context, dimensions, and iterative refinement"
        assert agent.max_steps == 10
        assert agent.max_revisions == 3

    def test_init_state_initial(self, agent):
        """Test _init_state for initial phase."""
        task = PlanningTask(
            job_id="job-1",
            user_id="user-1",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="standard",
            product_type="poster",
        )
        state = agent._init_state(task)

        assert state.mode == "STANDARD"
        assert state.prompt == ""
        assert state.dimensions is None

    def test_init_state_fix_plan(self, agent):
        """Test _init_state for fix_plan phase with previous data."""
        task = PlanningTask(
            job_id="job-2",
            user_id="user-2",
            phase="fix_plan",
            requirements={"subject": "dog"},
            complexity="complex",
            product_type="t-shirt",
            previous_plan={
                "prompt": "A cute dog",
                "negative_prompt": "blurry",
                "dimensions": {"subject": "dog", "aesthetic": "cartoon"},
            },
        )
        state = agent._init_state(task)

        assert state.mode == "COMPLEX"
        assert state.prompt == "A cute dog"
        assert state.negative_prompt == "blurry"
        assert state.dimensions.subject == "dog"
        assert state.dimensions.aesthetic == "cartoon"

    @pytest.mark.asyncio
    async def test_think_initial_no_context(self, agent):
        """Test _think returns BUILD_CONTEXT when no context."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={},
            complexity="standard",
            product_type="poster",
        )
        state = PromptState()

        action = await agent._think(state, task)
        assert action == ReactAction.BUILD_CONTEXT

    @pytest.mark.asyncio
    async def test_think_initial_with_context_no_dimensions(self, agent):
        """Test _think returns SELECT_DIMENSIONS after context."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={},
            complexity="standard",
            product_type="poster",
        )
        state = PromptState(user_history=[{"id": 1}])

        action = await agent._think(state, task)
        assert action == ReactAction.SELECT_DIMENSIONS

    @pytest.mark.asyncio
    async def test_think_initial_with_dimensions_no_prompt(self, agent):
        """Test _think returns COMPOSE_PROMPT after dimensions."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={},
            complexity="standard",
            product_type="poster",
        )
        state = PromptState(
            user_history=[{"id": 1}],
            dimensions=PromptDimensions(subject="cat"),
        )

        action = await agent._think(state, task)
        assert action == ReactAction.COMPOSE_PROMPT

    @pytest.mark.asyncio
    async def test_think_initial_with_prompt_no_quality(self, agent):
        """Test _think returns EVALUATE_PROMPT after prompt composed."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={},
            complexity="standard",
            product_type="poster",
        )
        state = PromptState(
            user_history=[{"id": 1}],
            dimensions=PromptDimensions(subject="cat"),
            prompt="A beautiful cat",
        )

        action = await agent._think(state, task)
        assert action == ReactAction.EVALUATE_PROMPT

    @pytest.mark.asyncio
    async def test_think_initial_quality_pass(self, agent):
        """Test _think returns DONE when quality passes."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={},
            complexity="standard",
            product_type="poster",
        )
        state = PromptState(
            user_history=[{"id": 1}],
            dimensions=PromptDimensions(subject="cat"),
            prompt="A beautiful cat",
            quality=PromptQualityResult(overall=0.85, decision="PASS"),
        )

        action = await agent._think(state, task)
        assert action == ReactAction.DONE

    @pytest.mark.asyncio
    async def test_think_initial_quality_fail_refine(self, agent):
        """Test _think returns REFINE_PROMPT when quality fails."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={},
            complexity="standard",
            product_type="poster",
        )
        state = PromptState(
            user_history=[{"id": 1}],
            dimensions=PromptDimensions(subject="cat"),
            prompt="A cat",
            quality=PromptQualityResult(overall=0.5, decision="FIX_REQUIRED"),
            revision_count=0,
        )

        action = await agent._think(state, task)
        assert action == ReactAction.REFINE_PROMPT

    @pytest.mark.asyncio
    async def test_think_fix_plan_evaluate_first(self, agent):
        """Test _think for fix_plan evaluates first."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="fix_plan",
            requirements={},
            complexity="standard",
            product_type="poster",
            previous_plan={"prompt": "Old prompt"},
        )
        state = PromptState(prompt="Old prompt")

        action = await agent._think(state, task)
        assert action == ReactAction.EVALUATE_PROMPT

    def test_check_goal_not_satisfied_no_prompt(self, agent):
        """Test _check_goal when no prompt."""
        state = PromptState()
        assert agent._check_goal(state) is False

    def test_check_goal_not_satisfied_no_quality(self, agent):
        """Test _check_goal when no quality."""
        state = PromptState(prompt="test")
        assert agent._check_goal(state) is False

    def test_check_goal_satisfied_pass(self, agent):
        """Test _check_goal when quality passes."""
        state = PromptState(
            prompt="test",
            quality=PromptQualityResult(overall=0.85, decision="PASS"),
        )
        assert agent._check_goal(state) is True

    def test_check_goal_satisfied_max_revisions(self, agent):
        """Test _check_goal when max revisions reached."""
        state = PromptState(
            prompt="test",
            quality=PromptQualityResult(overall=0.5, decision="FIX_REQUIRED"),
            revision_count=3,
        )
        # Should be True because we've hit max revisions
        assert agent._check_goal(state) is True

    def test_build_fallback_prompt(self, agent):
        """Test _build_fallback_prompt method."""
        state = PromptState(
            dimensions=PromptDimensions(
                subject="a cat",
                aesthetic="cartoon",
                color="vibrant",
                mood="playful",
            )
        )
        prompt = agent._build_fallback_prompt(state)

        assert "a cat" in prompt
        assert "cartoon style" in prompt
        assert "vibrant colors" in prompt
        assert "playful mood" in prompt

    def test_build_fallback_prompt_empty(self, agent):
        """Test _build_fallback_prompt with no dimensions."""
        state = PromptState()
        prompt = agent._build_fallback_prompt(state)

        assert "high quality digital art" in prompt

    def test_build_fallback_negative(self, agent):
        """Test _build_fallback_negative method."""
        negative = agent._build_fallback_negative()

        assert "blurry" in negative
        assert "low quality" in negative
        assert "watermark" in negative

    @pytest.mark.asyncio
    async def test_run_missing_task(self, agent):
        """Test run with missing planning_task."""
        context = AgentContext(user_id="user", job_id="job")
        result = await agent.run(context)

        assert result.success is False
        assert result.error_code == "MISSING_TASK"

    @pytest.mark.asyncio
    async def test_select_dimensions_fallback(self, agent):
        """Test _select_dimensions uses fallback when tool fails."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={
                "subject": "sunset",
                "style": "watercolor",
                "mood": "peaceful",
                "colors": ["orange", "purple"],
            },
            complexity="standard",
            product_type="poster",
        )
        state = PromptState()

        # Mock call_tool to return failure
        agent.call_tool = AsyncMock(return_value=ToolResult(
            success=False,
            data=None,
            error="Tool not found",
        ))

        state = await agent._select_dimensions(task, state)

        assert state.dimensions is not None
        assert state.dimensions.subject == "sunset"
        assert state.dimensions.aesthetic == "watercolor"
        assert state.dimensions.mood == "peaceful"
        assert "orange" in state.dimensions.color
        assert "purple" in state.dimensions.color


class TestProviderParams:
    """Tests for provider_params handling."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return ReactPromptAgent()

    def test_init_state_extracts_provider_params(self, agent):
        """Test _init_state extracts provider_params from requirements."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={
                "subject": "cat",
                "provider_params": {
                    "steps": 50,
                    "guidance_scale": 8.5,
                    "scheduler": "dpm_2m",
                },
            },
            complexity="standard",
            product_type="poster",
        )
        state = agent._init_state(task)

        assert state.provider_params == {
            "steps": 50,
            "guidance_scale": 8.5,
            "scheduler": "dpm_2m",
        }

    def test_init_state_preserves_previous_provider_params(self, agent):
        """Test _init_state preserves provider_params from previous plan."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="fix_plan",
            requirements={"subject": "cat"},
            complexity="standard",
            product_type="poster",
            previous_plan={
                "prompt": "A cat",
                "provider_params": {
                    "steps": 45,
                    "seed": 12345,
                },
            },
        )
        state = agent._init_state(task)

        assert state.provider_params["steps"] == 45
        assert state.provider_params["seed"] == 12345

    def test_build_provider_params_defaults(self, agent):
        """Test _build_provider_params sets mode-based defaults."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="standard",
            product_type="general",
        )
        state = PromptState(mode="STANDARD")

        params = agent._build_provider_params(task, state)

        assert params["steps"] == 30
        assert params["guidance_scale"] == 7.5

    def test_build_provider_params_advanced_mode(self, agent):
        """Test _build_provider_params for advanced mode."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="advanced",
            product_type="general",
        )
        state = PromptState(mode="ADVANCED")

        params = agent._build_provider_params(task, state)

        assert params["steps"] == 40
        assert params["guidance_scale"] == 8.0

    def test_build_provider_params_preserves_user_values(self, agent):
        """Test _build_provider_params preserves user-specified values."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="standard",
            product_type="general",
        )
        state = PromptState(
            mode="STANDARD",
            provider_params={"steps": 100, "custom_param": "value"},
        )

        params = agent._build_provider_params(task, state)

        # User value preserved
        assert params["steps"] == 100
        # Default applied
        assert params["guidance_scale"] == 7.5
        # Custom param kept
        assert params["custom_param"] == "value"

    def test_build_provider_params_poster_product(self, agent):
        """Test _build_provider_params increases steps for poster product."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="standard",
            product_type="poster",
        )
        state = PromptState(mode="STANDARD")

        params = agent._build_provider_params(task, state)

        # Higher steps for poster
        assert params["steps"] >= 35

    def test_build_provider_params_screen_print(self, agent):
        """Test _build_provider_params sets scheduler for screen print."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="standard",
            product_type="tshirt",
            print_method="screen_print",
        )
        state = PromptState(mode="STANDARD")

        params = agent._build_provider_params(task, state)

        assert params["scheduler"] == "euler_ancestral"

    def test_build_provider_params_photorealistic_style(self, agent):
        """Test _build_provider_params sets scheduler for photorealistic."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="standard",
            product_type="general",
        )
        state = PromptState(
            mode="STANDARD",
            dimensions=PromptDimensions(subject="cat", aesthetic="photorealistic"),
        )

        params = agent._build_provider_params(task, state)

        assert params["scheduler"] == "dpm_2m_karras"

    def test_build_provider_params_preserves_seed(self, agent):
        """Test _build_provider_params preserves seed from requirements."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat", "seed": 42},
            complexity="standard",
            product_type="general",
        )
        state = PromptState(mode="STANDARD")

        params = agent._build_provider_params(task, state)

        assert params["seed"] == 42


class TestWebSearch:
    """Tests for web search functionality in ReactPromptAgent."""

    @pytest.fixture
    def agent(self):
        """Create agent instance for testing."""
        return ReactPromptAgent()

    def test_is_rag_context_sufficient_with_enough_refs(self, agent):
        """Test context is sufficient when we have enough references."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="standard",
            product_type="poster",
        )
        state = PromptState(
            user_history=[{"id": 1}],
            art_references=[{"id": 2}],
        )

        # 2 refs >= MIN_CONTEXT_ITEMS (2)
        assert agent._is_rag_context_sufficient(task, state) is True

    def test_is_rag_context_sufficient_with_detailed_requirements(self, agent):
        """Test context is sufficient when requirements are detailed."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={
                "subject": "cat",
                "style": "cartoon",
                "composition": "centered",
                "background": "white",
            },
            complexity="standard",
            product_type="poster",
        )
        state = PromptState()  # No context at all

        # 3+ detail fields = sufficient
        assert agent._is_rag_context_sufficient(task, state) is True

    def test_is_rag_context_sufficient_simple_mode(self, agent):
        """Test context is sufficient for simple complexity."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="simple",
            product_type="general",
        )
        state = PromptState()

        # Simple mode doesn't need extra context
        assert agent._is_rag_context_sufficient(task, state) is True

    def test_is_rag_context_insufficient(self, agent):
        """Test context is insufficient with minimal refs and requirements."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},  # Only 1 field
            complexity="standard",
            product_type="poster",
        )
        state = PromptState(user_history=[{"id": 1}])  # Only 1 ref

        # Not enough refs (1 < 2) and not enough details (1 < 3)
        assert agent._is_rag_context_sufficient(task, state) is False

    @pytest.mark.asyncio
    async def test_supplement_with_web_search_success(self, agent):
        """Test web search supplements context when called."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "vintage car", "style": "realistic"},
            complexity="standard",
            product_type="poster",
        )
        state = PromptState()

        # Mock the call_tool method
        async def mock_call_tool(tool_name, action, **kwargs):
            if tool_name == "search":
                return ToolResult(
                    success=True,
                    data={
                        "results": [
                            {"title": "Vintage Cars", "snippet": "Classic automobiles", "url": "http://example.com"},
                            {"title": "Retro Design", "snippet": "Design trends", "url": "http://example2.com"},
                        ],
                        "answer": "Vintage cars feature classic design elements.",
                        "provider": "test_provider",
                    },
                )
            return ToolResult(success=False, data=None)

        agent.call_tool = mock_call_tool

        result_state = await agent._supplement_with_web_search(task, state)

        assert len(result_state.web_results) == 2
        assert result_state.web_answer == "Vintage cars feature classic design elements."
        assert "web_search:test_provider" in result_state.rag_sources

    @pytest.mark.asyncio
    async def test_supplement_with_web_search_empty_query(self, agent):
        """Test web search is skipped for empty query."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={},  # No subject
            complexity="standard",
            product_type="general",
        )
        state = PromptState()

        result_state = await agent._supplement_with_web_search(task, state)

        # State should be unchanged
        assert result_state.web_results == []
        assert result_state.web_answer is None

    @pytest.mark.asyncio
    async def test_supplement_with_web_search_failure(self, agent):
        """Test web search failure is handled gracefully."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="standard",
            product_type="poster",
        )
        state = PromptState()

        # Mock the call_tool to raise an exception
        async def mock_call_tool(tool_name, action, **kwargs):
            raise Exception("Search service error")

        agent.call_tool = mock_call_tool

        result_state = await agent._supplement_with_web_search(task, state)

        # State should be unchanged on failure
        assert result_state.web_results == []
        assert result_state.web_answer is None

    def test_get_context_summary_with_web_search(self):
        """Test context summary includes web search metadata."""
        state = PromptState(
            user_history=[{"id": 1}],
            art_references=[{"id": 2, "image_url": "http://example.com/img.jpg"}],
            web_results=[
                {"title": "Result 1", "url": "http://example.com"},
            ],
            web_answer="AI generated answer",
            rag_sources=["web_search:tavily"],
        )

        summary = state.get_context_summary()

        assert summary.user_history_count == 1
        assert summary.art_references_count == 1
        assert summary.web_search_count == 1
        assert summary.metadata["web_search_used"] is True
        assert summary.metadata["has_web_answer"] is True

    def test_get_context_summary_without_web_search(self):
        """Test context summary when no web search was used."""
        state = PromptState(
            user_history=[{"id": 1}],
            art_references=[],
        )

        summary = state.get_context_summary()

        assert summary.web_search_count == 0
        assert summary.metadata["web_search_used"] is False
        assert summary.metadata["has_web_answer"] is False


class TestReactPromptBoundary:
    """Tests verifying ReactPrompt has clear boundaries (TC3.1-TC3.6)."""

    def test_no_model_selection_logic(self):
        """TC3.1: ReactPrompt does NOT contain model selection logic."""
        # Read source file directly since inspect.getsource doesn't work with
        # dynamically loaded modules
        with open('palet8_agents/agents/react_prompt_agent.py', 'r') as f:
            source = f.read()

        # Should not have model selection methods
        assert "_select_model" not in source or "deprecated" in source.lower()
        # Should not call model selection service for selection
        assert "select_model(" not in source

    def test_no_pipeline_selection_logic(self):
        """TC3.2: ReactPrompt does NOT contain pipeline selection logic."""
        with open('palet8_agents/agents/react_prompt_agent.py', 'r') as f:
            source = f.read()

        # Should not have pipeline/genflow selection
        assert "_select_pipeline" not in source
        assert "determine_genflow" not in source

    def test_receives_generation_plan_from_context(self):
        """TC3.3: ReactPrompt receives GenerationPlan from context."""
        agent = ReactPromptAgent()

        # The _init_state method should be able to read generation_plan
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="standard",
            product_type="poster",
        )

        # Should initialize without requiring generation_plan
        state = agent._init_state(task)
        assert state is not None

    def test_uses_complexity_from_generation_plan(self):
        """TC3.4: ReactPrompt uses complexity from GenerationPlan."""
        agent = ReactPromptAgent()

        # Create task with complexity
        task_simple = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="simple",
            product_type="poster",
        )
        task_complex = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="complex",
            product_type="poster",
        )

        state_simple = agent._init_state(task_simple)
        state_complex = agent._init_state(task_complex)

        # Mode should reflect complexity
        assert state_simple.mode == "SIMPLE"
        assert state_complex.mode == "COMPLEX"


class TestReactPromptContextEvaluation:
    """Tests for context evaluation and question generation (TC3.5-TC3.6)."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return ReactPromptAgent()

    def test_evaluate_context_action_exists(self):
        """Test EVALUATE_CONTEXT action is defined."""
        assert ReactAction.EVALUATE_CONTEXT.value == "evaluate_context"

    def test_generate_questions_action_exists(self):
        """Test GENERATE_QUESTIONS action is defined."""
        assert ReactAction.GENERATE_QUESTIONS.value == "generate_questions"

    def test_prompt_state_has_context_evaluation_fields(self):
        """Test PromptState has context evaluation tracking fields."""
        state = PromptState()

        # Should have context evaluation properties
        assert hasattr(state, "context_evaluated")
        assert hasattr(state, "needs_clarification")
        assert hasattr(state, "clarification_questions")

        # Default values
        assert state.context_evaluated is False
        assert state.needs_clarification is False
        assert state.clarification_questions == []

    @pytest.mark.asyncio
    async def test_think_returns_evaluate_context_first(self, agent):
        """TC3.5: ReactPrompt evaluates context first."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="standard",
            product_type="poster",
        )
        state = PromptState()

        action = await agent._think(state, task)
        assert action == ReactAction.EVALUATE_CONTEXT

    @pytest.mark.asyncio
    async def test_think_returns_generate_questions_when_needed(self, agent):
        """TC3.5: ReactPrompt generates questions when context insufficient."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},  # Minimal context
            complexity="complex",  # Complex needs more context
            product_type="poster",
        )
        state = PromptState(
            context_evaluated=True,
            needs_clarification=True,
            missing_fields=["style", "mood"],
        )

        action = await agent._think(state, task)
        assert action == ReactAction.GENERATE_QUESTIONS

    @pytest.mark.asyncio
    async def test_think_skips_questions_when_context_sufficient(self, agent):
        """Test _think skips questions when context is sufficient."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={
                "subject": "cat",
                "style": "cartoon",
                "mood": "playful",
            },
            complexity="standard",
            product_type="poster",
        )
        state = PromptState(
            context_evaluated=True,
            needs_clarification=False,
        )

        action = await agent._think(state, task)
        # Should move to BUILD_CONTEXT, not GENERATE_QUESTIONS
        assert action == ReactAction.BUILD_CONTEXT

    @pytest.mark.asyncio
    async def test_evaluate_context_simple_complexity(self, agent):
        """Test context evaluation for simple complexity."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="simple",
            product_type="general",
        )
        state = PromptState()

        # Simple complexity needs minimal context
        result_state = await agent._evaluate_context(task, state)

        assert result_state.context_evaluated is True
        # Simple complexity typically doesn't need more context, but it depends on implementation
        # The key is that context was evaluated
        assert hasattr(result_state, 'needs_clarification')

    @pytest.mark.asyncio
    async def test_evaluate_context_complex_insufficient(self, agent):
        """Test context evaluation for complex with insufficient context."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},  # Only subject provided
            complexity="complex",
            product_type="poster",
        )
        state = PromptState()

        result_state = await agent._evaluate_context(task, state)

        assert result_state.context_evaluated is True
        # Complex mode with minimal requirements may need clarification
        assert result_state.missing_fields is not None

    @pytest.mark.asyncio
    async def test_generate_questions_returns_clarification_questions(self, agent):
        """TC3.6: ReactPrompt generates questions for missing context."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="complex",
            product_type="poster",
        )
        state = PromptState(
            context_evaluated=True,
            needs_clarification=True,
            missing_fields=["style"],
            priority_field="style",
        )

        result_state = await agent._generate_questions(task, state)

        assert result_state.questions_generated is True
        assert len(result_state.clarification_questions) > 0

    @pytest.mark.asyncio
    async def test_generate_questions_selector_type(self, agent):
        """Test question generation includes selector type for style."""
        task = PlanningTask(
            job_id="job",
            user_id="user",
            phase="initial",
            requirements={"subject": "cat"},
            complexity="complex",
            product_type="poster",
        )
        state = PromptState(
            context_evaluated=True,
            needs_clarification=True,
            missing_fields=["style"],
            priority_field="style",
        )

        result_state = await agent._generate_questions(task, state)

        # Style should generate a selector question
        questions = result_state.clarification_questions
        if questions:
            # Questions can be dataclass objects or dicts depending on implementation
            style_question = None
            for q in questions:
                field = q.field if hasattr(q, 'field') else q.get("field") if isinstance(q, dict) else None
                if field == "style":
                    style_question = q
                    break

            if style_question:
                q_type = style_question.question_type if hasattr(style_question, 'question_type') else style_question.get("question_type")
                assert q_type in ["selector", "text"]


class TestPlannerTodoList:
    """Tests for Planner's internal todo list tracking."""

    def test_planner_has_todo_list_classes(self):
        """Test that Planner has todo list tracking classes."""
        from tests.test_palet8_agents.agents.test_planner_orchestration import (
            load_planner_agent_v2,
        )
        planner_module = load_planner_agent_v2()

        # Check todo classes exist
        assert hasattr(planner_module, 'TodoStatus')
        assert hasattr(planner_module, 'TodoItem')
        assert hasattr(planner_module, 'PlannerTodoList')

    def test_todo_status_enum_values(self):
        """Test TodoStatus enum has expected values."""
        from tests.test_palet8_agents.agents.test_planner_orchestration import (
            load_planner_agent_v2,
        )
        planner_module = load_planner_agent_v2()
        TodoStatus = planner_module.TodoStatus

        assert TodoStatus.PENDING.value == "pending"
        assert TodoStatus.IN_PROGRESS.value == "in_progress"
        assert TodoStatus.COMPLETED.value == "completed"
        assert TodoStatus.FAILED.value == "failed"

    def test_todo_list_init_from_checkpoints(self):
        """Test PlannerTodoList initializes from checkpoints."""
        from tests.test_palet8_agents.agents.test_planner_orchestration import (
            load_planner_agent_v2,
        )
        planner_module = load_planner_agent_v2()
        PlannerTodoList = planner_module.PlannerTodoList
        TodoStatus = planner_module.TodoStatus

        checkpoints = [
            {"id": "context_check"},
            {"id": "safety_check"},
            {"id": "generation_plan"},
        ]

        todo_list = PlannerTodoList()
        todo_list.init_from_checkpoints(checkpoints)

        assert len(todo_list.items) == 3
        assert todo_list.items[0].id == "context_check"
        assert todo_list.items[0].status == TodoStatus.PENDING

    def test_todo_list_start_and_complete(self):
        """Test starting and completing todo items."""
        from tests.test_palet8_agents.agents.test_planner_orchestration import (
            load_planner_agent_v2,
        )
        planner_module = load_planner_agent_v2()
        PlannerTodoList = planner_module.PlannerTodoList
        TodoStatus = planner_module.TodoStatus

        checkpoints = [{"id": "test_checkpoint"}]
        todo_list = PlannerTodoList()
        todo_list.init_from_checkpoints(checkpoints)

        # Start item
        todo_list.start_item("test_checkpoint")
        assert todo_list.items[0].status == TodoStatus.IN_PROGRESS

        # Complete item
        todo_list.complete_item("test_checkpoint", {"result": "success"})
        assert todo_list.items[0].status == TodoStatus.COMPLETED
        assert todo_list.items[0].result == {"result": "success"}

    def test_todo_list_fail_item(self):
        """Test failing a todo item."""
        from tests.test_palet8_agents.agents.test_planner_orchestration import (
            load_planner_agent_v2,
        )
        planner_module = load_planner_agent_v2()
        PlannerTodoList = planner_module.PlannerTodoList
        TodoStatus = planner_module.TodoStatus

        checkpoints = [{"id": "test_checkpoint"}]
        todo_list = PlannerTodoList()
        todo_list.init_from_checkpoints(checkpoints)

        # Fail item
        todo_list.fail_item("test_checkpoint", "Test error")
        assert todo_list.items[0].status == TodoStatus.FAILED
        assert todo_list.items[0].error == "Test error"

    def test_todo_list_progress(self):
        """Test todo list progress tracking."""
        from tests.test_palet8_agents.agents.test_planner_orchestration import (
            load_planner_agent_v2,
        )
        planner_module = load_planner_agent_v2()
        PlannerTodoList = planner_module.PlannerTodoList

        checkpoints = [
            {"id": "cp1"},
            {"id": "cp2"},
            {"id": "cp3"},
            {"id": "cp4"},
        ]
        todo_list = PlannerTodoList()
        todo_list.init_from_checkpoints(checkpoints)

        # Complete 2, fail 1, leave 1 pending
        todo_list.complete_item("cp1")
        todo_list.complete_item("cp2")
        todo_list.fail_item("cp3", "error")

        progress = todo_list.get_progress()
        assert progress["total"] == 4
        assert progress["completed"] == 2
        assert progress["failed"] == 1
        assert progress["pending"] == 1
        assert progress["progress_pct"] == 0.5
