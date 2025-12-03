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
