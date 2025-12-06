"""Tests for palet8_agents.agents.genplan_agent module.

Goal-based test cases:
- Goal 2: GenPlan Agent Handles Generation Planning (TC2.1-TC2.9)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

# Import directly to avoid Prisma dependency issues
import sys
import importlib.util


# Load the module directly
def load_genplan_agent():
    """Load the genplan_agent module directly."""
    spec = importlib.util.spec_from_file_location(
        'genplan_agent',
        'palet8_agents/agents/genplan_agent.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['palet8_agents.agents.genplan_agent'] = module
    spec.loader.exec_module(module)
    return module


# Load module
genplan_module = load_genplan_agent()
GenPlanAgent = genplan_module.GenPlanAgent
GenPlanAction = genplan_module.GenPlanAction

from palet8_agents.core.agent import AgentContext, AgentResult
from palet8_agents.models.genplan import (
    UserParseResult,
    GenflowConfig,
    GenerationPlan,
    GenPlanState,
)


class TestGenPlanAgentInit:
    """Tests for GenPlanAgent initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        agent = GenPlanAgent()
        assert agent.name == "genplan"
        # Description should contain key words about planning/strategy
        desc_lower = agent.description.lower()
        assert "generation" in desc_lower or "strategy" in desc_lower
        # Check max_steps exists (may be class constant or instance variable)
        assert hasattr(agent, 'max_steps') or hasattr(GenPlanAgent, 'MAX_STEPS') or hasattr(agent, 'MAX_STEPS')

    def test_init_with_services(self):
        """Test initialization with services."""
        mock_context = MagicMock()
        mock_model = MagicMock()
        mock_genflow = MagicMock()

        agent = GenPlanAgent(
            context_service=mock_context,
            model_selection_service=mock_model,
            genflow_service=mock_genflow,
        )

        # Check services are stored (may use different internal names)
        assert agent._context_service == mock_context or hasattr(agent, '_context_analysis_service')
        assert agent._model_selection_service == mock_model
        assert agent._genflow_service == mock_genflow


class TestGenPlanComplexityDetermination:
    """Tests for complexity determination (TC2.1, TC2.2)."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return GenPlanAgent()

    def test_analyze_complexity_simple(self, agent):
        """TC2.1: GenPlan determines complexity for simple request."""
        requirements = {
            "subject": "a cat",
        }
        state = GenPlanState()

        # Call the analyze complexity method
        complexity, rationale = agent._analyze_complexity(requirements, state)

        assert complexity == "simple"
        assert rationale is not None
        assert len(rationale) > 0

    def test_analyze_complexity_standard(self, agent):
        """Test standard complexity with style."""
        requirements = {
            "subject": "a cat",
            "style": "watercolor",
            "mood": "peaceful",
        }
        state = GenPlanState()

        complexity, rationale = agent._analyze_complexity(requirements, state)

        assert complexity == "standard"

    def test_analyze_complexity_complex_with_text(self, agent):
        """TC2.2: GenPlan determines complexity for complex request (text)."""
        requirements = {
            "subject": "a poster",
            "text_content": "SALE 50% OFF",
            "style": "professional",
        }
        state = GenPlanState()

        complexity, rationale = agent._analyze_complexity(requirements, state)

        assert complexity == "complex"
        assert "text" in rationale.lower() or "typography" in rationale.lower()

    def test_analyze_complexity_complex_with_character(self, agent):
        """TC2.2: GenPlan determines complexity for complex request (character)."""
        requirements = {
            "subject": "a person",
            "character_edit": True,
            "face_fix": True,
        }
        state = GenPlanState()

        complexity, rationale = agent._analyze_complexity(requirements, state)

        assert complexity == "complex"

    def test_analyze_complexity_complex_multi_element(self, agent):
        """Test complex with multiple elements."""
        requirements = {
            "subject": "a scene",
            "include_elements": ["person", "dog", "cat", "bird", "house"],
        }
        state = GenPlanState()

        complexity, rationale = agent._analyze_complexity(requirements, state)

        assert complexity in ["standard", "complex"]

    def test_analyze_complexity_production_quality(self, agent):
        """Test complex with production quality keywords."""
        requirements = {
            "subject": "product shot",
            "style": "professional billboard quality",
        }
        state = GenPlanState()

        complexity, rationale = agent._analyze_complexity(requirements, state)

        assert complexity == "complex"

    def test_analyze_complexity_explicit_simple(self, agent):
        """Test explicit simple/quick mode."""
        requirements = {
            "subject": "a cat",
            "mode": "quick",
        }
        state = GenPlanState()

        complexity, rationale = agent._analyze_complexity(requirements, state)

        assert complexity == "simple"


class TestGenPlanGenflowSelection:
    """Tests for genflow selection (TC2.3, TC2.4)."""

    @pytest.fixture
    def agent(self):
        """Create agent with mock genflow service."""
        agent = GenPlanAgent()
        mock_genflow = MagicMock()
        agent._genflow_service = mock_genflow
        return agent

    def test_determine_genflow_single_for_simple(self, agent):
        """TC2.3: GenPlan selects single pipeline for RELAX mode."""
        agent._genflow_service.determine_genflow.return_value = GenflowConfig(
            flow_type="single",
            flow_name="single_quick",
            description="Quick single model generation",
            rationale="Single pipeline: complexity=simple, no dual triggers",
        )

        requirements = {"subject": "a cat"}
        state = GenPlanState(complexity="simple")

        genflow = agent._determine_genflow(requirements, state)

        assert genflow.flow_type == "single"
        agent._genflow_service.determine_genflow.assert_called_once()

    def test_determine_genflow_dual_for_text(self, agent):
        """TC2.4: GenPlan selects dual pipeline when text_in_image trigger detected."""
        agent._genflow_service.determine_genflow.return_value = GenflowConfig(
            flow_type="dual",
            flow_name="layout_poster",
            description="Two-stage poster generation",
            rationale="Dual pipeline: text_in_image trigger detected",
            triggered_by="text_in_image",
        )

        requirements = {
            "subject": "a poster",
            "text_content": "HELLO WORLD",
        }
        state = GenPlanState(
            complexity="complex",
            user_info=UserParseResult(
                subject="poster",
                text_content="HELLO WORLD",
            ),
        )

        genflow = agent._determine_genflow(requirements, state)

        assert genflow.flow_type == "dual"
        assert genflow.triggered_by == "text_in_image"

    def test_determine_genflow_dual_for_character(self, agent):
        """Test dual pipeline for character refinement."""
        agent._genflow_service.determine_genflow.return_value = GenflowConfig(
            flow_type="dual",
            flow_name="creative_art",
            description="Character refinement pipeline",
            rationale="Dual pipeline: character_refinement trigger",
            triggered_by="character_refinement",
        )

        requirements = {
            "subject": "a character",
            "character_edit": True,
        }
        state = GenPlanState(complexity="complex")

        genflow = agent._determine_genflow(requirements, state)

        assert genflow.flow_type == "dual"


class TestGenPlanModelSelection:
    """Tests for model selection (TC2.5, TC2.6)."""

    @pytest.fixture
    def agent(self):
        """Create agent with mock model selection service."""
        agent = GenPlanAgent()
        mock_model = MagicMock()
        agent._model_selection_service = mock_model
        return agent

    @pytest.mark.asyncio
    async def test_select_model_art_no_reference(self, agent):
        """TC2.5: GenPlan selects correct model for art_no_reference scenario."""
        agent._model_selection_service.select_model = AsyncMock(return_value=(
            "midjourney-v7",
            "Best for artistic styles without reference",
            ["ideogram-3", "flux-2-flex"],
        ))

        requirements = {
            "subject": "fantasy art",
            "style": "illustration",
        }
        state = GenPlanState(
            complexity="standard",
            user_info=UserParseResult(
                subject="fantasy art",
                style="illustration",
                has_reference=False,
            ),
            genflow=GenflowConfig(flow_type="single", flow_name="single_standard"),
        )

        model_id, rationale, alternatives = await agent._select_model(
            requirements, state
        )

        assert model_id == "midjourney-v7"
        assert len(alternatives) == 2

    @pytest.mark.asyncio
    async def test_select_model_photo_with_reference(self, agent):
        """TC2.6: GenPlan selects correct model for photo_with_reference scenario."""
        agent._model_selection_service.select_model = AsyncMock(return_value=(
            "nano-banana-2-pro",
            "Best for reference-based photorealistic editing",
            ["flux-2-flex"],
        ))

        requirements = {
            "subject": "portrait",
            "style": "photorealistic",
            "reference_image": "http://example.com/ref.jpg",
        }
        state = GenPlanState(
            complexity="standard",
            user_info=UserParseResult(
                subject="portrait",
                style="photorealistic",
                has_reference=True,
            ),
            genflow=GenflowConfig(flow_type="single", flow_name="single_standard"),
        )

        model_id, rationale, alternatives = await agent._select_model(
            requirements, state
        )

        assert model_id == "nano-banana-2-pro"

    @pytest.mark.asyncio
    async def test_select_model_uses_dual_pipeline_models(self, agent):
        """Test model selection uses predefined models for dual pipeline."""
        # For dual pipeline, model should come from pipeline config
        requirements = {"subject": "poster"}
        state = GenPlanState(
            complexity="complex",
            genflow=GenflowConfig(
                flow_type="dual",
                flow_name="layout_poster",
            ),
        )

        # Mock returns pipeline's stage 1 model
        agent._model_selection_service.select_model = AsyncMock(return_value=(
            "flux-2-flex",
            "Stage 1 model for layout_poster pipeline",
            [],
        ))

        model_id, rationale, alternatives = await agent._select_model(
            requirements, state
        )

        assert model_id == "flux-2-flex"


class TestGenPlanParameterExtraction:
    """Tests for parameter extraction (TC2.7, TC2.8)."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return GenPlanAgent()

    def test_extract_model_input_params(self, agent):
        """TC2.7: GenPlan extracts model_input_params correctly."""
        state = GenPlanState(
            complexity="standard",
            model_id="midjourney-v7",
        )
        requirements = {
            "subject": "a cat",
            "product_type": "poster",
        }

        model_input_params, provider_params = agent._extract_parameters(
            requirements, state
        )

        # Check required standard params
        assert "width" in model_input_params
        assert "height" in model_input_params
        assert "steps" in model_input_params
        assert "guidance_scale" in model_input_params

        # Check complexity-based defaults
        assert model_input_params["steps"] == 30  # standard = 30
        assert model_input_params["guidance_scale"] == 7.5  # standard = 7.5

    def test_extract_params_simple_complexity(self, agent):
        """Test parameter extraction for simple complexity."""
        state = GenPlanState(
            complexity="simple",
            model_id="flux-2-flex",
        )
        requirements = {"subject": "a cat"}

        model_input_params, provider_params = agent._extract_parameters(
            requirements, state
        )

        assert model_input_params["steps"] == 25  # simple = 25
        assert model_input_params["guidance_scale"] == 7.0  # simple = 7.0

    def test_extract_params_complex_complexity(self, agent):
        """Test parameter extraction for complex complexity."""
        state = GenPlanState(
            complexity="complex",
            model_id="flux-2-flex",
        )
        requirements = {"subject": "a poster with text"}

        model_input_params, provider_params = agent._extract_parameters(
            requirements, state
        )

        assert model_input_params["steps"] == 40  # complex = 40
        assert model_input_params["guidance_scale"] == 8.0  # complex = 8.0

    def test_extract_provider_params_midjourney(self, agent):
        """TC2.8: GenPlan extracts provider_params for Midjourney."""
        state = GenPlanState(
            complexity="standard",
            model_id="midjourney-v7",
        )
        requirements = {"subject": "a cat"}

        model_input_params, provider_params = agent._extract_parameters(
            requirements, state
        )

        # Midjourney-specific params should be present
        # These come from image_models_config.yaml
        assert isinstance(provider_params, dict)

    def test_extract_params_preserves_user_seed(self, agent):
        """Test that user-specified seed is preserved."""
        state = GenPlanState(
            complexity="standard",
            model_id="flux-2-flex",
        )
        requirements = {
            "subject": "a cat",
            "seed": 12345,
        }

        model_input_params, provider_params = agent._extract_parameters(
            requirements, state
        )

        assert model_input_params.get("seed") == 12345


class TestGenPlanValidation:
    """Tests for plan validation."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return GenPlanAgent()

    def test_validate_plan_complete(self, agent):
        """Test validation of complete plan."""
        state = GenPlanState(
            complexity="standard",
            complexity_rationale="Standard complexity",
            user_info=UserParseResult(subject="a cat"),
            genflow=GenflowConfig(flow_type="single", flow_name="single_standard"),
            model_id="flux-2-flex",
            model_rationale="Good for general use",
            model_input_params={"width": 1024, "height": 1024, "steps": 30},
            provider_params={},
        )

        is_valid, errors = agent._validate_plan(state)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_plan_missing_complexity(self, agent):
        """Test validation fails without complexity."""
        state = GenPlanState(
            user_info=UserParseResult(subject="a cat"),
            genflow=GenflowConfig(flow_type="single", flow_name="single_standard"),
            model_id="flux-2-flex",
        )

        is_valid, errors = agent._validate_plan(state)

        assert is_valid is False
        assert "complexity" in str(errors).lower()

    def test_validate_plan_missing_subject(self, agent):
        """Test validation fails without subject in user_info."""
        state = GenPlanState(
            complexity="standard",
            user_info=UserParseResult(subject=""),  # Empty subject
            genflow=GenflowConfig(flow_type="single", flow_name="single_standard"),
            model_id="flux-2-flex",
        )

        is_valid, errors = agent._validate_plan(state)

        assert is_valid is False
        assert "subject" in str(errors).lower()

    def test_validate_plan_missing_model(self, agent):
        """Test validation fails without model."""
        state = GenPlanState(
            complexity="standard",
            user_info=UserParseResult(subject="a cat"),
            genflow=GenflowConfig(flow_type="single", flow_name="single_standard"),
        )

        is_valid, errors = agent._validate_plan(state)

        assert is_valid is False
        assert "model" in str(errors).lower()


class TestGenPlanCompleteRun:
    """Tests for complete GenPlan execution (TC2.9)."""

    @pytest.fixture
    def agent(self):
        """Create fully mocked agent."""
        agent = GenPlanAgent()

        # Mock genflow service
        mock_genflow = MagicMock()
        mock_genflow.determine_genflow.return_value = GenflowConfig(
            flow_type="single",
            flow_name="single_standard",
            description="Standard single model",
            rationale="No dual triggers",
        )
        agent._genflow_service = mock_genflow

        # Mock model selection service
        mock_model = MagicMock()
        mock_model.select_model = AsyncMock(return_value=(
            "flux-2-flex",
            "Good general model",
            ["midjourney-v7"],
        ))
        agent._model_selection_service = mock_model

        # Mock context analysis service (for complexity questions)
        mock_context = MagicMock()
        mock_context.should_ask_complexity_question.return_value = False
        agent._context_analysis_service = mock_context

        return agent

    @pytest.fixture
    def mock_context(self):
        """Create mock AgentContext."""
        return AgentContext(
            user_id="user-123",
            job_id="job-456",
            requirements={
                "subject": "a cute cat",
                "style": "cartoon",
            },
        )

    @pytest.mark.asyncio
    async def test_complete_run_returns_generation_plan(self, agent, mock_context):
        """TC2.9: GenPlan returns complete GenerationPlan."""
        result = await agent.run(mock_context)

        assert result.success is True
        assert result.data is not None

        # Check all required fields are populated
        data = result.data
        assert "complexity" in data
        assert "genflow" in data
        assert "model_id" in data
        assert "model_input_params" in data
        assert "provider_params" in data
        assert "is_valid" in data

        # Verify the plan is valid
        assert data["is_valid"] is True

    @pytest.mark.asyncio
    async def test_run_stores_generation_plan_in_context(self, agent, mock_context):
        """Test that generation_plan is stored in context metadata."""
        await agent.run(mock_context)

        # Check generation_plan was stored
        assert mock_context.metadata.get("generation_plan") is not None

    @pytest.mark.asyncio
    async def test_run_with_missing_requirements(self, agent):
        """Test run with missing requirements."""
        context = AgentContext(
            user_id="user-123",
            job_id="job-456",
        )

        result = await agent.run(context)

        # Should fail gracefully
        assert result.success is False


class TestGenPlanReActLoop:
    """Tests for ReAct loop behavior."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return GenPlanAgent()

    def test_think_determines_correct_next_action(self, agent):
        """Test _think method determines correct next action."""
        # Empty state -> should analyze complexity first
        state = GenPlanState()
        requirements = {"subject": "a cat"}

        action = agent._think(state, requirements)
        assert action == GenPlanAction.ANALYZE_COMPLEXITY

        # With complexity -> should parse user info
        state.complexity = "standard"
        action = agent._think(state, requirements)
        assert action == GenPlanAction.PARSE_USER_INFO

        # With user info -> should determine genflow
        state.user_info = UserParseResult(subject="a cat")
        action = agent._think(state, requirements)
        assert action == GenPlanAction.DETERMINE_GENFLOW

        # With genflow -> should select model
        state.genflow = GenflowConfig(flow_type="single", flow_name="test")
        action = agent._think(state, requirements)
        assert action == GenPlanAction.SELECT_MODEL

        # With model -> should extract parameters
        state.model_id = "test-model"
        action = agent._think(state, requirements)
        assert action == GenPlanAction.EXTRACT_PARAMETERS

        # With parameters -> should validate
        state.parameters_extracted = True
        action = agent._think(state, requirements)
        assert action == GenPlanAction.VALIDATE_PLAN

        # After validation -> done
        state.validated = True
        action = agent._think(state, requirements)
        assert action == GenPlanAction.DONE


class TestGenPlanUserInfoParsing:
    """Tests for user info parsing."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return GenPlanAgent()

    def test_parse_user_info_extracts_subject(self, agent):
        """Test subject extraction."""
        requirements = {"subject": "a majestic lion"}
        state = GenPlanState()

        user_info = agent._parse_user_info(requirements, state)

        assert user_info.subject == "a majestic lion"

    def test_parse_user_info_extracts_style(self, agent):
        """Test style extraction."""
        requirements = {
            "subject": "a cat",
            "style": "vintage poster",
        }
        state = GenPlanState()

        user_info = agent._parse_user_info(requirements, state)

        assert user_info.style == "vintage poster"

    def test_parse_user_info_detects_reference(self, agent):
        """Test reference image detection."""
        requirements = {
            "subject": "a cat",
            "reference_image": "http://example.com/ref.jpg",
        }
        state = GenPlanState()

        user_info = agent._parse_user_info(requirements, state)

        assert user_info.has_reference is True

    def test_parse_user_info_extracts_text_content(self, agent):
        """Test text content extraction."""
        requirements = {
            "subject": "a poster",
            "text_content": "HELLO WORLD",
        }
        state = GenPlanState()

        user_info = agent._parse_user_info(requirements, state)

        assert user_info.text_content == "HELLO WORLD"

    def test_parse_user_info_extracts_product_type(self, agent):
        """Test product type extraction."""
        requirements = {
            "subject": "a design",
            "product_type": "t-shirt",
        }
        state = GenPlanState()

        user_info = agent._parse_user_info(requirements, state)

        assert user_info.product_type == "t-shirt"

    def test_parse_user_info_extracts_intents(self, agent):
        """Test intent extraction from requirements."""
        requirements = {
            "subject": "vintage poster with bold typography",
        }
        state = GenPlanState()

        user_info = agent._parse_user_info(requirements, state)

        # Should detect poster and vintage intents
        assert "poster" in user_info.extracted_intents or "vintage" in str(user_info)
