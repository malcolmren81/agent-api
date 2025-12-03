"""Tests for new PR 4 tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestRequirementsTool:
    """Tests for RequirementsTool."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test tool initialization."""
        from palet8_agents.tools.requirements_tool import RequirementsTool

        tool = RequirementsTool()
        assert tool.name == "requirements"
        assert len(tool.parameters) > 0

    @pytest.mark.asyncio
    async def test_schema(self):
        """Test tool schema generation."""
        from palet8_agents.tools.requirements_tool import RequirementsTool

        tool = RequirementsTool()
        schema = tool.get_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "requirements"
        assert "action" in schema["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_invalid_action(self):
        """Test invalid action returns error."""
        from palet8_agents.tools.requirements_tool import RequirementsTool

        tool = RequirementsTool()
        result = await tool.execute(action="invalid_action")

        assert result.success is False
        assert "Unknown action" in result.error

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        from palet8_agents.tools.requirements_tool import RequirementsTool

        async with RequirementsTool() as tool:
            assert tool is not None


class TestDimensionTool:
    """Tests for DimensionTool."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test tool initialization."""
        from palet8_agents.tools.dimension_tool import DimensionTool

        tool = DimensionTool()
        assert tool.name == "dimension"
        assert len(tool.parameters) > 0

    @pytest.mark.asyncio
    async def test_get_required(self):
        """Test get_required action."""
        from palet8_agents.tools.dimension_tool import DimensionTool

        tool = DimensionTool()
        result = await tool.execute(action="get_required", mode="STANDARD")

        assert result.success is True
        assert "required_dimensions" in result.data
        assert "subject" in result.data["required_dimensions"]

    @pytest.mark.asyncio
    async def test_map_requirements(self):
        """Test map_requirements action."""
        from palet8_agents.tools.dimension_tool import DimensionTool

        tool = DimensionTool()
        result = await tool.execute(
            action="map_requirements",
            requirements={"subject": "A cat", "style": "cartoon"},
        )

        assert result.success is True
        assert "dimensions" in result.data


class TestModelSelectorTool:
    """Tests for ModelSelectorTool."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test tool initialization."""
        from palet8_agents.tools.model_selector_tool import ModelSelectorTool

        tool = ModelSelectorTool()
        assert tool.name == "model_selector"
        assert len(tool.parameters) > 0

    @pytest.mark.asyncio
    async def test_get_pipelines(self):
        """Test get_pipelines action."""
        from palet8_agents.tools.model_selector_tool import ModelSelectorTool

        tool = ModelSelectorTool()
        result = await tool.execute(action="get_pipelines")

        assert result.success is True
        assert "pipelines" in result.data
        assert "triggers" in result.data

    @pytest.mark.asyncio
    async def test_select_pipeline(self):
        """Test select_pipeline action."""
        from palet8_agents.tools.model_selector_tool import ModelSelectorTool

        tool = ModelSelectorTool()
        result = await tool.execute(
            action="select_pipeline",
            requirements={"subject": "A simple cat"},
            prompt="A simple cat illustration",
        )

        assert result.success is True
        assert "pipeline_type" in result.data


class TestPromptQualityTool:
    """Tests for PromptQualityTool."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test tool initialization."""
        from palet8_agents.tools.prompt_quality_tool import PromptQualityTool

        tool = PromptQualityTool()
        assert tool.name == "prompt_quality"
        assert len(tool.parameters) > 0

    @pytest.mark.asyncio
    async def test_assess_quality(self):
        """Test assess_quality action."""
        from palet8_agents.tools.prompt_quality_tool import PromptQualityTool

        tool = PromptQualityTool()
        result = await tool.execute(
            action="assess_quality",
            prompt="A beautiful mountain landscape at sunset",
            negative_prompt="blurry, low quality",
            mode="STANDARD",
            dimensions={"subject": "mountain landscape"},
        )

        assert result.success is True
        assert "overall" in result.data
        assert "dimensions" in result.data
        assert "decision" in result.data

    @pytest.mark.asyncio
    async def test_get_weights(self):
        """Test get_weights action."""
        from palet8_agents.tools.prompt_quality_tool import PromptQualityTool

        tool = PromptQualityTool()
        result = await tool.execute(action="get_weights", mode="COMPLEX")

        assert result.success is True
        assert "weights" in result.data


class TestImageEvaluationTool:
    """Tests for ImageEvaluationTool."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test tool initialization."""
        from palet8_agents.tools.image_evaluation_tool import ImageEvaluationTool

        tool = ImageEvaluationTool()
        assert tool.name == "image_evaluation"
        assert len(tool.parameters) > 0

    @pytest.mark.asyncio
    async def test_evaluate_image(self):
        """Test evaluate_image action."""
        from palet8_agents.tools.image_evaluation_tool import ImageEvaluationTool

        tool = ImageEvaluationTool()
        result = await tool.execute(
            action="evaluate_image",
            image_data={
                "width": 1024,
                "height": 1024,
                "sharpness_score": 0.8,
            },
            plan={
                "job_id": "test-123",
                "prompt": "A beautiful sunset",
                "mode": "STANDARD",
            },
        )

        assert result.success is True
        assert "overall" in result.data
        assert "decision" in result.data

    @pytest.mark.asyncio
    async def test_get_thresholds(self):
        """Test get_thresholds action."""
        from palet8_agents.tools.image_evaluation_tool import ImageEvaluationTool

        tool = ImageEvaluationTool()
        result = await tool.execute(action="get_thresholds", mode="COMPLEX")

        assert result.success is True
        assert "thresholds" in result.data


class TestSafetyTool:
    """Tests for SafetyTool."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test tool initialization."""
        from palet8_agents.tools.safety_tool import SafetyTool

        tool = SafetyTool()
        assert tool.name == "safety"
        assert len(tool.parameters) > 0

    @pytest.mark.asyncio
    async def test_classify_safe_content(self):
        """Test classify_content with safe content."""
        from palet8_agents.tools.safety_tool import SafetyTool

        tool = SafetyTool()
        result = await tool.execute(
            action="classify_content",
            text="A beautiful sunset over mountains",
            source="input",
        )

        assert result.success is True
        assert result.data["is_safe"] is True

    @pytest.mark.asyncio
    async def test_classify_unsafe_content(self):
        """Test classify_content with unsafe content."""
        from palet8_agents.tools.safety_tool import SafetyTool

        tool = SafetyTool()
        result = await tool.execute(
            action="classify_content",
            text="nude explicit content",
            source="input",
        )

        assert result.success is True
        assert result.data["is_safe"] is False

    @pytest.mark.asyncio
    async def test_get_severity_penalty(self):
        """Test get_severity_penalty action."""
        from palet8_agents.tools.safety_tool import SafetyTool

        tool = SafetyTool()
        result = await tool.execute(
            action="get_severity_penalty",
            severity="high",
        )

        assert result.success is True
        assert "penalty" in result.data


class TestMemoryTool:
    """Tests for MemoryTool."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test tool initialization."""
        from palet8_agents.tools.memory_tool import MemoryTool

        tool = MemoryTool()
        assert tool.name == "memory"
        assert len(tool.parameters) > 0

    @pytest.mark.asyncio
    async def test_store_missing_params(self):
        """Test store action with missing parameters."""
        from palet8_agents.tools.memory_tool import MemoryTool

        tool = MemoryTool()
        result = await tool.execute(action="store")

        assert result.success is False
        assert "job_id is required" in result.error

    @pytest.mark.asyncio
    async def test_search_prompts_missing_query(self):
        """Test search_prompts with missing query."""
        from palet8_agents.tools.memory_tool import MemoryTool

        tool = MemoryTool()
        result = await tool.execute(action="search_prompts")

        assert result.success is False
        assert "query is required" in result.error

    @pytest.mark.asyncio
    async def test_get_history_missing_user(self):
        """Test get_history with missing user_id."""
        from palet8_agents.tools.memory_tool import MemoryTool

        tool = MemoryTool()
        result = await tool.execute(action="get_history")

        assert result.success is False
        assert "user_id is required" in result.error
